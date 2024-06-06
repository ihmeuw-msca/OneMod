"""Get kernel regression model uncertainty.

Copied from Peng's MSCA-284-copy/get-kreg-uncertainty.py

"""

import fire
import jax
import jax.numpy as jnp
import pandas as pd
from kreg.uncertainty.lanczos_sample import get_PC_inv_rootH
from scipy.special import expit
from tqdm.auto import tqdm

from onemod.modeling.uncertainty import calibrate_pred_sd
from onemod.utils import get_handle


def kreg_uncertainty(directory: str, submodel_id: str) -> None:
    """Get kernel regression model uncertainty

    Parameters
    ----------
    directory
        Experiment directory.
    submodel_id
        Submodel to run based on groupby settings.

    Outputs
    -------
    predictions.parquet
        Submodel predictions.

    """
    dataif, config = get_handle(directory)
    num_samples = config.kreg.kreg_uncertainty.num_samples
    lanczos_order = config.kreg.kreg_uncertainty.lanczos_order

    data = pd.merge(
        left=dataif.load_kreg(f"submodels/{submodel_id}/predictions.parquet"),
        right=dataif.load_data(
            columns=config.ids
            + ["region_id", config.obs, config.weights, config.test]
        ),
        on=config.ids,
        how="left",
    )
    model = dataif.load_kreg(f"submodels/{submodel_id}/model.pkl")
    op_hess = model.hessian(jnp.asarray(data["kreg_y"]))

    def op_root_pc(x):
        return model.kernel.op_root_k @ x

    def op_pced_hess(x):
        return model.kernel.op_root_k @ (op_hess(model.kernel.op_root_k @ x))

    sampler = jax.jit(get_PC_inv_rootH(op_pced_hess, op_root_pc, lanczos_order))

    moment = jnp.zeros(len(data))
    for i in tqdm(range(num_samples)):
        probe = jax.random.normal(jax.random.PRNGKey(i), (len(data),))
        moment += sampler(probe) ** 2

    data["kreg_y_sd"] = jnp.sqrt(moment / num_samples)
    data["kreg_linear"] = data.eval("kreg_y + offset")

    data["kreg_lwr"] = expit(data.eval("kreg_linear - 1.96 * kreg_y_sd"))
    data["kreg_upr"] = expit(data.eval("kreg_linear + 1.96 * kreg_y_sd"))

    # calibrate by region
    # TODO: hard coded by region
    data["cali_kreg_y_sd"] = data["kreg_y_sd"]
    data_group = data.groupby("region_id")

    for key, data_sub in data_group:
        cali_kreg_y_sd = calibrate_pred_sd(
            data_sub.query(f"{config.test} == 0"),
            config.mtype,
            "kreg_linear",
            "kreg_y_sd",
            config.obs,
            config.weights,
        )
        alpha = (
            cali_kreg_y_sd / data_sub.query(f"{config.test} == 0")["kreg_y_sd"]
        ).mean()
        index = data_group.groups[key]
        data.loc[index, "cali_kreg_y_sd"] = alpha * data.loc[index, "kreg_y_sd"]

    data["cali_kreg_lwr"] = expit(
        data.eval("kreg_linear - 1.96 * cali_kreg_y_sd")
    )
    data["cali_kreg_upr"] = expit(
        data.eval("kreg_linear + 1.96 * cali_kreg_y_sd")
    )

    dataif.dump_kreg(
        data[
            config.ids
            + [
                config.pred,
                "kreg_lwr",
                "kreg_upr",
                "cali_kreg_lwr",
                "cali_kreg_upr",
            ]
        ],
        f"submodels/{submodel_id}/predictions.parquet",
    )


def main() -> None:
    fire.Fire(kreg_uncertainty)
