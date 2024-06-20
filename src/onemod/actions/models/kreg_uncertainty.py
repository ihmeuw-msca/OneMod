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

    # Load data
    data = pd.merge(
        left=dataif.load_kreg(f"submodels/{submodel_id}/predictions.parquet"),
        right=dataif.load_data(
            columns=config.ids
            + [
                "super_region_id",
                "region_id",
                "age_mid",
                config.obs,
                config.weights,
                config.test,
            ]
        ),
        on=config.ids,
        how="left",
    ).sort_values(
        by=[
            "super_region_id",
            "region_id",
            "location_id",
            "age_mid",
            "year_id",
        ],
        ignore_index=True,
    )

    # Zero test data, rescale age, add offset
    # TODO: generalize offset for different model types
    index = data.eval(f"{config.test} == 1")
    data.loc[index, config.obs] = 0.0
    data.loc[index, config.weights] = 0.0
    age_scale = config.kreg.kreg_model.age_scale
    data["transformed_age"] = data.eval(
        "log(exp(@age_scale * age_mid) - 1) / @age_scale"
    )
    data["offset"] = data.eval(f"log({config.pred} / (1 - {config.pred}))")

    # Load kernel regression model
    model = dataif.load_kreg(f"submodels/{submodel_id}/model.pkl")
    model.likelihood.attach(data)
    op_hess = model.hessian(jnp.asarray(data["kreg_y"]))

    def op_root_pc(x):
        return model.kernel.op_root_k @ x

    def op_pced_hess(x):
        return model.kernel.op_root_k @ (op_hess(model.kernel.op_root_k @ x))

    sampler = jax.jit(get_PC_inv_rootH(op_pced_hess, op_root_pc, lanczos_order))
    
    moment = jnp.zeros(len(data))
    error_draw_columns = [f'kreg_error_draw_{i}' for i in tqdm(range(num_samples))]
    for i in tqdm(range(num_samples)):
        probe = jax.random.normal(jax.random.PRNGKey(i), (len(data),))
        sample = sampler(probe)
        data[f'kreg_error_draw_{i}'] = sample
        moment += sample ** 2

    data["kreg_y_sd"] = jnp.sqrt(moment / num_samples)
    data["kreg_linear"] = data.eval("kreg_y + offset")

    data["kreg_lwr"] = expit(data.eval("kreg_linear - 1.96 * kreg_y_sd"))
    data["kreg_upr"] = expit(data.eval("kreg_linear + 1.96 * kreg_y_sd"))

    # calibrate by region
    # TODO: hard coded by region
    data["cali_kreg_y_sd"] = data["kreg_y_sd"]
    data_group = data.groupby("region_id")
    cali_draw_cols = [f'cali_kreg_draw_{i}' for i in tqdm(range(num_samples))]

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
        data.loc[index, "cali_kreg_y_sd"] = alpha * data.loc[index, "kreg_y_sd"]
        data.loc[index,cali_draw_cols] = data.loc[index,'kreg_linear'] + alpha * data.loc[index,error_draw_columns]


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
            ] + cali_draw_cols
        ],
        f"submodels/{submodel_id}/predictions.parquet",
    )


def main() -> None:
    fire.Fire(kreg_uncertainty)
