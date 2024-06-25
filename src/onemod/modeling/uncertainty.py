import numpy as np
import pandas as pd
from msca.c2fun import c2fun_dict
from scipy.optimize import brentq
from scipy.stats import norm

from onemod.modeling.residual import ResidualCalculator

_inv_link_funs = {
    "binomial": c2fun_dict["expit"],
    "poisson": c2fun_dict["exp"],
    "gaussian": c2fun_dict["identity"],
}


def get_ci_coverage(
    data: pd.DataFrame, pred: str, pred_sd: str, truth: str, alpha: float = 0.05
) -> float:
    """Calculate the coverage of the confidence interval (CI) for the
    prediction. This should only be used when truth are avaliable. Both the
    prediction and the truth has to be in linear space.

    Parameters
    ----------
    data
        The data frame containing the prediction, the standard deviation of the
        prediction, and the truth.
    pred
        The column name of the prediction.
    pred_sd
        The column name of the standard deviation of the prediction.
    truth
        The column name of the truth.
    alpha
        The significance level, by default 0.05.

    Returns
    -------
    float
        The coverage of the CI for the prediction.

    """
    data = data[[pred, pred_sd, truth]].copy()

    lwr = 0.5 * alpha
    upr = 1.0 - lwr
    data["lwr"] = norm.ppf(lwr, loc=data[pred], scale=data[pred_sd])
    data["upr"] = norm.ppf(upr, loc=data[pred], scale=data[pred_sd])

    coverage = data.eval(f"{truth} >= lwr and {truth} <= upr").mean()
    return coverage


def get_pi_coverage(
    data: pd.DataFrame,
    model_type: str,
    pred: str,
    pred_sd: str,
    obs: str,
    weights: str,
    alpha: float = 0.05,
) -> float:
    """Calculate the coverage of the prediction interval (PI). Predictions and
    the their standard deviations should be in the linear space. Observation
    and weights should be in the natural space. The prediction interval is for
    GLM is tricky and is computed with residual as a approximation.

    Parameters
    ----------
    data
        The data frame containing the prediction, the standard deviation of the
        prediction, the observation, and the weights.
    model_type
        The type of the model. Right now only choice from 'binomial', 'poisson',
        or 'gaussian'.
    pred
        The column name of the prediction.
    pred_sd
        The column name of the standard deviation of the prediction.
    obs
        The column name of the observation.
    weights
        The column name of the weights.
    alpha
        The significance level, by default 0.05.

    Returns
    -------
    float
        The coverage of the PI for the prediction.

    """
    data = data[[pred, pred_sd, obs, weights]].copy()

    get_residual = ResidualCalculator(model_type)
    data[f"pred_{obs}"] = _inv_link_funs[model_type](data[pred])

    residual = get_residual(data, f"pred_{obs}", obs, weights)
    residual["total_sd"] = np.sqrt(
        residual["residual_se"] ** 2 + data[pred_sd] ** 2
    )

    lwr = 0.5 * alpha
    upr = 1.0 - lwr
    residual["lwr"] = norm.ppf(lwr, loc=0.0, scale=residual["total_sd"])
    residual["upr"] = norm.ppf(upr, loc=0.0, scale=residual["total_sd"])
    coverage = residual.eval("residual >= lwr and residual <= upr").mean()
    return coverage


def calibrate_pred_sd(
    data: pd.DataFrame,
    model_type: str,
    pred: str,
    pred_sd: str,
    obs: str,
    weights: str,
    inflate: bool = True,
) -> pd.Series:
    """Calibrate the reported prediction standard deviation in the linear space.
    The goal is to make the Pearson residual in the transformed residual space
    to be 1.

    Parameters
    ----------
    data
        The data frame containing the prediction, the standard deviation of the
        prediction, the observation, and the weights.
    model_type
        The type of the model. Right now only choice from 'binomial', 'poisson',
        or 'gaussian'.
    pred
        The column name of the prediction.
    pred_sd
        The column name of the standard deviation of the prediction.
    obs
        The column name of the observation.
    weights
        The column name of the weights.
    inflate
        If true, always inflate the reported prediction standard deviation and
        ignore the case when the correction factor is less than 1. By default
        True.

    """
    data = data[[pred, pred_sd, obs, weights]].copy()

    get_residual = ResidualCalculator(model_type)
    data[f"pred_{obs}"] = _inv_link_funs[model_type](data[pred])

    residual = get_residual(data, f"pred_{obs}", obs, weights)

    # we want to find an alpha such that the equation is 0
    def equation(alpha: float) -> float:
        pearson_residual = residual["residual"] / np.sqrt(
            residual["residual_se"] ** 2 + (alpha * data[pred_sd]) ** 2
        )
        return pearson_residual.std() - 1.0

    if equation(0.0) > 0:
        # Here we assume all of data[pred_sd] are strictly greater than 0
        # We want to find the upper bound of alpha, such that the equation is
        # less than 0.
        # By Popoviciu's inequality, we know that a random variables standard
        # deviation is bounded by the range of the random variable divided by 2.
        # So, we want to find an alpha such that the maximum absolute value of
        # the Person residual is less than 1.
        alpha_upr = 1.1 * np.sqrt(
            np.max(
                residual.eval("residual ** 2 - residual_se ** 2")
                / data[pred_sd] ** 2
            )
        )
        alpha = brentq(equation, 0.0, alpha_upr)
    else:
        alpha = 0.0

    if inflate:
        alpha = max(alpha, 1.0)

    return alpha * data[pred_sd]
