import numpy as np
from pandas import DataFrame


class Metric:
    """Metric function. Its instance is a callable function.

    Parameters
    ----------
    name
        The name of the metric

    Raises
    ------
    AttributeError
        Raised when the ``get_{name}`` function is not defined.

    """

    def __init__(self, name: str) -> None:
        if not hasattr(self, f"get_{name}"):
            raise AttributeError(f"'{name}' is not a valid metric")
        self.name = name

    @staticmethod
    def get_rmse(
        df: DataFrame, obs: str, pred: str, by: str | list[str] | None = None
    ) -> float | DataFrame:
        """Compute the RMSE between the observations and the predictions.

        Example
        -------
        >>> import pandas as pd
        >>> df = pd.DataFrame(
        ...     {
        ...         "id": [1, 1, 2, 2],
        ...         "obs": [0.1, 0.1, 0.2, 0.2],
        ...         "pred": [0.11, 0.12, 0.25, 0.18],
        ...     }
        ... )
        >>> metric = Metric("rmse")
        >>> metric(df, "obs", "pred")
        >>> 0.0291547594742265
        >>> metric(df, "obs", "pred", by="id")
        id      rmse
        0   1  0.015811
        1   2  0.038079

        """
        name = "rmse"
        by = _process_by(by)
        df = df[[obs, pred] + by].copy()
        df["rmse"] = (df[obs] - df[pred]) ** 2
        if by:
            return np.sqrt(df.groupby(by)[name].mean()).reset_index()
        return np.sqrt(df[name].mean())

    @staticmethod
    def get_winsorized_rmse(
        df: DataFrame,
        obs: str,
        pred: str,
        by: str | list[str] | None = None,
        percentile_lwr: float = 0.0,
        percentile_upr: float = 0.95,
    ) -> float | DataFrame:
        """Winsorized RMSE. Truncate errors to be between `percentile_lwr` and
        `percentile_upr` quantiles and then compute the RMSE.

        Example
        -------
        >>> import pandas as pd
        >>> df = pd.DataFrame(
        ...     {
        ...         "id": [1, 1, 1, 2, 2, 2],
        ...         "obs": [0.1, 0.1, 0.1, 0.2, 0.2, 0.2],
        ...         "pred": [0.11, 0.09, 0.12, 0.21, 0.19, 0.22],
        ...     }
        ... )
        >>> metric = Metric("winsorized_rmse")
        >>> metric(df, "obs", "pred", percentile_upr=0.5)
        >>> 0.01
        >>> metric(df, "obs", "pred", by="id", percentile_upr=0.5)
        id      rmse
        0   1  0.01
        1   2  0.01

        """
        name = "winsorized_rmse"
        by = _process_by(by)
        df = df[[obs, pred] + by].copy()
        df[name] = (df[obs] - df[pred]) ** 2
        if by:
            df["upr"] = df.groupby(by, sort=False)[name].transform(
                lambda x: x.quantile(percentile_upr)
            )
            df["lwr"] = df.groupby(by, sort=False)[name].transform(
                lambda x: x.quantile(percentile_lwr)
            )
        else:
            df["upr"] = df[name].quantile(percentile_upr)
            df["lwr"] = df[name].quantile(percentile_lwr)

        df[name] = np.maximum(df["lwr"], df[name])
        df[name] = np.minimum(df["upr"], df[name])

        if by:
            return np.sqrt(df.groupby(by)[name].mean()).reset_index()
        return np.sqrt(df[name].mean())

    def __call__(self, *args, **kwargs) -> float | DataFrame:
        return getattr(self, f"get_{self.name}")(*args, **kwargs)


def _process_by(by: str | list[str] | None) -> list[str]:
    if not by:
        return []
    if isinstance(by, str):
        return [by]
    return list(by)
