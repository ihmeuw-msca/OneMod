import numpy as np
import pandas as pd


class ResidualCalculator:
    def __init__(self, model_type: str) -> None:
        if not hasattr(self, f"get_residual_{model_type}"):
            raise AttributeError(f"'{model_type}' is not a valid model type")
        self.model_type = model_type
        self.get_residual = getattr(self, f"get_residual_{model_type}")
        self.predict = getattr(self, f"predict_{model_type}")

    @staticmethod
    def get_residual_binomial(
        data: pd.DataFrame, pred: str, obs: str, weights: str
    ) -> pd.DataFrame:
        result = pd.DataFrame(index=data.index)
        result["residual"] = (data[obs] - data[pred]) / (
            data[pred] * (1 - data[pred])
        )
        result["residual_se"] = 1 / np.sqrt(
            data[pred] * (1 - data[pred]) * data[weights]
        )
        return result

    @staticmethod
    def predict_binomial(
        data: pd.DataFrame, pred: str, residual: str = "residual"
    ) -> pd.Series:
        return data[pred] + data[residual] * data[pred] * (1 - data[pred])

    @staticmethod
    def get_residual_poisson(
        data: pd.DataFrame, pred: str, obs: str, weights: str
    ) -> pd.DataFrame:
        result = pd.DataFrame(index=data.index)
        result["residual"] = data[obs] / data[pred] - 1
        result["residual_se"] = 1 / np.sqrt(data[pred] * data[weights])
        return result

    @staticmethod
    def predict_poisson(
        data: pd.DataFrame, pred: str, residual: str = "residual"
    ) -> pd.Series:
        return (data[residual] + 1) * data[pred]

    @staticmethod
    def get_residual_gaussian(
        data: pd.DataFrame, pred: str, obs: str, weights: str
    ) -> pd.DataFrame:
        result = pd.DataFrame(index=data.index)
        result["residual"] = data[obs] - data[pred]
        result["residual_se"] = 1 / np.sqrt(data[weights])
        return result

    @staticmethod
    def predict_gaussian(
        data: pd.DataFrame, pred: str, residual: str = "residual"
    ) -> pd.Series:
        return data[pred] + data[residual]

    def __call__(self, *args, **kwargs) -> pd.DataFrame:
        return self.get_residual(*args, **kwargs)
