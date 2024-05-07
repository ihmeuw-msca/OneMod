import pandas as pd


class ResidualCalculator:
    def __init__(self, model_type: str) -> None:
        if not hasattr(self, f"get_residual_{model_type}"):
            raise AttributeError(f"'{model_type}' is not a valid model type")
        self.model_type = model_type

    @staticmethod
    def get_residual_binomial(
        data: pd.DataFrame, pred: str, obs: str, weights: str
    ) -> pd.DataFrame:
        result = pd.DataFrame(index=data.index)
        result["residual"] = data.eval(
            f"({obs} - {pred}) / ({pred} * (1 - {pred}))"
        )
        result["residual_se"] = data.eval(
            f"1 / sqrt({pred} * (1 - {pred}) * {weights})"
        )
        return result

    @staticmethod
    def get_residual_poisson(
        data: pd.DataFrame, pred: str, obs: str, weights: str
    ) -> pd.DataFrame:
        result = pd.DataFrame(index=data.index)
        result["residual"] = data.eval(f"{obs} / {pred} - 1")
        result["residual_se"] = data.eval(f"1 / sqrt({pred} * {weights})")
        return result

    @staticmethod
    def get_residual_gaussian(
        data: pd.DataFrame, pred: str, obs: str, weights: str
    ) -> pd.DataFrame:
        result = pd.DataFrame(index=data.index)
        result["residual"] = data.eval(f"{obs} - {pred}")
        result["residual_se"] = data.eval(f"1 / sqrt({weights})")
        return result

    def __call__(self, *args, **kwargs) -> pd.DataFrame:
        return getattr(self, f"get_residual_{self.model_type}")(*args, **kwargs)
