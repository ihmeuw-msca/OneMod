from onemod.schema.models.base import ParametrizedBaseModel


class EnsembleConfiguration(ParametrizedBaseModel):
    groupby: list[str] = []
    max_attempts: int | None = None
    metric: str = "rmse"
    score: str = "neg_exp"
    top_pct_score: float = 1.0
    top_pct_model: float = 1.0

    def inherit(self) -> None:
        super().inherit(keys=["groupby", "max_attempts"])
