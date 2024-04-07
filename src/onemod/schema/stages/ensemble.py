from onemod.schema.base import StageConfig


class EnsembleConfig(StageConfig):
    """Configuration class for ensemble stage. All fields have default values."""

    metric: str = "rmse"
    score: str = "neg_exp"
    top_pct_score: float = 1.0
    top_pct_model: float = 1.0
