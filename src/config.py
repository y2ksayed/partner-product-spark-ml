"""Shared configuration for the partner product pipeline.

Everything the producer, streaming job and recommender need lives here,
so you only have to change values in one place. Environment variables
override the defaults, which makes it easy to run the same code locally
and in a container without editing anything.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

# Project root is the repo root (the parent of src/). Deriving paths from
# the file location means the code works no matter where you clone it.
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _env(name: str, default: str) -> str:
    """Read an env var, falling back to a default when it is not set."""
    return os.environ.get(name, default)


def _env_int(name: str, default: int) -> int:
    """Same as _env but parses the value as an integer."""
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    """Same as _env but parses the value as a float."""
    try:
        return float(os.environ.get(name, str(default)))
    except ValueError:
        return default


@dataclass
class Settings:
    """All pipeline settings in one place."""

    # Kafka connection details
    kafka_bootstrap: str = field(
        default_factory=lambda: _env("KAFKA_BOOTSTRAP", "localhost:9092")
    )
    events_topic: str = field(
        default_factory=lambda: _env("EVENTS_TOPIC", "partner-events")
    )

    # Spark settings
    app_name: str = field(
        default_factory=lambda: _env("SPARK_APP_NAME", "partner-product-pipeline")
    )
    spark_master: str = field(
        default_factory=lambda: _env("SPARK_MASTER", "local[*]")
    )
    shuffle_partitions: int = field(
        default_factory=lambda: _env_int("SPARK_SHUFFLE_PARTITIONS", 8)
    )

    # Streaming behavior
    watermark_delay: str = field(
        default_factory=lambda: _env("WATERMARK_DELAY", "10 minutes")
    )
    trigger_interval: str = field(
        default_factory=lambda: _env("TRIGGER_INTERVAL", "30 seconds")
    )
    max_offsets_per_trigger: int = field(
        default_factory=lambda: _env_int("MAX_OFFSETS_PER_TRIGGER", 5000)
    )

    # Output layout (bronze = raw, silver = cleaned/enriched)
    data_dir: Path = field(
        default_factory=lambda: Path(_env("DATA_DIR", str(PROJECT_ROOT / "data")))
    )
    checkpoint_dir: Path = field(
        default_factory=lambda: Path(
            _env("CHECKPOINT_DIR", str(PROJECT_ROOT / "data" / "checkpoints"))
        )
    )

    # Recommender hyperparameters
    als_rank: int = field(default_factory=lambda: _env_int("ALS_RANK", 20))
    als_max_iter: int = field(default_factory=lambda: _env_int("ALS_MAX_ITER", 15))
    als_reg_param: float = field(
        default_factory=lambda: _env_float("ALS_REG_PARAM", 0.1)
    )
    train_test_split: float = field(
        default_factory=lambda: _env_float("TRAIN_TEST_SPLIT", 0.8)
    )

    @property
    def bronze_path(self) -> str:
        return str(self.data_dir / "bronze" / "partner_events")

    @property
    def silver_path(self) -> str:
        return str(self.data_dir / "silver" / "partner_events")

    @property
    def model_dir(self) -> str:
        return str(PROJECT_ROOT / "models" / "als_recommender")


def ensure_dirs(settings: Settings) -> None:
    """Create the local data and checkpoint directories if missing."""
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.checkpoint_dir.mkdir(parents=True, exist_ok=True)
