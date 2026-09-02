"""Typed settings loaded from config.yaml.

Usage::

    from src.config.settings import get_settings

    settings = get_settings()
    df = load_sessions(settings.paths.dataset_csv)

``get_settings()`` is cached, so the YAML is parsed once per process and
every caller (notebooks, API, dashboard, tests) shares the identical,
already-validated configuration object.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from src.utils.exceptions import ConfigError

logger = logging.getLogger(__name__)

# Repo root: three parents up from this file (src/config/settings.py -> repo root).
_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CONFIG_PATH = _REPO_ROOT / "src" / "config" / "config.yaml"


class PathsConfig(BaseModel):
    dataset_csv: Path
    data_dictionary_csv: Path
    models_dir: Path
    log_file: Path


class DataConfig(BaseModel):
    target_column: str
    numeric_columns: list[str]
    duration_columns: list[str]
    categorical_columns: list[str]
    int_coded_categorical_columns: list[str]


class ModelingConfig(BaseModel):
    test_size: float = Field(gt=0, lt=1)
    cv_folds: int = Field(gt=1)
    decision_threshold: float = Field(ge=0, le=1)


class MlflowConfig(BaseModel):
    experiment_name: str
    tracking_uri: str


class Settings(BaseModel):
    """Root settings object, validated on load."""

    random_seed: int
    paths: PathsConfig
    data: DataConfig
    modeling: ModelingConfig
    mlflow: MlflowConfig

    @property
    def all_feature_columns(self) -> list[str]:
        """Every input column (numeric + categorical), excluding the target."""
        return self.data.numeric_columns + self.data.categorical_columns


def _load_settings(config_path: Path) -> Settings:
    if not config_path.exists():
        raise ConfigError(f"Config file not found at {config_path}")

    try:
        with config_path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Could not parse YAML config at {config_path}: {exc}") from exc

    if not raw:
        raise ConfigError(f"Config file at {config_path} is empty")

    # Resolve relative paths against the repo root, so it doesn't matter
    # whether the caller is a notebook, a script, or the API.
    paths = raw.get("paths", {})
    for key, value in paths.items():
        paths[key] = str(_REPO_ROOT / value)
    raw["paths"] = paths

    try:
        settings = Settings.model_validate(raw)
    except Exception as exc:  # pydantic.ValidationError, kept generic in the except
        raise ConfigError(f"Invalid config at {config_path}: {exc}") from exc

    logger.debug("Loaded settings from %s", config_path)
    return settings


@lru_cache(maxsize=1)
def get_settings(config_path: str | Path = _DEFAULT_CONFIG_PATH) -> Settings:
    """Load and cache the project settings.

    Args:
        config_path: Path to the YAML config file. Defaults to
            ``src/config/config.yaml`` in the repo.

    Returns:
        A validated :class:`Settings` object.

    Raises:
        ConfigError: If the file is missing, malformed, or fails validation.
    """
    return _load_settings(Path(config_path))
