from importlib.resources import files
from typing import Literal

from pydantic import BaseModel
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)


class DataSettings(BaseModel):
    """Settings for data generation."""

    train_labels_path: str = "/home/lenni/projects/nlp/hw01/data/corpus1_train.labels"
    test_labels_path: str = "/home/lenni/projects/nlp/hw01/data/corpus1_test.labels"
    output_path: str = "/home/lenni/projects/nlp/hw01/output.txt"
    split: int = 0.2


class TrainingSettings(BaseModel):
    """Settings for model training."""

    mode: Literal["test", "eval"] = "test"


class AppSettings(BaseSettings):
    """Main application settings."""

    debug: bool = True
    random_seed: int = 31451
    data: DataSettings = DataSettings()
    training: TrainingSettings = TrainingSettings()

    model_config = SettingsConfigDict(
        toml_file=files("hw01").joinpath("config.toml"),
        env_nested_delimiter="__",
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """
        Set the priority of settings sources.

        We use a TOML file for configuration.
        """
        return (
            init_settings,
            TomlConfigSettingsSource(settings_cls),
            env_settings,
            dotenv_settings,
            file_secret_settings,
        )


def load_settings() -> AppSettings:
    """Load application settings."""
    return AppSettings()
