"""Tests for the shared _BaseModel Pydantic compatibility shim."""

from __future__ import annotations

from api.routers.base_models import _BaseModel


def _model_to_dict(model: _BaseModel) -> dict[str, object]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def test_base_model_ignores_extra_fields() -> None:
    """_BaseModel should ignore extra fields regardless of Pydantic version."""

    class SampleModel(_BaseModel):
        name: str

    instance = SampleModel(name="demo", extra_field="ignored")
    assert _model_to_dict(instance) == {"name": "demo"}


def test_base_model_exposes_ignore_config() -> None:
    """_BaseModel should expose extra="ignore" configuration."""

    config = getattr(_BaseModel, "model_config", None)
    if config is not None:
        # ConfigDict behaves like a mapping across supported versions.
        assert config.get("extra") == "ignore"
    else:
        legacy_config = getattr(_BaseModel, "Config")
        assert getattr(legacy_config, "extra") == "ignore"
