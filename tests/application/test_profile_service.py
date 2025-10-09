import json
from pathlib import Path

from application.profile_service import DEFAULT_PROFILE, ProfileService


def test_profile_service_persists_profile_with_encryption(tmp_path: Path) -> None:
    storage = tmp_path / "config.json"
    session: dict[str, object] = {}
    service = ProfileService(
        storage_path=storage,
        session_state=session,
        encryption_key="test-profile-key",
    )

    loaded = service.get_profile()
    assert loaded == DEFAULT_PROFILE

    updated = service.update_profile(
        risk_tolerance="alto",
        investment_horizon="largo",
        preferred_mode="max_return",
    )

    assert updated["risk_tolerance"] == "alto"
    assert session[ProfileService.SESSION_KEY]["preferred_mode"] == "max_return"

    content = json.loads(storage.read_text(encoding="utf-8"))
    encrypted_blob = content[ProfileService.STORAGE_KEY]
    assert isinstance(encrypted_blob, str)
    assert "max_return" not in encrypted_blob

    new_session: dict[str, object] = {}
    new_service = ProfileService(
        storage_path=storage,
        session_state=new_session,
        encryption_key="test-profile-key",
    )
    rehydrated = new_service.get_profile()
    assert rehydrated == updated


def test_profile_service_uses_encrypted_secrets(tmp_path: Path) -> None:
    template = ProfileService(
        storage_path=tmp_path / "template.json",
        session_state={},
        encryption_key="secret-seed",
    )
    payload = {
        "risk_tolerance": "bajo",
        "investment_horizon": "mediano",
        "preferred_mode": "diversify",
    }
    encrypted = template._cipher.encrypt(json.dumps(payload).encode("utf-8")).decode("utf-8")

    secrets = {ProfileService.STORAGE_KEY: encrypted}
    session: dict[str, object] = {}
    service = ProfileService(
        storage_path=tmp_path / "config.json",
        session_state=session,
        secrets=secrets,
        encryption_key="secret-seed",
    )

    profile = service.get_profile()
    assert profile == payload
    assert session[ProfileService.SESSION_KEY]["risk_tolerance"] == "bajo"
