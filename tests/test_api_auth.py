import importlib

import pytest
from fastapi import HTTPException


@pytest.fixture
def api_module(monkeypatch, tmp_path):
    monkeypatch.setenv("EFFECTIVE_MEMORY_DB", str(tmp_path / "auth_test.db"))
    monkeypatch.setenv("EFFECTIVE_MEMORY_API_KEY", "expected-key")
    from effective_memory import api

    importlib.reload(api)
    yield api
    api.store.close()


def test_missing_key_rejected(api_module):
    with pytest.raises(HTTPException) as exc:
        api_module.require_api_key(None)
    assert exc.value.status_code == 401


def test_wrong_key_rejected(api_module):
    with pytest.raises(HTTPException) as exc:
        api_module.require_api_key("wrong-key")
    assert exc.value.status_code == 401


def test_correct_key_accepted(api_module):
    api_module.require_api_key("expected-key")  # should not raise


def test_auth_disabled_when_key_unset(monkeypatch, tmp_path):
    monkeypatch.setenv("EFFECTIVE_MEMORY_DB", str(tmp_path / "noauth_test.db"))
    monkeypatch.delenv("EFFECTIVE_MEMORY_API_KEY", raising=False)
    from effective_memory import api

    importlib.reload(api)
    try:
        api.require_api_key(None)  # should not raise -- auth disabled
    finally:
        api.store.close()


def test_multiple_keys_each_accepted_independently(monkeypatch, tmp_path):
    monkeypatch.setenv("EFFECTIVE_MEMORY_DB", str(tmp_path / "multikey_test.db"))
    monkeypatch.setenv("EFFECTIVE_MEMORY_API_KEY", "agent-one-key, agent-two-key")
    from effective_memory import api

    importlib.reload(api)
    try:
        assert api.API_KEYS == ["agent-one-key", "agent-two-key"]
        api.require_api_key("agent-one-key")  # should not raise
        api.require_api_key("agent-two-key")  # should not raise
        with pytest.raises(HTTPException) as exc:
            api.require_api_key("revoked-key")
        assert exc.value.status_code == 401
    finally:
        api.store.close()
