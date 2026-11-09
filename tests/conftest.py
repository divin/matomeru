"""Shared test fixtures: keep the suite hermetic regardless of the host machine."""

import pytest


@pytest.fixture(autouse=True)
def _no_dotenv(monkeypatch: pytest.MonkeyPatch) -> None:
    """Never load the developer's real .env file during tests.

    load_runtime_config() calls load_dotenv() to support .env files in
    production. Tests must be hermetic: they set exactly the environment
    they need, so the real .env (present on configured machines) must not
    leak required variables or overrides into test expectations.
    """
    monkeypatch.setattr("matomeru.config.load_dotenv", lambda *args, **kwargs: None)
