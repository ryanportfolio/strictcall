import pytest

from strictcall.llm import OPENROUTER_BASE_URL, get_chat_model


def test_default_is_anthropic():
    model = get_chat_model("claude-opus-5")
    assert type(model).__name__ == "ChatAnthropic"
    assert model.model == "claude-opus-5"


def test_openrouter_prefix_routes_through_openrouter(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    model = get_chat_model("openrouter:cohere/north-mini-code:free")
    assert type(model).__name__ == "ChatOpenAI"
    assert model.model_name == "cohere/north-mini-code:free"
    assert model.openai_api_base == OPENROUTER_BASE_URL


def test_openrouter_without_key_fails_clearly(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        get_chat_model("openrouter:some/model")


def test_env_var_picks_the_model(monkeypatch):
    monkeypatch.setenv("STRICTCALL_MODEL", "claude-haiku-4-5")
    assert get_chat_model().model == "claude-haiku-4-5"
