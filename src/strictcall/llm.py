"""Chat model factory.

Provider is a config string, not a code change:

    STRICTCALL_MODEL=claude-opus-5                            # Anthropic (default)
    STRICTCALL_MODEL=openrouter:cohere/north-mini-code:free   # any OpenRouter model

Tests inject a scripted fake model instead, so the suite runs with no API key.
The Phase 3 Bedrock deployment will swap in langchain-aws ChatBedrockConverse
behind this same factory.
"""

import os

from langchain_core.language_models import BaseChatModel

DEFAULT_MODEL = "claude-opus-5"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def get_chat_model(model: str | None = None) -> BaseChatModel:
    name = model or os.environ.get("STRICTCALL_MODEL", DEFAULT_MODEL)

    if name.startswith("openrouter:"):
        from langchain_openai import ChatOpenAI

        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY is required for openrouter: models.")
        return ChatOpenAI(
            model=name.removeprefix("openrouter:"),
            base_url=OPENROUTER_BASE_URL,
            api_key=api_key,
            max_completion_tokens=2048,
            timeout=120,
        )

    from langchain_anthropic import ChatAnthropic

    return ChatAnthropic(model=name, max_tokens=2048, timeout=120)
