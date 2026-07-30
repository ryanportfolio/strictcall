"""Chat model factory.

Local runs use the Anthropic API (claude-opus-5 by default, override with
STRICTCALL_MODEL). Tests inject a scripted fake model instead, so the suite
runs with no API key. The Phase 3 Bedrock deployment will swap in
langchain-aws ChatBedrockConverse behind this same factory.
"""

import os

from langchain_core.language_models import BaseChatModel

DEFAULT_MODEL = "claude-opus-5"


def get_chat_model(model: str | None = None) -> BaseChatModel:
    from langchain_anthropic import ChatAnthropic

    return ChatAnthropic(
        model=model or os.environ.get("STRICTCALL_MODEL", DEFAULT_MODEL),
        max_tokens=2048,
        timeout=120,
    )
