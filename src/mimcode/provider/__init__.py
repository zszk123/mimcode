"""mimcode.provider：协议即 provider 的双协议实现与 faux transport。

公共入口：
- ``OpenAIProtocolProvider`` / ``AnthropicProtocolProvider``：真实协议实现
- ``FauxOpenAIProvider`` / ``FauxAnthropicProvider``：测试/e2e 回放
- ``StreamProtocolError``：流协议违约（归一为流内 error 事件）
"""

from mimcode.provider.anthropic_protocol import (
    DEFAULT_THINKING_BUDGETS,
    MIN_ANSWER_TOKENS,
    AnthropicProtocolProvider,
    convert_anthropic_messages,
    thinking_budget_for_level,
    translate_anthropic_events,
)
from mimcode.provider.base import (
    Provider,
    StreamProtocolError,
    identity_index,
    make_partial_message,
    terminal_error_event,
)
from mimcode.provider.faux import (
    FauxAnthropicProvider,
    FauxFixture,
    FauxOpenAIProvider,
    load_fixture,
)
from mimcode.provider.openai_protocol import (
    OpenAIProtocolProvider,
    convert_openai_messages,
    translate_openai_chunks,
)

__all__ = [
    "AnthropicProtocolProvider",
    "DEFAULT_THINKING_BUDGETS",
    "FauxAnthropicProvider",
    "FauxFixture",
    "FauxOpenAIProvider",
    "MIN_ANSWER_TOKENS",
    "OpenAIProtocolProvider",
    "Provider",
    "StreamProtocolError",
    "convert_anthropic_messages",
    "convert_openai_messages",
    "identity_index",
    "load_fixture",
    "make_partial_message",
    "terminal_error_event",
    "thinking_budget_for_level",
    "translate_anthropic_events",
    "translate_openai_chunks",
]
