from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class CheckMode(str, Enum):
    TOKEN = "token"
    CHUNK = "chunk"
    SENTENCE = "sentence"
    FULL = "full"


class WindowMode(str, Enum):
    PREFIX = "prefix"
    LOCAL = "local"


@dataclass
class Decision:
    label: str
    risk_score: Optional[float] = None
    latency_ms: float = 0.0

    @property
    def blocked(self) -> bool:
        return self.label.lower() == "unsafe"


@dataclass
class Event:
    event: str
    token_start: int
    token_end: int
    generated_tokens: int
    checked_tokens: int
    shown_tokens: int
    hidden_tokens: int
    decision: Optional[str] = None
    risk_score: Optional[float] = None
    latency_ms: Optional[float] = None
    text: str = ""


@dataclass
class StreamResult:
    response: str
    shown_text: str
    generated_tokens: int
    checked_tokens: int
    shown_tokens: int
    hidden_tokens: int
    blocked: bool
    first_block_token: Optional[int]
    events: list[Event] = field(default_factory=list)

    @property
    def leaked_tokens(self) -> int:
        return self.shown_tokens


class GuardAdapter:
    def reset(self, prompt: str = "") -> None:
        pass

    def score_prefix(
        self,
        prompt: str,
        response_prefix: str,
    ) -> Decision:
        raise NotImplementedError

    def score_local(
        self,
        prompt: str,
        text: str,
    ) -> Decision:
        return self.score_prefix(prompt, text)


class CallableGuard(GuardAdapter):
    def __init__(self, fn):
        self.fn = fn

    def score_prefix(
        self,
        prompt: str,
        response_prefix: str,
    ) -> Decision:
        result = self.fn(prompt, response_prefix)

        if isinstance(result, Decision):
            return result

        if isinstance(result, str):
            return Decision(label=result)

        if isinstance(result, dict):
            return Decision(
                label=result["label"],
                risk_score=result.get("risk_score"),
                latency_ms=result.get("latency_ms", 0.0),
            )

        raise TypeError("Guard must return Decision, str or dict")
