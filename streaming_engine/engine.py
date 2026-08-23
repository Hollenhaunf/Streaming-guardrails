import re
import time
from typing import Optional

from streaming_engine.data_classes import (
    CheckMode,
    Decision,
    Event,
    GuardAdapter,
    StreamResult,
    WindowMode,
)


class StreamingEngine:
    def __init__(
        self,
        tokenizer,
        guard: GuardAdapter,
        mode: CheckMode | str = CheckMode.CHUNK,
        chunk_size: int = 16,
        window: WindowMode | str = WindowMode.PREFIX,
        max_sentence_tokens: int = 128,
        stop_on_unsafe: bool = True,
    ):
        self.tokenizer = tokenizer
        self.guard = guard
        self.mode = CheckMode(mode)
        self.chunk_size = chunk_size
        self.window = WindowMode(window)
        self.max_sentence_tokens = max_sentence_tokens
        self.stop_on_unsafe = stop_on_unsafe

        if self.mode == CheckMode.CHUNK and chunk_size not in (8, 16, 32):
            raise ValueError("chunk_size must be 8, 16 or 32")

    def _decode(self, token_ids: list[int]) -> str:
        if not token_ids:
            return ""

        return self.tokenizer.decode(
            token_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )

    def _sentence_boundary(self, text: str) -> bool:
        if not text:
            return False

        stripped = text.rstrip()

        if not stripped:
            return False

        return bool(
            re.search(
                r"""(?:[.!?。！？]+["'»”’)\]]*|\n\s*\n)$""",
                stripped,
            )
        )

    def _check(
        self,
        prompt: str,
        token_ids: list[int],
        start: int,
        end: int,
    ) -> tuple[Decision, str]:

        if self.window == WindowMode.PREFIX:
            ids = token_ids[:end]
        else:
            ids = token_ids[start:end]

        text = self._decode(ids)

        started = time.perf_counter()

        decision = self.guard.score_prefix(
            prompt,
            text,
        )

        elapsed = (time.perf_counter() - started) * 1000

        if decision.latency_ms == 0:
            decision.latency_ms = elapsed

        return decision, text

    def run(
        self,
        prompt: str,
        response: str,
        token_ids: Optional[list[int]] = None,
    ) -> StreamResult:

        if token_ids is None:
            token_ids = self.tokenizer.encode(
                response,
                add_special_tokens=False,
            )

        self.guard.reset(prompt)

        if self.mode == CheckMode.FULL:
            return self._run_full(
                prompt,
                response,
                token_ids,
            )

        events: list[Event] = []
        generated = 0
        checked = 0
        shown = 0
        blocked = False
        first_block = None
        sentence_start = 0

        while generated < len(token_ids):
            generated += 1
            should_check = False
            check_start = 0

            if self.mode == CheckMode.TOKEN:
                should_check = True
                check_start = generated - 1

            elif self.mode == CheckMode.CHUNK:
                should_check = generated % self.chunk_size == 0 or generated == len(
                    token_ids
                )
                check_start = max(
                    0,
                    generated - self.chunk_size,
                )

            elif self.mode == CheckMode.SENTENCE:
                sentence_text = self._decode(token_ids[sentence_start:generated])
                sentence_length = generated - sentence_start

                should_check = (
                    self._sentence_boundary(sentence_text)
                    or sentence_length >= self.max_sentence_tokens
                    or generated == len(token_ids)
                )

                check_start = sentence_start

            if not should_check:
                continue

            decision, checked_text = self._check(
                prompt,
                token_ids,
                check_start,
                generated,
            )

            checked = generated

            events.append(
                Event(
                    event="check",
                    token_start=check_start + 1,
                    token_end=generated,
                    generated_tokens=generated,
                    checked_tokens=checked,
                    shown_tokens=shown,
                    hidden_tokens=generated - shown,
                    decision=decision.label,
                    risk_score=decision.risk_score,
                    latency_ms=decision.latency_ms,
                    text=checked_text,
                )
            )

            if decision.blocked and self.stop_on_unsafe:
                blocked = True
                first_block = generated

                events.append(
                    Event(
                        event="block",
                        token_start=shown + 1,
                        token_end=generated,
                        generated_tokens=generated,
                        checked_tokens=checked,
                        shown_tokens=shown,
                        hidden_tokens=generated - shown,
                        decision=decision.label,
                        risk_score=decision.risk_score,
                        latency_ms=decision.latency_ms,
                        text="",
                    )
                )

                break

            if self.mode == CheckMode.SENTENCE:
                sentence_start = generated

            if generated > shown:
                shown_start = shown
                shown_end = generated
                shown_text = self._decode(token_ids[shown_start:shown_end])

                shown = generated

                events.append(
                    Event(
                        event="show",
                        token_start=shown_start + 1,
                        token_end=shown_end,
                        generated_tokens=generated,
                        checked_tokens=checked,
                        shown_tokens=shown,
                        hidden_tokens=generated - shown,
                        text=shown_text,
                    )
                )

        shown_text = self._decode(token_ids[:shown])

        return StreamResult(
            response=response,
            shown_text=shown_text,
            generated_tokens=generated,
            checked_tokens=checked,
            shown_tokens=shown,
            hidden_tokens=generated - shown,
            blocked=blocked,
            first_block_token=first_block,
            events=events,
        )

    def _run_full(
        self,
        prompt: str,
        response: str,
        token_ids: list[int],
    ) -> StreamResult:

        started = time.perf_counter()

        decision = self.guard.score_prefix(
            prompt,
            response,
        )

        latency = (time.perf_counter() - started) * 1000

        if decision.latency_ms == 0:
            decision.latency_ms = latency

        n = len(token_ids)
        blocked = decision.blocked and self.stop_on_unsafe
        shown = 0 if blocked else n

        events = [
            Event(
                event="check",
                token_start=1,
                token_end=n,
                generated_tokens=n,
                checked_tokens=n,
                shown_tokens=0,
                hidden_tokens=n,
                decision=decision.label,
                risk_score=decision.risk_score,
                latency_ms=decision.latency_ms,
                text=response,
            )
        ]

        if blocked:
            events.append(
                Event(
                    event="block",
                    token_start=1,
                    token_end=n,
                    generated_tokens=n,
                    checked_tokens=n,
                    shown_tokens=0,
                    hidden_tokens=n,
                    decision=decision.label,
                    risk_score=decision.risk_score,
                    latency_ms=decision.latency_ms,
                )
            )
            shown_text = ""
        else:
            events.append(
                Event(
                    event="show",
                    token_start=1,
                    token_end=n,
                    generated_tokens=n,
                    checked_tokens=n,
                    shown_tokens=n,
                    hidden_tokens=0,
                    text=response,
                )
            )
            shown_text = response

        return StreamResult(
            response=response,
            shown_text=shown_text,
            generated_tokens=n,
            checked_tokens=n,
            shown_tokens=shown,
            hidden_tokens=n - shown,
            blocked=blocked,
            first_block_token=n if blocked else None,
            events=events,
        )
