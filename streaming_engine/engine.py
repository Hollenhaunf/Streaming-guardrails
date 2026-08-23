import logging
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
import pysbd

logger = logging.getLogger(__name__)


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
        self.sentence_segmenter = pysbd.Segmenter(
            language="en",
            clean=False,
        )
        self.stop_on_unsafe = stop_on_unsafe

        logger.debug(
            "StreamingEngine.__init__: mode=%s chunk_size=%s window=%s max_sentence_tokens=%s stop_on_unsafe=%s",
            self.mode,
            self.chunk_size,
            self.window,
            self.max_sentence_tokens,
            self.stop_on_unsafe,
        )

        if self.mode == CheckMode.CHUNK and chunk_size not in (8, 16, 32):
            raise ValueError("chunk_size must be 8, 16 or 32")

    def _decode(self, token_ids: list[int]) -> str:

        if not token_ids:
            return ""

        text = self.tokenizer.decode(
            token_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )

        logger.debug(
            "_decode: token_count=%d ids=%s text=%r",
            len(token_ids),
            token_ids,
            text,
        )

        return text

    # def _sentence_boundary(self, text: str) -> bool:

    #     if not text:
    #         return False

    #     stripped = text.rstrip()

    #     if not stripped:
    #         return False

    #     result = bool(
    #         re.search(
    #             r"""(?:[.!?。！？]+["'»”’)**\\]**]\*|**\n**\s\***\n**)$""",
    #             stripped,
    #         )
    #     )

    #     logger.debug(
    #         "_sentence_boundary: text=%r result=%s",
    #         text,
    #         result,
    #     )

    #     return result
    def _sentence_boundary(self, text: str) -> bool:
        if not text:
            return False

        stripped = text.rstrip()

        if not stripped:
            return False

        segments = self.sentence_segmenter.segment(stripped)

        if len(segments) <= 1:
            return False

        last_segment = segments[-1].strip()

        if not last_segment:
            return True

        if last_segment == stripped:
            return False

        result = True

        logger.debug(
            "_sentence_boundary: text=%r segments=%r result=%s",
            text,
            segments,
            result,
        )

        return result

    def _check(
        self,
        prompt: str,
        token_ids: list[int],
        start: int,
        end: int,
    ) -> tuple[Decision, str]:

        logger.debug(
            "_check START: window=%s mode=%s start=%d end=%d total_tokens=%d",
            self.window,
            self.mode,
            start,
            end,
            len(token_ids),
        )

        if self.window == WindowMode.PREFIX:
            ids = token_ids[:end]
        else:
            ids = token_ids[start:end]

        logger.debug(
            "_check IDS: selected_count=%d selected_ids=%s",
            len(ids),
            ids,
        )

        text = self._decode(ids)

        logger.debug(
            "_check TEXT: %r",
            text,
        )

        started = time.perf_counter()

        if self.window == WindowMode.PREFIX:

            logger.debug("_check: calling guard.score_prefix(prompt, text)")

            decision = self.guard.score_prefix(
                prompt,
                text,
            )

        else:

            logger.debug("_check: calling guard.score_local(prompt, text)")

            decision = self.guard.score_local(
                prompt,
                text,
            )

        elapsed = (time.perf_counter() - started) * 1000

        logger.debug(
            "_check RESULT: label=%s risk=%s latency=%s measured=%.3f ms text=%r",
            decision.label,
            decision.risk_score,
            decision.latency_ms,
            elapsed,
            text,
        )

        if decision.latency_ms == 0:
            decision.latency_ms = elapsed

        return decision, text

    def run(
        self,
        prompt: str,
        response: str,
        token_ids: Optional[list[int]] = None,
    ) -> StreamResult:

        logger.debug(
            "RUN START: prompt=%r response=%r mode=%s window=%s chunk_size=%s stop_on_unsafe=%s",
            prompt,
            response,
            self.mode,
            self.window,
            self.chunk_size,
            self.stop_on_unsafe,
        )

        if token_ids is None:
            token_ids = self.tokenizer.encode(
                response,
                add_special_tokens=False,
            )

        logger.debug(
            "RUN TOKENS: count=%d ids=%s",
            len(token_ids),
            token_ids,
        )

        self.guard.reset(prompt)

        logger.debug("RUN: guard.reset completed")

        if self.mode == CheckMode.FULL:

            logger.debug("RUN: entering FULL mode")

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

            logger.debug(
                "RUN LOOP: generated=%d/%d should_check=%s check_start=%d shown=%d checked=%d",
                generated,
                len(token_ids),
                should_check,
                check_start,
                shown,
                checked,
            )

            if not should_check:
                continue

            logger.debug(
                "RUN CHECK: token range [%d:%d]",
                check_start,
                generated,
            )

            decision, checked_text = self._check(
                prompt,
                token_ids,
                check_start,
                generated,
            )

            checked = generated

            logger.debug(
                "RUN CHECK RESULT: generated=%d label=%s risk=%s checked_text=%r",
                generated,
                decision.label,
                decision.risk_score,
                checked_text,
            )

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

                logger.debug(
                    "RUN BLOCK: generated=%d label=%s risk=%s shown=%d hidden=%d",
                    generated,
                    decision.label,
                    decision.risk_score,
                    shown,
                    generated - shown,
                )

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

                logger.debug(
                    "RUN: sentence completed, sentence_start %d -> %d",
                    sentence_start,
                    generated,
                )

                sentence_start = generated

            if generated > shown:

                shown_start = shown
                shown_end = generated

                shown_text = self._decode(token_ids[shown_start:shown_end])

                shown = generated

                logger.debug(
                    "RUN SHOW: tokens [%d:%d] text=%r total_shown=%d",
                    shown_start,
                    shown_end,
                    shown_text,
                    shown,
                )

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

        logger.debug(
            "RUN END: generated=%d checked=%d shown=%d hidden=%d blocked=%s first_block=%s shown_text=%r",
            generated,
            checked,
            shown,
            generated - shown,
            blocked,
            first_block,
            shown_text,
        )

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

        logger.debug(
            "_run_full START: prompt=%r response=%r token_count=%d",
            prompt,
            response,
            len(token_ids),
        )

        decision = self.guard.score_prefix(
            prompt,
            response,
        )

        latency = (time.perf_counter() - started) * 1000

        logger.debug(
            "_run_full DECISION: label=%s risk=%s latency=%s measured=%.3f ms",
            decision.label,
            decision.risk_score,
            decision.latency_ms,
            latency,
        )

        if decision.latency_ms == 0:
            decision.latency_ms = latency

        n = len(token_ids)

        blocked = decision.blocked and self.stop_on_unsafe
        shown = 0 if blocked else n

        logger.debug(
            "_run_full STATE: n=%d blocked=%s shown=%d stop_on_unsafe=%s",
            n,
            blocked,
            shown,
            self.stop_on_unsafe,
        )

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

            logger.debug(
                "_run_full BLOCK: all %d tokens hidden",
                n,
            )

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

            logger.debug(
                "_run_full SHOW: all %d tokens shown",
                n,
            )

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

        logger.debug(
            "_run_full END: blocked=%s shown=%d hidden=%d",
            blocked,
            shown,
            n - shown,
        )

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
