import logging
import time
from typing import Optional

import torch
from transformers import AutoModel, AutoTokenizer

from streaming_engine.data_classes import (
    GuardAdapter,
    Decision,
)

logger = logging.getLogger(__name__)


class Qwen3GuardStreamAdapter(GuardAdapter):

    def __init__(
        self,
        model_id_or_path: str = "Qwen/Qwen3Guard-Stream-0.6B",
        device: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
    ):

        logger.debug(
            "Qwen3GuardStreamAdapter.__init__: model=%s device=%s dtype=%s",
            model_id_or_path,
            device,
            torch_dtype,
        )

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device)

        if torch_dtype is None:
            if self.device.type == "cuda":
                if torch.cuda.is_bf16_supported():
                    torch_dtype = torch.bfloat16
                else:
                    torch_dtype = torch.float16
            else:
                torch_dtype = torch.float32

        logger.debug(
            "Selected device=%s torch_dtype=%s",
            self.device,
            torch_dtype,
        )

        print(f"Loading tokenizer: {model_id_or_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id_or_path,
            trust_remote_code=True,
        )

        print(f"Loading Qwen3Guard-Stream on {self.device}")

        if self.device.type == "cuda":
            self.model = AutoModel.from_pretrained(
                model_id_or_path,
                trust_remote_code=True,
                torch_dtype=torch_dtype,
                device_map="auto",
            )
        else:
            self.model = AutoModel.from_pretrained(
                model_id_or_path,
                trust_remote_code=True,
                torch_dtype=torch_dtype,
            )

            self.model.to(self.device)

        self.model.eval()

        self.prompt = ""
        self.stream_state = None
        self.assistant_token_ids = []
        self.processed_assistant_tokens = 0

        logger.debug(
            "Adapter initialized: device=%s prompt=%r stream_state=%s processed=%d",
            self.device,
            self.prompt,
            self.stream_state,
            self.processed_assistant_tokens,
        )

    def _encode_user(
        self,
        prompt: str,
    ) -> torch.Tensor:

        user_text = "<|im_start|>user\n" f"{prompt}" "<|im_end|>"

        token_ids = self.tokenizer(
            user_text,
            add_special_tokens=False,
            return_tensors="pt",
        ).input_ids[0]

        result = token_ids.to(self.device)

        logger.debug(
            "_encode_user: text=%r tokens=%d ids=%s device=%s",
            user_text,
            len(result),
            result.detach().cpu().tolist(),
            result.device,
        )

        return result

    def _encode_assistant(
        self,
        text: str,
    ) -> torch.Tensor:

        tokenized = self.tokenizer(
            text,
            add_special_tokens=False,
            return_attention_mask=False,
        )

        result = torch.tensor(
            tokenized["input_ids"],
            dtype=torch.long,
            device=self.device,
        )

        logger.debug(
            "_encode_assistant: text=%r tokens=%d ids=%s device=%s",
            text,
            len(result),
            result.detach().cpu().tolist(),
            result.device,
        )

        return result

    def reset(
        self,
        prompt: str = "",
    ) -> None:

        logger.debug(
            "RESET START: prompt=%r old_prompt=%r old_processed=%d old_stream_state=%s",
            prompt,
            self.prompt,
            self.processed_assistant_tokens,
            self.stream_state,
        )

        self.prompt = prompt
        self.stream_state = None
        self.assistant_token_ids = []
        self.processed_assistant_tokens = 0

        logger.debug(
            "RESET STATE CLEARED: stream_state=%s processed=%d",
            self.stream_state,
            self.processed_assistant_tokens,
        )

        if not prompt:
            logger.debug("RESET END: empty prompt")
            return

        user_token_ids = self._encode_user(prompt)

        logger.debug(
            "RESET: sending user tokens to model: count=%d ids=%s",
            len(user_token_ids),
            user_token_ids.detach().cpu().tolist(),
        )

        _, self.stream_state = self.model.stream_moderate_from_ids(
            user_token_ids,
            role="user",
            stream_state=None,
        )

        logger.debug(
            "RESET END: new stream_state=%s",
            self.stream_state,
        )

    @staticmethod
    def _to_list(value):

        logger.debug(
            "_to_list: type=%s value=%r",
            type(value),
            value,
        )

        if torch.is_tensor(value):
            return value.detach().cpu().flatten().tolist()

        if isinstance(value, (list, tuple)):
            return list(value)

        return [value]

    @classmethod
    def _extract_labels(
        cls,
        result,
    ) -> list[str]:

        logger.debug(
            "_extract_labels: result_keys=%s result=%r",
            list(result.keys()) if isinstance(result, dict) else None,
            result,
        )

        risk_level = result.get("risk_level")

        if risk_level is None:
            raise RuntimeError(
                "Qwen3Guard did not return "
                f"'risk_level'. Returned keys: "
                f"{list(result.keys())}"
            )

        values = cls._to_list(risk_level)

        labels = [str(value).lower() for value in values]

        logger.debug(
            "_extract_labels: risk_level=%r labels=%s",
            risk_level,
            labels,
        )

        return labels

    @classmethod
    def _normalize_label(
        cls,
        result,
    ) -> str:

        labels = cls._extract_labels(result)

        if not labels:
            logger.debug("_normalize_label: no labels -> safe")
            return "safe"

        if any(label == "unsafe" for label in labels):
            logger.debug(
                "_normalize_label: labels=%s -> unsafe",
                labels,
            )
            return "unsafe"

        if any(label == "controversial" for label in labels):
            logger.debug(
                "_normalize_label: labels=%s -> controversial",
                labels,
            )
            return "controversial"

        logger.debug(
            "_normalize_label: labels=%s -> %s",
            labels,
            labels[-1],
        )

        return labels[-1]

    @classmethod
    def _extract_score(
        cls,
        result,
        label: str,
    ) -> float:

        logger.debug(
            "_extract_score: label=%s result_keys=%s",
            label,
            list(result.keys()) if isinstance(result, dict) else None,
        )

        possible_keys = (
            "risk_score",
            "risk_probability",
            "unsafe_probability",
            "probability",
            "probs",
            "probabilities",
            "risk_prob",
        )

        for key in possible_keys:

            value = result.get(key)

            if value is None:
                continue

            values = cls._to_list(value)

            if not values:
                continue

            try:
                numeric_values = [float(item) for item in values]

                score = numeric_values[-1]

                logger.debug(
                    "_extract_score: key=%s values=%s score=%s",
                    key,
                    numeric_values,
                    score,
                )

                return score

            except (
                TypeError,
                ValueError,
            ):
                logger.debug(
                    "_extract_score: key=%s contains non-numeric value=%r",
                    key,
                    values,
                )
                continue

        if label == "safe":
            return 0.0

        if label == "unsafe":
            return 1.0

        if label == "controversial":
            return 0.5

        return 0.0

    @torch.no_grad()
    def score_prefix(
        self,
        prompt: str,
        response_prefix: str,
    ) -> Decision:

        started = time.perf_counter()

        logger.debug(
            "SCORE_PREFIX START: prompt=%r response_prefix=%r",
            prompt,
            response_prefix,
        )

        if prompt != self.prompt or self.stream_state is None:
            logger.debug(
                "SCORE_PREFIX: reset required prompt_changed=%s stream_state_none=%s",
                prompt != self.prompt,
                self.stream_state is None,
            )
            self.reset(prompt)

        current_tensor = self._encode_assistant(response_prefix)

        current_token_ids = current_tensor.detach().cpu().tolist()

        previous_count = self.processed_assistant_tokens

        logger.debug(
            "SCORE_PREFIX TOKENS: current_count=%d previous_count=%d current_ids=%s",
            len(current_token_ids),
            previous_count,
            current_token_ids,
        )

        if len(current_token_ids) < previous_count:
            logger.debug(
                "SCORE_PREFIX: current token count decreased: %d < %d. RESET.",
                len(current_token_ids),
                previous_count,
            )

            self.reset(prompt)
            previous_count = 0

        new_token_ids = current_token_ids[previous_count:]

        logger.debug(
            "SCORE_PREFIX NEW TOKENS: previous_count=%d new_count=%d new_ids=%s",
            previous_count,
            len(new_token_ids),
            new_token_ids,
        )

        if not new_token_ids:
            elapsed = (time.perf_counter() - started) * 1000

            logger.debug(
                "SCORE_PREFIX END: no new tokens -> safe latency=%.3f ms",
                elapsed,
            )

            return Decision(
                label="safe",
                risk_score=0.0,
                latency_ms=elapsed,
            )

        labels = []
        scores = []

        for index, token_id in enumerate(new_token_ids, 1):

            token_tensor = torch.tensor(
                [token_id],
                dtype=torch.long,
                device=self.device,
            )

            token_text = self.tokenizer.decode(
                [token_id],
                skip_special_tokens=False,
            )

            logger.debug(
                "SCORE_PREFIX TOKEN %d/%d: id=%s text=%r tensor_shape=%s dtype=%s device=%s stream_state=%s",
                index,
                len(new_token_ids),
                token_id,
                token_text,
                tuple(token_tensor.shape),
                token_tensor.dtype,
                token_tensor.device,
                self.stream_state,
            )

            result, self.stream_state = self.model.stream_moderate_from_ids(
                token_tensor,
                role="assistant",
                stream_state=self.stream_state,
            )

            logger.debug(
                "SCORE_PREFIX MODEL RESULT TOKEN %d: result=%r new_stream_state=%s",
                index,
                result,
                self.stream_state,
            )

            token_labels = self._extract_labels(result)

            token_label = token_labels[-1] if token_labels else "safe"

            token_score = self._extract_score(
                result,
                token_label,
            )

            labels.append(token_label)
            scores.append(token_score)

            logger.debug(
                "SCORE_PREFIX TOKEN DECISION %d: label=%s score=%s",
                index,
                token_label,
                token_score,
            )

        if any(label == "unsafe" for label in labels):
            label = "unsafe"

        elif any(label == "controversial" for label in labels):
            label = "controversial"

        else:
            label = "safe"

        score = max(scores) if scores else 0.0

        self.assistant_token_ids = current_token_ids

        self.processed_assistant_tokens = len(current_token_ids)

        elapsed = (time.perf_counter() - started) * 1000

        logger.debug(
            "SCORE_PREFIX END: label=%s score=%s tokens=%d processed=%d latency=%.3f ms labels=%s scores=%s",
            label,
            score,
            len(current_token_ids),
            self.processed_assistant_tokens,
            elapsed,
            labels,
            scores,
        )

        return Decision(
            label=label,
            risk_score=score,
            latency_ms=elapsed,
        )

    @torch.no_grad()
    def score_local(
        self,
        prompt: str,
        text: str,
    ) -> Decision:

        started = time.perf_counter()

        logger.debug(
            "SCORE_LOCAL START: prompt=%r text=%r",
            prompt,
            text,
        )

        if not text:
            elapsed = (time.perf_counter() - started) * 1000

            logger.debug(
                "SCORE_LOCAL END: empty text -> safe latency=%.3f ms",
                elapsed,
            )

            return Decision(
                label="safe",
                risk_score=0.0,
                latency_ms=elapsed,
            )

        user_token_ids = self._encode_user(prompt)

        logger.debug(
            "SCORE_LOCAL USER: count=%d ids=%s",
            len(user_token_ids),
            user_token_ids.detach().cpu().tolist(),
        )

        _, stream_state = self.model.stream_moderate_from_ids(
            user_token_ids,
            role="user",
            stream_state=None,
        )

        logger.debug(
            "SCORE_LOCAL AFTER USER: stream_state=%s",
            stream_state,
        )

        token_tensor = self._encode_assistant(text)
        token_ids = token_tensor.detach().cpu().tolist()

        logger.debug(
            "SCORE_LOCAL ASSISTANT: count=%d ids=%s tensor_shape=%s dtype=%s device=%s",
            len(token_ids),
            token_ids,
            tuple(token_tensor.shape),
            token_tensor.dtype,
            token_tensor.device,
        )

        labels = []
        scores = []

        for index, token_id in enumerate(token_ids, 1):

            token_tensor = torch.tensor(
                [token_id],
                dtype=torch.long,
                device=self.device,
            )

            token_text = self.tokenizer.decode(
                [token_id],
                skip_special_tokens=False,
            )

            logger.debug(
                "SCORE_LOCAL TOKEN %d/%d: id=%s text=%r tensor_shape=%s dtype=%s device=%s stream_state=%s",
                index,
                len(token_ids),
                token_id,
                token_text,
                tuple(token_tensor.shape),
                token_tensor.dtype,
                token_tensor.device,
                stream_state,
            )

            result, stream_state = self.model.stream_moderate_from_ids(
                token_tensor,
                role="assistant",
                stream_state=stream_state,
            )

            logger.debug(
                "SCORE_LOCAL MODEL RESULT TOKEN %d: result=%r new_stream_state=%s",
                index,
                result,
                stream_state,
            )

            token_labels = self._extract_labels(result)

            token_label = token_labels[-1] if token_labels else "safe"

            token_score = self._extract_score(
                result,
                token_label,
            )

            labels.append(token_label)
            scores.append(token_score)

            logger.debug(
                "SCORE_LOCAL TOKEN DECISION %d: label=%s score=%s",
                index,
                token_label,
                token_score,
            )

        if any(label.lower() == "unsafe" for label in labels):
            label = "unsafe"

        elif any(label.lower() == "controversial" for label in labels):
            label = "controversial"

        else:
            label = "safe"

        score = max(scores) if scores else 0.0

        elapsed = (time.perf_counter() - started) * 1000

        logger.debug(
            "SCORE_LOCAL END: label=%s score=%s tokens=%d latency=%.3f ms labels=%s scores=%s",
            label,
            score,
            len(token_ids),
            elapsed,
            labels,
            scores,
        )

        return Decision(
            label=label,
            risk_score=score,
            latency_ms=elapsed,
        )
