import time

import torch

from streaming_engine.data_classes import Decision, GuardAdapter


class Qwen3GuardStreamAdapter(GuardAdapter):
    def __init__(
        self,
        model_id_or_path: str,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    ):
        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id_or_path,
            trust_remote_code=True,
        )

        self.model = AutoModel.from_pretrained(
            model_id_or_path,
            device_map=device_map,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        ).eval()

        self.device = self.model.device
        self.stream_state = None
        self.prompt = ""
        self.assistant_token_ids: list[int] = []
        self.processed_assistant_tokens = 0

    def reset(self, prompt: str = "") -> None:
        self.prompt = prompt
        self.assistant_token_ids = []
        self.processed_assistant_tokens = 0

        if self.stream_state is not None:
            self.model.close_stream(self.stream_state)

        self.stream_state = None

        user_ids = self._tokenize_user(prompt)

        _, self.stream_state = self.model.stream_moderate_from_ids(
            user_ids,
            role="user",
            stream_state=None,
        )

    def _tokenize_user(self, prompt: str) -> torch.Tensor:
        messages = [
            {
                "role": "user",
                "content": prompt,
            }
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
        )

        return inputs.input_ids[0].to(self.device)

    def _tokenize_assistant_response(
        self,
        prompt: str,
        response_prefix: str,
    ) -> torch.Tensor:

        messages = [
            {
                "role": "user",
                "content": prompt,
            },
            {
                "role": "assistant",
                "content": response_prefix,
            },
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
        )

        token_ids = inputs.input_ids[0]

        im_start_id = self.tokenizer.convert_tokens_to_ids("<|im_start|>")
        assistant_id = self.tokenizer.convert_tokens_to_ids("assistant")
        im_end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")

        token_ids_list = token_ids.tolist()

        assistant_start = None

        for i in range(len(token_ids_list) - 1):
            if token_ids_list[i : i + 2] == [
                im_start_id,
                assistant_id,
            ]:
                assistant_start = i

        if assistant_start is None:
            raise RuntimeError("Could not find assistant turn in Qwen chat template.")

        assistant_end = len(token_ids_list)

        for i in range(
            assistant_start + 2,
            len(token_ids_list),
        ):
            if token_ids_list[i] == im_end_id:
                assistant_end = i
                break

        return token_ids[assistant_start + 2 : assistant_end].to(self.device)

    @staticmethod
    def _extract_risk_score(result) -> float:
        risk_prob = result.get("risk_prob")

        if risk_prob is None:
            return 1.0 if result["risk_level"][-1] != "Safe" else 0.0

        risk_prob = risk_prob[-1]

        if isinstance(risk_prob, dict):
            for key in ("Unsafe", "unsafe", "unsafe_prob"):
                if key in risk_prob:
                    return float(risk_prob[key])

        if torch.is_tensor(risk_prob):
            if risk_prob.numel() == 1:
                return float(risk_prob.item())

        if isinstance(risk_prob, (int, float)):
            return float(risk_prob)

        return 1.0 if result["risk_level"][-1] != "Safe" else 0.0

    @torch.inference_mode()
    def score_prefix(
        self,
        prompt: str,
        response_prefix: str,
    ) -> Decision:

        started = time.perf_counter()

        current_token_ids = self._tokenize_assistant_response(
            prompt,
            response_prefix,
        )

        current_token_ids = current_token_ids.tolist()

        previous_length = self.processed_assistant_tokens

        new_token_ids = current_token_ids[previous_length:]

        if not new_token_ids:
            elapsed = (time.perf_counter() - started) * 1000

            return Decision(
                label="safe",
                risk_score=0.0,
                latency_ms=elapsed,
            )

        result = None

        for token_id in new_token_ids:
            token_tensor = torch.tensor(
                token_id,
                dtype=torch.long,
                device=self.device,
            )

            result, self.stream_state = self.model.stream_moderate_from_ids(
                token_tensor,
                role="assistant",
                stream_state=self.stream_state,
            )

        self.assistant_token_ids = current_token_ids
        self.processed_assistant_tokens = len(current_token_ids)

        elapsed = (time.perf_counter() - started) * 1000

        risk = result["risk_level"][-1]
        risk_score = self._extract_risk_score(result)

        if risk == "Safe":
            label = "safe"
        else:
            label = "unsafe"

        return Decision(
            label=label,
            risk_score=risk_score,
            latency_ms=elapsed,
        )

    def close(self) -> None:
        if self.stream_state is not None:
            self.model.close_stream(self.stream_state)
            self.stream_state = None
