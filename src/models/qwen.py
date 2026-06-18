import numpy as np
import torch
from transformers.models.qwen2_5_omni import (
    Qwen2_5OmniThinkerForConditionalGeneration,
    Qwen2_5OmniProcessor,
)

from src.models.text_utils import find_text_content_positions


class QwenAudioModel:
    """
    Qwen2.5-Omni wrapper for activation extraction experiments.

    Responsibilities:
    - Load model and processor
    - Format chat prompts (system + user with audio)
    - Prepare processor inputs from raw audio + text
    - Locate audio token positions in the token sequence
    """

    MODEL_IDS = {
        "3B": "Qwen/Qwen2.5-Omni-3B",
        "7B": "Qwen/Qwen2.5-Omni-7B",
    }

    def __init__(self, device: str = "cuda", model_size: str = "3B"):
        if model_size not in self.MODEL_IDS:
            raise ValueError(f"Unknown model_size '{model_size}', choose from {list(self.MODEL_IDS)}")

        self.device = device
        self.model_id = self.MODEL_IDS[model_size]
        self.processor = Qwen2_5OmniProcessor.from_pretrained(self.model_id)

        # Special token IDs that bracket audio feature tokens in the input sequence
        self.AUDIO_START_TOKEN_ID = self.processor.tokenizer.convert_tokens_to_ids(self.processor.audio_bos_token)
        self.AUDIO_END_TOKEN_ID = self.processor.tokenizer.convert_tokens_to_ids(self.processor.audio_eos_token)
        self.model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            device_map=device,
        )
        self.model.eval()

    def _build_conversation(self, text: str) -> list:
        return [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text},
                    {"type": "audio", "audio_url": "x"},
                ],
            },
        ]

    def format_prompt(self, text: str) -> str:
        conversation = self._build_conversation(text)
        return self.processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )

    def prepare_inputs(self, audio: np.ndarray, text: str) -> dict:
        """Build processor inputs ready for model(**inputs)."""
        prompt = self.format_prompt(text)
        inputs = self.processor(
            audio=[audio],
            text=[prompt],
            return_tensors="pt",
            sampling_rate=16000,
        ).to(self.device)
        return inputs

    def _build_text_conversation(self, text: str) -> list:
        return [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
                ],
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": text}],
            },
        ]

    def prepare_text_inputs(self, text: str) -> dict:
        """Build processor inputs for a text-only forward pass (no audio)."""
        conversation = self._build_text_conversation(text)
        prompt = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )
        return self.processor(text=[prompt], return_tensors="pt").to(self.device)

    def find_text_content_positions(self, input_ids: torch.Tensor, text: str) -> list[int]:
        return find_text_content_positions(self.processor.tokenizer, input_ids, text)

    def generate(self, inputs: dict, max_new_tokens: int = 256) -> str:
        """Generate a text response from prepared inputs (for sanity-checking)."""
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.processor.tokenizer.pad_token_id,
            bos_token_id=self.processor.tokenizer.bos_token_id,
            eos_token_id=self.processor.tokenizer.eos_token_id,
        )
        input_len = inputs["input_ids"].shape[1]
        return self.processor.batch_decode(
            output_ids[:, input_len:], skip_special_tokens=True
        )[0]

    def find_audio_positions(self, input_ids: torch.Tensor) -> list[int]:
        """
        Return list of token positions that correspond to audio feature tokens.
        Audio tokens sit strictly between AUDIO_START and AUDIO_END markers.
        """
        ids = input_ids[0]
        start_pos = (ids == self.AUDIO_START_TOKEN_ID).nonzero(as_tuple=True)[0]
        end_pos = (ids == self.AUDIO_END_TOKEN_ID).nonzero(as_tuple=True)[0]

        if len(start_pos) == 0 or len(end_pos) == 0:
            return []

        st = start_pos[0].item() + 1   # first token after <|AUDIO|>
        ed = end_pos[0].item()          # position of </|AUDIO|> (exclusive)
        return list(range(st, ed))
