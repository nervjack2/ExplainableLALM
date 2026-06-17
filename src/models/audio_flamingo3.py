import numpy as np
import torch
from transformers import AudioFlamingo3ForConditionalGeneration, AutoProcessor

from src.models.text_utils import find_text_content_positions


class AudioFlamingo3AudioModel:
    """
    NVIDIA Audio Flamingo 3 wrapper for activation extraction experiments.

    Requires transformers>=5.0.0 -- AudioFlamingo3ForConditionalGeneration was
    only added in that release (not present in the 4.x line).
    """

    MODEL_ID = "nvidia/audio-flamingo-3-hf"

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.processor = AutoProcessor.from_pretrained(self.MODEL_ID)
        self.tokenizer = self.processor.tokenizer
        self.model = AudioFlamingo3ForConditionalGeneration.from_pretrained(
            self.MODEL_ID,
            torch_dtype="auto",
            device_map=device,
        )
        self.model.eval()

    def _build_conversation(self, text: str, audio: np.ndarray = None) -> list:
        content = [{"type": "text", "text": text}]
        if audio is not None:
            content.append({"type": "audio", "audio": audio})
        return [{"role": "user", "content": content}]

    def prepare_inputs(self, audio: np.ndarray, text: str) -> dict:
        conversation = self._build_conversation(text, audio)
        return self.processor.apply_chat_template(
            conversation, tokenize=True, add_generation_prompt=True, return_dict=True,
        ).to(self.device)

    def prepare_text_inputs(self, text: str) -> dict:
        conversation = self._build_conversation(text)
        return self.processor.apply_chat_template(
            conversation, tokenize=True, add_generation_prompt=True, return_dict=True,
        ).to(self.device)

    def find_text_content_positions(self, input_ids: torch.Tensor, text: str) -> list[int]:
        return find_text_content_positions(self.tokenizer, input_ids, text)

    def find_audio_positions(self, input_ids: torch.Tensor) -> list[int]:
        ids = input_ids[0]
        positions = (ids == self.processor.audio_token_id).nonzero(as_tuple=True)[0]
        return positions.tolist()

    def generate(self, inputs: dict, max_new_tokens: int = 256) -> str:
        output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        input_len = inputs["input_ids"].shape[1]
        return self.processor.batch_decode(
            output_ids[:, input_len:], skip_special_tokens=True
        )[0]
