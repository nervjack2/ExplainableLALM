import numpy as np
import torch

from src.models.desta_model import DeSTA2_5Model, Desta2_5Processor
from src.models.text_utils import find_text_content_positions


class DeSTAAudioModel:
    """
    DeSTA2.5-Audio (Llama-3.1-8B backbone) wrapper for activation extraction experiments.

    Requires the official `desta` package to be importable:
        pip install -e /path/to/DeSTA2.5-Audio  (https://github.com/kehanlu/DeSTA2.5-Audio)

    Audio is injected as a fixed-size block of placeholder embeddings (Qformer-compressed
    audio features followed by ASR transcription embeddings) rather than as raw audio
    tokens like Qwen2.5-Omni, so audio token positions are found via a placeholder token id
    instead of start/end markers.
    """

    MODEL_ID = "DeSTA-ntu/DeSTA2.5-Audio-Llama-3.1-8B"

    AUDIO_PLACEHOLDER_TOKEN_ID = 128096
    TRANSCRIPTION_PLACEHOLDER_TOKEN_ID = 128097

    SYSTEM_PROMPT = "Focus on the audio clips and instructions."

    def __init__(self, device: str = "cuda"):
        self.device = device

        self.model = DeSTA2_5Model.from_pretrained(self.MODEL_ID)
        self.model.to(self.device)
        self.model.eval()

        processing_config = {
            "audio_prompt_size": self.model.config.prompt_size,
            "audio_locator": self.model.config.audio_locator,
            "audio_placeholder_token": "<|reserved_special_token_87|>",
            "transcription_placeholder_token": "<|reserved_special_token_88|>",
            "audio_placeholder_token_id": self.AUDIO_PLACEHOLDER_TOKEN_ID,
            "transcription_placeholder_token_id": self.TRANSCRIPTION_PLACEHOLDER_TOKEN_ID,
        }
        self.processor = Desta2_5Processor(
            text_tokenizer_id=self.model.config.llm_model_id,
            speech_processor_id=self.model.config.encoder_model_id,
            processing_config=processing_config,
        )
        self.tokenizer = self.processor.tokenizer
        self.model.set_processing_config(processing_config)

    def format_prompt(self, audio: np.ndarray, transcription: str, text: str) -> str:
        conversation = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"<|AUDIO|>\n{text}",
                "audios": [{"audio": audio, "text": transcription}],
            },
        ]
        return self.processor.apply_chat_template(conversation, add_generation_prompt=True)

    def prepare_inputs(self, audio: np.ndarray, text: str) -> dict:
        transcription = self.model.prepare_transcriptions([audio], [None])[0]
        prompt = self.format_prompt(audio, transcription, text)
        inputs = self.processor(
            audio=[audio], transcription=[transcription], text=[prompt],
            add_special_tokens=False, return_tensors="pt", padding=True,
        ).to(self.device)
        return inputs

    def prepare_text_inputs(self, text: str) -> dict:
        conversation = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ]
        prompt = self.tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        )
        return self.tokenizer(prompt, return_tensors="pt").to(self.device)

    def find_text_content_positions(self, input_ids: torch.Tensor, text: str) -> list[int]:
        return find_text_content_positions(self.tokenizer, input_ids, text)

    def find_audio_positions(self, input_ids: torch.Tensor) -> list[int]:
        ids = input_ids[0]
        positions = (ids == self.AUDIO_PLACEHOLDER_TOKEN_ID).nonzero(as_tuple=True)[0]
        return positions.tolist()

    def generate(self, inputs: dict, max_new_tokens: int = 256) -> str:
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        # Generation here runs on inputs_embeds (no input_ids), so output_ids
        # already excludes the prompt -- no slicing needed.
        return self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
