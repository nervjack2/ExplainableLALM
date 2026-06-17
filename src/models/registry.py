"""
Registry mapping a --model CLI choice to (module path, class name, constructor kwargs).

Every wrapper class must implement the interface used by extract_activations.py /
extract_text_activations.py:
    prepare_inputs(audio, text) -> dict
    prepare_text_inputs(text) -> dict
    find_audio_positions(input_ids) -> list[int]
    find_text_content_positions(input_ids, text) -> list[int]
    generate(inputs, max_new_tokens) -> str
    .model  (the underlying nn.Module, for hook registration)

The module is imported lazily inside build_model() so that picking one model
doesn't require every other model's (sometimes heavy/optional) dependencies
to be installed.
"""

import importlib

MODEL_REGISTRY = {
    "qwen2.5-omni-3b": ("src.models.qwen", "QwenAudioModel", {"model_size": "3B"}),
    "qwen2.5-omni-7b": ("src.models.qwen", "QwenAudioModel", {"model_size": "7B"}),
    "desta2.5-audio": ("src.models.desta", "DeSTAAudioModel", {}),
    "audio-flamingo-3": ("src.models.audio_flamingo3", "AudioFlamingo3AudioModel", {}),
}


def build_model(name: str, device: str = "cuda"):
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}', choose from {list(MODEL_REGISTRY)}")
    module_path, class_name, kwargs = MODEL_REGISTRY[name]
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls(device=device, **kwargs)
