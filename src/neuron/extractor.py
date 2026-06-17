from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn


class MLPActivationExtractor:
    """
    Capture MLP intermediate activations for every transformer layer.

    For SwiGLU-style MLPs:
        intermediate = act_fn(gate_proj(x)) * up_proj(x)
        output       = down_proj(intermediate)

    We register a forward pre-hook on each `down_proj` layer so the hook
    receives `intermediate` as its first positional argument — this is the
    true per-neuron activation before the projection back to hidden dim.

    A neuron at dimension k is considered "activated" when intermediate[k] > 0.
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self._hooks: list = []
        self._activations: Dict[str, torch.Tensor] = {}
        self._layer_names: List[str] = []
        self._register_hooks()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _register_hooks(self) -> None:
        for full_name, module in self.model.named_modules():
            parts = full_name.split(".")
            if len(parts) >= 2 and parts[-1] == "down_proj" and parts[-2] == "mlp" and "visual" not in parts:
                self._layer_names.append(full_name)
                hook = module.register_forward_pre_hook(self._make_hook(full_name))
                self._hooks.append(hook)

        # Sort by numeric layer index so layers are in model order
        self._layer_names.sort(
            key=lambda n: [int(p) if p.isdigit() else p for p in n.split(".")]
        )

    def _make_hook(self, name: str):
        def hook(module, args):
            # args[0]: (batch, seq_len, intermediate_dim)  – move to CPU immediately
            self._activations[name] = args[0].detach().cpu().float()
        return hook

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """Discard cached activations between samples."""
        self._activations.clear()

    def remove_hooks(self) -> None:
        """Clean up all registered hooks (call when analysis is done)."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    @property
    def layer_names(self) -> List[str]:
        return list(self._layer_names)

    def get_audio_frame_activations(
        self,
        audio_positions: Optional[List[int]],
    ) -> Dict[str, torch.Tensor]:
        """
        Slice activations to audio token positions for every layer.

        Returns
        -------
        dict  layer_name -> tensor (n_audio_frames, intermediate_dim)
        """
        result: Dict[str, torch.Tensor] = {}
        for name in self._layer_names:
            if name not in self._activations:
                continue
            act = self._activations[name]   # (1, seq_len, intermediate_dim)
            if audio_positions:
                idx = torch.tensor(audio_positions, dtype=torch.long)
                result[name] = act[0, idx]  # (n_frames, intermediate_dim)
            else:
                result[name] = act[0]       # fallback: all positions
        return result
