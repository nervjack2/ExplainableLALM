from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List

import numpy as np


@dataclass
class NeuronRecord:
    """Identifies a single neuron and its statistics for a given attribute label."""
    layer: str          # original module path (e.g. "model.thinker.model.layers.3.mlp.down_proj")
    layer_idx: int      # sequential index among all hooked layers (0-based)
    neuron_idx: int     # dimension index within the intermediate projection
    activation_rate: float   # P(neuron > threshold | target label)
    specificity: float       # activation_rate − mean activation_rate over other labels


class NeuronAnalyzer:
    """
    Load pre-computed per-sample activation files and compute
    per-neuron, per-label activation statistics.

    Expected directory layout produced by extract_activations.py:
        <data_dir>/
            layers.json          – ordered list of layer names
            metadata.json        – [{file_id, label}, …]
            <file_id>.npz        – keys "layer_0" … "layer_N",
                                   each (intermediate_dim,) float16
                                   = mean activation across audio frames
    """

    def __init__(self, data_dir: str, threshold: float = 0.0):
        self.data_dir = data_dir
        self.threshold = threshold

        with open(os.path.join(data_dir, "layers.json")) as f:
            self.layer_names: List[str] = json.load(f)

        with open(os.path.join(data_dir, "metadata.json")) as f:
            metadata: List[dict] = json.load(f)

        self._samples: List[dict] = []
        for item in metadata:
            path = os.path.join(data_dir, f"{item['file_id']}.npz")
            if not os.path.isfile(path):
                continue
            data = np.load(path)
            # Re-key from "layer_0" → layer_names[0], etc.
            activations = {
                self.layer_names[int(k.split("_")[1])]: data[k].astype(np.float32)
                for k in data.files
            }
            self._samples.append({"id": item["file_id"], "label": item["label"],
                                   "activations": activations})

        self.labels: List[str] = sorted({s["label"] for s in self._samples})

        # Prune layer_names to those actually present in the saved data
        # (e.g. visual encoder layers are hooked but never fire during audio-only inference)
        layers_with_data: set = set()
        for s in self._samples:
            layers_with_data.update(s["activations"].keys())
        self.layer_names = [l for l in self.layer_names if l in layers_with_data]

    # ------------------------------------------------------------------

    def summary(self) -> dict:
        counts = {l: sum(1 for s in self._samples if s["label"] == l) for l in self.labels}
        return {
            "n_samples": len(self._samples),
            "labels": self.labels,
            "label_counts": counts,
            "n_layers": len(self.layer_names),
        }

    def activation_rates(self) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Compute P(neuron > threshold) for every (label, layer, neuron) triple.

        Returns
        -------
        {label: {layer_name: np.ndarray(intermediate_dim)}}
        """
        rates: Dict[str, Dict[str, np.ndarray]] = {l: {} for l in self.labels}
        for label in self.labels:
            samples = [s for s in self._samples if s["label"] == label]
            for layer in self.layer_names:
                stacked = np.stack([s["activations"][layer] for s in samples])  # (N, D)
                rates[label][layer] = (stacked > self.threshold).mean(axis=0)    # (D,)
        return rates

    def find_attribute_neurons(
        self,
        target_label: str,
        top_k: int = 20,
    ) -> List[NeuronRecord]:
        """
        Return the top-k neurons most specifically associated with *target_label*.

        Specificity score = P(act | target) − mean_over_other_labels P(act | other).
        Neurons are ranked by specificity (descending).
        """
        rates = self.activation_rates()
        other_labels = [l for l in self.labels if l != target_label]

        records: List[NeuronRecord] = []
        for layer_idx, layer in enumerate(self.layer_names):
            target_rate = rates[target_label][layer]          # (D,)
            if other_labels:
                mean_other = np.stack(
                    [rates[l][layer] for l in other_labels]
                ).mean(axis=0)                                 # (D,)
            else:
                mean_other = np.zeros_like(target_rate)

            specificity = target_rate - mean_other             # (D,)

            for neuron_idx in range(len(specificity)):
                records.append(NeuronRecord(
                    layer=layer,
                    layer_idx=layer_idx,
                    neuron_idx=neuron_idx,
                    activation_rate=float(target_rate[neuron_idx]),
                    specificity=float(specificity[neuron_idx]),
                ))

        records.sort(key=lambda r: r.specificity, reverse=True)
        return records[:top_k]

    def all_attribute_neurons(self, top_k: int = 20) -> Dict[str, List[NeuronRecord]]:
        """Convenience: run find_attribute_neurons for every label."""
        return {label: self.find_attribute_neurons(label, top_k) for label in self.labels}
