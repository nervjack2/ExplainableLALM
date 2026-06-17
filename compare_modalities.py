"""
compare_modalities.py
=====================
Compare audio vs. text MLP activation patterns for each emotion label.

Method 1 — Layer-wise cosine similarity
    For each (emotion, layer): cosine similarity between the mean audio
    activation vector (across all samples of that label) and the single
    text activation vector produced by "The speaker feels {emotion}.".

Method 2 — Top-k neuron Jaccard overlap
    For each (emotion, k): Jaccard similarity between the top-k neurons
    ranked by audio mean activation and top-k neurons ranked by text
    activation value.  A random baseline curve is overlaid.

Outputs (written to --out_dir):
    cosine_similarity.png / .json
    jaccard_overlap.png   / .json

Usage
-----
    python compare_modalities.py \\
        --audio_dir ./activations/sakura_emotion \\
        --text_dir  ./activations/text_emotion \\
        --out_dir   ./results/modality_comparison
"""

import argparse
import json
import os

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _load_layer_names(data_dir: str) -> list[str]:
    with open(os.path.join(data_dir, "layers.json")) as f:
        return json.load(f)


def load_audio_mean_activations(audio_dir: str) -> tuple[dict, list[str]]:
    """Returns ({label: {layer: ndarray(D)}}, layer_names)."""
    layer_names = _load_layer_names(audio_dir)
    with open(os.path.join(audio_dir, "metadata.json")) as f:
        metadata = json.load(f)

    label_samples: dict[str, list[dict]] = {}
    for item in metadata:
        path = os.path.join(audio_dir, f"{item['file_id']}.npz")
        if not os.path.isfile(path):
            continue
        data = np.load(path)
        acts = {
            layer_names[int(k.split("_")[1])]: data[k].astype(np.float32)
            for k in data.files
        }
        label_samples.setdefault(item["label"], []).append(acts)

    result: dict[str, dict[str, np.ndarray]] = {}
    for label, samples in label_samples.items():
        result[label] = {}
        for layer in layer_names:
            vecs = [s[layer] for s in samples if layer in s]
            if vecs:
                result[label][layer] = np.stack(vecs).mean(axis=0)

    return result, layer_names


def load_text_activations(text_dir: str) -> tuple[dict, list[str]]:
    """Returns ({label: {layer: ndarray(D)}}, layer_names)."""
    layer_names = _load_layer_names(text_dir)
    with open(os.path.join(text_dir, "metadata.json")) as f:
        metadata = json.load(f)

    result: dict[str, dict[str, np.ndarray]] = {}
    for item in metadata:
        path = os.path.join(text_dir, f"{item['file_id']}.npz")
        if not os.path.isfile(path):
            continue
        data = np.load(path)
        result[item["label"]] = {
            layer_names[int(k.split("_")[1])]: data[k].astype(np.float32)
            for k in data.files
        }

    return result, layer_names


# ---------------------------------------------------------------------------
# Method 1 — layer-wise cosine similarity
# ---------------------------------------------------------------------------

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


def method1_cosine(audio_acts, text_acts, layer_names, out_dir) -> dict:
    labels = sorted(set(audio_acts) & set(text_acts))
    valid_layers = [l for l in layer_names if l in next(iter(audio_acts.values()))]

    results: dict[str, list[float]] = {}
    fig, ax = plt.subplots(figsize=(12, 5))

    for label in labels:
        sims = [
            _cosine(audio_acts[label][l], text_acts[label][l])
            if l in audio_acts[label] and l in text_acts[label]
            else float("nan")
            for l in valid_layers
        ]
        results[label] = sims
        ax.plot(sims, label=label)

    ax.set_xlabel("Layer index")
    ax.set_ylabel("Cosine similarity")
    ax.set_title("Audio vs. Text — layer-wise cosine similarity of MLP activations")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "cosine_similarity.png"), dpi=150)
    plt.close(fig)

    with open(os.path.join(out_dir, "cosine_similarity.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("[Method 1] cosine_similarity.png + .json saved")
    return results


# ---------------------------------------------------------------------------
# Method 2 — top-k Jaccard overlap
# ---------------------------------------------------------------------------

def compute_specificity(acts: dict[str, dict[str, np.ndarray]]) -> dict[str, dict[str, np.ndarray]]:
    """
    Transform raw activations into per-label specificity scores.

    specificity(l*, layer, k) = acts[l*][layer][k] - mean over l != l* of acts[l][layer][k]

    Works for both audio (mean-pooled across samples) and text (single vector).
    """
    labels = list(acts.keys())
    result: dict[str, dict[str, np.ndarray]] = {}
    for label in labels:
        other_labels = [l for l in labels if l != label]
        result[label] = {}
        for layer, vec in acts[label].items():
            if other_labels:
                other_vecs = [acts[l][layer] for l in other_labels if layer in acts[l]]
                mean_other = np.stack(other_vecs).mean(axis=0) if other_vecs else np.zeros_like(vec)
            else:
                mean_other = np.zeros_like(vec)
            result[label][layer] = vec - mean_other
    return result


def _top_k_neurons(acts: dict[str, np.ndarray], valid_layers: list[str], k: int) -> set[tuple]:
    """Return set of (layer_idx, neuron_idx) for the top-k neurons by score."""
    entries: list[tuple[float, int, int]] = []
    for li, layer in enumerate(valid_layers):
        if layer not in acts:
            continue
        for ni, val in enumerate(acts[layer].tolist()):
            entries.append((val, li, ni))
    entries.sort(reverse=True)
    return {(li, ni) for _, li, ni in entries[:k]}


def method2_jaccard(
    audio_acts, text_acts, layer_names, out_dir, k_values: list[int],
    use_specificity: bool = False,
) -> dict:
    labels = sorted(set(audio_acts) & set(text_acts))
    valid_layers = [l for l in layer_names if l in next(iter(audio_acts.values()))]

    # Optionally re-score by specificity before ranking
    score_audio = compute_specificity(audio_acts) if use_specificity else audio_acts
    score_text  = compute_specificity(text_acts)  if use_specificity else text_acts
    score_label = "specificity score" if use_specificity else "activation value"

    # Total neurons across all layers (for random baseline)
    D = next(iter(audio_acts[labels[0]].values())).shape[0]
    total_neurons = len(valid_layers) * D

    def random_baseline(k: int) -> float:
        return k / (2 * total_neurons - k)

    results: dict[str, dict] = {}
    fig, ax = plt.subplots(figsize=(8, 5))

    baseline = [random_baseline(k) for k in k_values]
    ax.plot(k_values, baseline, "k--", linewidth=1, label="random baseline")

    for label in labels:
        jaccards = []
        for k in k_values:
            audio_set = _top_k_neurons(score_audio[label], valid_layers, k)
            text_set  = _top_k_neurons(score_text[label],  valid_layers, k)
            inter = len(audio_set & text_set)
            union = len(audio_set | text_set)
            jaccards.append(inter / union if union > 0 else 0.0)
        results[label] = {"k_values": k_values, "jaccard": jaccards}
        ax.plot(k_values, jaccards, marker="o", markersize=3, label=label)

    ax.set_xlabel("k (top-k neurons)")
    ax.set_ylabel("Jaccard similarity")
    ax.set_title(f"Audio vs. Text — top-k neuron Jaccard overlap (ranked by {score_label})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    suffix = "_specificity" if use_specificity else ""
    fig.savefig(os.path.join(out_dir, f"jaccard_overlap{suffix}.png"), dpi=150)
    plt.close(fig)

    with open(os.path.join(out_dir, f"jaccard_overlap{suffix}.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"[Method 2] jaccard_overlap{suffix}.png + .json saved")
    return results


# ---------------------------------------------------------------------------
# Method 2b — cross-emotion Jaccard matrix
# ---------------------------------------------------------------------------

def method2_jaccard_matrix(
    audio_acts, text_acts, layer_names, out_dir, k_values: list[int],
    use_specificity: bool = False,
) -> dict:
    """
    Compute the full (audio_label × text_label) Jaccard matrix for each k.
    If the diagonal is higher than off-diagonal entries, it confirms that
    the shared neurons are emotion-specific rather than generally active.
    Saves a multi-panel heatmap PNG and a JSON.
    """
    audio_labels = sorted(audio_acts.keys())
    text_labels  = sorted(text_acts.keys())
    valid_layers = [l for l in layer_names if l in next(iter(audio_acts.values()))]

    score_audio = compute_specificity(audio_acts) if use_specificity else audio_acts
    score_text  = compute_specificity(text_acts)  if use_specificity else text_acts
    suffix      = "_specificity" if use_specificity else ""
    score_label = "specificity score" if use_specificity else "activation value"

    n     = len(k_values)
    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows), squeeze=False)

    all_results: dict = {}
    for idx, k in enumerate(k_values):
        matrix = np.zeros((len(audio_labels), len(text_labels)))
        for i, al in enumerate(audio_labels):
            for j, tl in enumerate(text_labels):
                a_set = _top_k_neurons(score_audio[al], valid_layers, k)
                t_set = _top_k_neurons(score_text[tl],  valid_layers, k)
                inter = len(a_set & t_set)
                union = len(a_set | t_set)
                matrix[i, j] = inter / union if union > 0 else 0.0
        all_results[k] = matrix.tolist()

        ax = axes[idx // ncols][idx % ncols]
        im = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=matrix.max() or 1)
        ax.set_xticks(range(len(text_labels)))
        ax.set_yticks(range(len(audio_labels)))
        ax.set_xticklabels(text_labels, rotation=45, ha="right", fontsize=9)
        ax.set_yticklabels(audio_labels, fontsize=9)
        ax.set_xlabel("Text emotion", fontsize=9)
        ax.set_ylabel("Audio emotion", fontsize=9)
        ax.set_title(f"k = {k}", fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        thresh = matrix.max() * 0.6
        for i in range(len(audio_labels)):
            for j in range(len(text_labels)):
                ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center",
                        fontsize=7, color="white" if matrix[i, j] > thresh else "black")

    for idx in range(len(k_values), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle(
        f"Audio vs. Text — Jaccard matrix (ranked by {score_label})\n"
        "rows = audio emotion, cols = text emotion", fontsize=10,
    )
    plt.tight_layout()
    fname = f"jaccard_matrix{suffix}"
    fig.savefig(os.path.join(out_dir, f"{fname}.png"), dpi=150)
    plt.close(fig)

    with open(os.path.join(out_dir, f"{fname}.json"), "w") as f:
        json.dump({str(k): v for k, v in all_results.items()}, f, indent=2)

    print(f"[Method 2 matrix] {fname}.png + .json saved")
    return all_results


# ---------------------------------------------------------------------------
# Method 3 — per-layer Jaccard matrix (fixed k)
# ---------------------------------------------------------------------------

def method3_jaccard_per_layer(
    audio_acts, text_acts, layer_names, out_dir, k: int = 20,
    use_specificity: bool = False,
) -> dict:
    """
    Fix k and draw one Jaccard matrix heatmap per layer.
    Rows = audio emotion, cols = text emotion.
    All panels share the same colorscale so cross-layer comparison is valid.
    """
    audio_labels = sorted(audio_acts.keys())
    text_labels  = sorted(text_acts.keys())
    valid_layers = [l for l in layer_names if l in next(iter(audio_acts.values()))]

    score_audio = compute_specificity(audio_acts) if use_specificity else audio_acts
    score_text  = compute_specificity(text_acts)  if use_specificity else text_acts
    suffix      = "_specificity" if use_specificity else ""
    score_label = "specificity score" if use_specificity else "activation value"

    # Pre-compute all matrices to find global max for a consistent colorscale
    all_matrices: list[np.ndarray] = []
    for layer in valid_layers:
        matrix = np.zeros((len(audio_labels), len(text_labels)))
        for i, al in enumerate(audio_labels):
            if layer not in score_audio.get(al, {}):
                continue
            a_top = set(np.argsort(score_audio[al][layer])[-k:].tolist())
            for j, tl in enumerate(text_labels):
                if layer not in score_text.get(tl, {}):
                    continue
                t_top = set(np.argsort(score_text[tl][layer])[-k:].tolist())
                inter = len(a_top & t_top)
                union = len(a_top | t_top)
                matrix[i, j] = inter / union if union > 0 else 0.0
        all_matrices.append(matrix)

    global_max = max((m.max() for m in all_matrices), default=1.0) or 1.0

    n = len(valid_layers)
    ncols = 4
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows), squeeze=False)

    all_results: dict = {}
    for idx, (layer, matrix) in enumerate(zip(valid_layers, all_matrices)):
        all_results[layer] = matrix.tolist()

        ax = axes[idx // ncols][idx % ncols]
        im = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=global_max)
        ax.set_xticks(range(len(text_labels)))
        ax.set_yticks(range(len(audio_labels)))
        ax.set_xticklabels(text_labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(audio_labels, fontsize=8)
        ax.set_xlabel("Text emotion", fontsize=8)
        ax.set_ylabel("Audio emotion", fontsize=8)
        ax.set_title(f"Layer {idx}", fontsize=9)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        thresh = global_max * 0.6
        for i in range(len(audio_labels)):
            for j in range(len(text_labels)):
                ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center",
                        fontsize=6, color="white" if matrix[i, j] > thresh else "black")

    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle(
        f"Audio vs. Text — per-layer Jaccard matrix (k={k}, ranked by {score_label})\n"
        "rows = audio emotion, cols = text emotion",
        fontsize=11,
    )
    plt.tight_layout()
    fname = f"jaccard_per_layer_k{k}{suffix}"
    fig.savefig(os.path.join(out_dir, f"{fname}.png"), dpi=150)
    plt.close(fig)

    with open(os.path.join(out_dir, f"{fname}.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"[Method 3] {fname}.png + .json saved")
    return all_results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading audio activations ...")
    audio_acts, layer_names = load_audio_mean_activations(args.audio_dir)
    print("Loading text activations ...")
    text_acts, _ = load_text_activations(args.text_dir)

    k_values = list(map(int, args.k_values.split(",")))

    method1_cosine(audio_acts, text_acts, layer_names, args.out_dir)
    method2_jaccard(audio_acts, text_acts, layer_names, args.out_dir, k_values,
                    use_specificity=args.use_specificity)
    method2_jaccard_matrix(audio_acts, text_acts, layer_names, args.out_dir, k_values,
                           use_specificity=args.use_specificity)
    method3_jaccard_per_layer(audio_acts, text_acts, layer_names, args.out_dir,
                              k=args.k_layer, use_specificity=args.use_specificity)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare audio vs. text MLP neuron activations across emotion labels"
    )
    parser.add_argument("--audio_dir", default="./activations/sakura_emotion")
    parser.add_argument("--text_dir", default="./activations/text_emotion")
    parser.add_argument("--out_dir", default="./results/modality_comparison")
    parser.add_argument(
        "--k_values", default="10,20,50,100,200,500",
        help="Comma-separated k values for Jaccard (default: 10,20,50,100,200,500)",
    )
    parser.add_argument(
        "--k_layer", type=int, default=20,
        help="Fixed k for per-layer Jaccard matrix (default: 20)",
    )
    parser.add_argument(
        "--use_specificity", action="store_true",
        help="Rank neurons by specificity score (value - mean of other labels) instead of raw activation value",
    )
    args = parser.parse_args()
    main(args)
