# ExplainableLALM

Probe which MLP neurons in **Qwen2.5-Omni-3B** are specifically associated with each emotion class, and compare whether audio and text representations of the same emotion activate similar neurons.

Dataset: [EmotionSAKURA](https://huggingface.co/datasets/SLLM-multi-hop/EmotionQA) — 5 emotions × 100 speech samples.

---

## Setup

```bash
conda create -n explainableLALM python=3.10 -y
conda activate explainableLALM
pip install -r requirements.txt
```

> Requires a CUDA GPU for extraction steps.

---

## Usage

### Step 1 — Extract audio MLP activations

```bash
python extract_activations.py -o ./activations/sakura_emotion
# add --overwrite to re-extract existing files
```

Runs each audio sample through the model. At each transformer layer, captures the SwiGLU intermediate activations at **audio token positions** and mean-pools across frames.

Outputs: `layers.json`, `metadata.json`, one `<file_id>.npz` per sample (keys `layer_0 … layer_N`, each `(intermediate_dim,)` float16).

### Step 2 — Extract text MLP activations

```bash
python extract_text_activations.py -o ./activations/text_emotion
```

Runs **5 semantically varied prompts per emotion** (e.g. `"The speaker feels happy."`, `"The speaker sounds joyful and cheerful."`, …) and averages activations across them — making the text side comparable to the audio side which averages 100 samples. Same output format as Step 1.

### Step 3 — Identify attribute neurons

```bash
# Print top-20 neurons per emotion
python analyze_neurons.py -d ./activations/sakura_emotion

# Single emotion, top-50, save JSON
python analyze_neurons.py -d ./activations/sakura_emotion \
    --label happy --top_k 50 --save results/neurons.json

# Stricter activation threshold
python analyze_neurons.py -d ./activations/sakura_emotion --threshold 0.5
```

### Step 4 — Compare audio vs. text neurons

```bash
python compare_modalities.py \
    --audio_dir ./activations/sakura_emotion \
    --text_dir  ./activations/text_emotion \
    --out_dir   ./results/modality_comparison
```

Optional flags:
- `--k_values 10,20,50,100,200,500` — k values used for Jaccard (default as shown)
- `--use_specificity` — rank neurons by specificity score instead of raw activation value (outputs `jaccard_overlap_specificity.png/json`)

Outputs written to `--out_dir`:

| File | Content |
|---|---|
| `cosine_similarity.png/json` | Per-layer cosine similarity (Method 1) |
| `jaccard_overlap[_specificity].png/json` | Top-k Jaccard overlap curve (Method 2) |
| `jaccard_matrix[_specificity].png/json` | 5×5 cross-emotion Jaccard heatmap (Method 2b) |

---

## Methods

### Neuron identification (Steps 1 & 3)

The model uses **SwiGLU** MLPs:
```
intermediate = act_fn(gate_proj(x)) * up_proj(x)
output       = down_proj(intermediate)
```
A `forward_pre_hook` on each `down_proj` captures `intermediate` — the true per-neuron value before projection. Only **audio token positions** are retained and mean-pooled.

**Activation rate** — fraction of samples of label $l$ where neuron $k$ exceeds threshold $\tau$:

$$R(l, k) = \mathbb{E}_{x \sim l}\bigl[\mathbf{1}[\text{intermediate}_k(x) > \tau]\bigr]$$

**Specificity score** — how much more active a neuron is for the target label than for others:

$$\text{specificity}(l^*, k) = R(l^*, k) - \frac{1}{|L|-1} \sum_{l \neq l^*} R(l, k)$$

Neurons are ranked by specificity (descending).

> **Note — two variants of specificity score**
> - `analyze_neurons.py` computes specificity on **activation rates** $R(l, k)$ (fraction of samples exceeding threshold $\tau$).
> - `compare_modalities.py --use_specificity` computes specificity on **mean activation values** (the raw intermediate activation averaged across samples), because the text side has only one vector per emotion and no meaningful threshold-based rate.
>
> $$\text{specificity}_{\text{value}}(l^*, k) = \bar{a}_{l^*,k} - \frac{1}{|L|-1} \sum_{l \neq l^*} \bar{a}_{l,k}$$

### Cross-modal comparison (Steps 2 & 4)

#### Method 1 — Layer-wise cosine similarity

For each (emotion, layer) pair, compare the **mean audio activation vector** (across all 100 samples) with the single **text activation vector** from `"The speaker feels {emotion}."`:

$$\text{sim}(l, \text{layer}) = \cos\!\left(\bar{\mathbf{a}}_{l,\text{layer}},\; \mathbf{t}_{l,\text{layer}}\right)$$

Plotting similarity vs. layer index reveals at which depths audio and text representations align.

#### Method 2 — Top-k neuron Jaccard overlap

For each emotion and varying $k$, take the top-$k$ neurons by mean activation value from each modality and compute:

$$J(k) = \frac{|\text{top-}k_{\text{audio}} \cap \text{top-}k_{\text{text}}|}{|\text{top-}k_{\text{audio}} \cup \text{top-}k_{\text{text}}|}$$

A random baseline is overlaid. A companion **5×5 cross-emotion heatmap** (`jaccard_matrix`) shows Jaccard for every (audio_emotion, text_emotion) pair — if diagonal entries are higher than off-diagonal entries, the overlap is emotion-specific rather than a general model behaviour. Values significantly above the baseline indicate genuine shared neuron usage across modalities.

**Random baseline derivation** — if the two top-$k$ sets were drawn uniformly at random from $N$ total neurons, by linearity of expectation:

$$\mathbb{E}[|A \cap B|] = N \cdot \frac{k}{N} \cdot \frac{k}{N} = \frac{k^2}{N}$$

$$\mathbb{E}[J] \approx \frac{\mathbb{E}[|A \cap B|]}{\mathbb{E}[|A \cup B|]} = \frac{k^2/N}{2k - k^2/N} = \frac{k}{2N - k}$$

With $N \approx 308{,}000$ this is effectively $0$ for all $k$ values tested.

---

## Emotion Labels

`disgust` · `sad` · `angry` · `fear` · `happy`
