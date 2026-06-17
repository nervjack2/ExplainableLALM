"""
extract_text_activations.py
============================
Run multiple text-only forward passes per label through an audio-LALM
(see src/models/registry.py for the supported --model choices) and save
the mean MLP activations across all prompts of that label.

Using multiple prompts per label reduces prompt-specific noise and makes
the text side comparable to the audio side which averages across many samples.

Output format is identical to extract_activations.py so that
compare_modalities.py can load both directories with the same loader.

Usage
-----
    # Default: average activations over all tokens in each prompt
    python extract_text_activations.py --task emotion  -o ./activations/text_emotion
    python extract_text_activations.py --task animal   -o ./activations/text_animal
    python extract_text_activations.py --task gender   -o ./activations/text_gender
    python extract_text_activations.py --task language -o ./activations/text_language

    # Label-word-only: activations at the label keyword token(s) only
    # (e.g. 'happy' instead of all tokens in "The speaker feels happy.")
    python extract_text_activations.py --task emotion  --label_word_only
    python extract_text_activations.py --task animal   --label_word_only
    python extract_text_activations.py --task gender   --label_word_only
    python extract_text_activations.py --task language --label_word_only
"""

import argparse
import json
import os

import numpy as np
import torch
from tqdm import tqdm

from src.dataset.sakura import SAKURA_TASKS
from src.models.registry import MODEL_REGISTRY, build_model
from src.neuron.extractor import MLPActivationExtractor

# ---------------------------------------------------------------------------
# Label keywords: for --label_word_only mode.
# Each entry lists candidate surface forms to search for (tried in order).
# The first one found in the tokenised prompt is used.
# ---------------------------------------------------------------------------
LABEL_KEYWORDS: dict[str, dict[str, list[str]]] = {
    "emotion": {
        "happy":   ["happy", "happiness", "joyful", "glad"],
        "sad":     ["sad", "sadness", "sorrowful", "mournful"],
        "angry":   ["angry", "anger", "furious", "wrathful"],
        "fear":    ["afraid", "fear", "fearful", "frightened", "scared"],
        "disgust": ["disgusted", "disgust", "revolted", "repulsed"],
    },
    "animal": {label: [label]
               for label in ["cat", "cow", "crow", "dog", "frog",
                             "hen", "pig", "rooster", "sheep"]},
    "gender": {
        "female": ["female", "woman"],
        "male":   ["male", "man"],
    },
    "language": {lang: [lang]
                 for lang in ["Chinese", "English", "French", "German",
                              "Italian", "Japanese", "Korean", "Spanish"]},
}


def _find_label_word_positions(model_wrapper, input_ids: "torch.Tensor",
                               keywords: list[str]) -> tuple[list[int], str | None]:
    """Return (positions, matched_keyword) for the first keyword found, or ([], None).

    BPE tokenizers encode a word differently depending on whether it has a
    preceding space (e.g. 'happy' standalone vs ' happy' mid-sentence).
    We try the space-prefixed variant first since keywords appear inside
    sentences, then fall back to the bare form.
    """
    for kw in keywords:
        for variant in (" " + kw, kw):
            positions = model_wrapper.find_text_content_positions(input_ids, variant)
            if positions:
                return positions, kw
    return [], None


TEXT_PROMPTS: dict[str, dict[str, list[str]]] = {
    "emotion": {
        "happy": [
            "The speaker feels happy.",
            "The speaker sounds joyful and cheerful.",
            "This is a happy and upbeat voice.",
            "The speaker is expressing happiness.",
            "The emotional tone of the speaker is positive and glad.",
        ],
        "sad": [
            "The speaker feels sad.",
            "The speaker sounds sorrowful and melancholic.",
            "This is a sad and gloomy voice.",
            "The speaker is expressing sadness.",
            "The emotional tone of the speaker is mournful.",
        ],
        "angry": [
            "The speaker feels angry.",
            "The speaker sounds furious and aggressive.",
            "This is an angry and intense voice.",
            "The speaker is expressing anger.",
            "The emotional tone of the speaker is hostile and wrathful.",
        ],
        "fear": [
            "The speaker feels afraid.",
            "The speaker sounds fearful and anxious.",
            "This is a frightened and trembling voice.",
            "The speaker is expressing fear.",
            "The emotional tone of the speaker is scared and nervous.",
        ],
        "disgust": [
            "The speaker feels disgusted.",
            "The speaker sounds revolted and repulsed.",
            "This is a disgusted and contemptuous voice.",
            "The speaker is expressing disgust.",
            "The emotional tone of the speaker is disapproving and disgusted.",
        ],
    },
    "animal": {
        "cat": [
            "The sound in the audio is from a cat.",
            "This audio contains the sound of a cat meowing.",
            "A cat is making the sound in this audio.",
            "The animal making this sound is a cat.",
            "This is the sound of a cat.",
        ],
        "cow": [
            "The sound in the audio is from a cow.",
            "This audio contains the sound of a cow mooing.",
            "A cow is making the sound in this audio.",
            "The animal making this sound is a cow.",
            "This is the sound of a cow.",
        ],
        "crow": [
            "The sound in the audio is from a crow.",
            "This audio contains the sound of a crow cawing.",
            "A crow is making the sound in this audio.",
            "The animal making this sound is a crow.",
            "This is the sound of a crow.",
        ],
        "dog": [
            "The sound in the audio is from a dog.",
            "This audio contains the sound of a dog barking.",
            "A dog is making the sound in this audio.",
            "The animal making this sound is a dog.",
            "This is the sound of a dog.",
        ],
        "frog": [
            "The sound in the audio is from a frog.",
            "This audio contains the sound of a frog croaking.",
            "A frog is making the sound in this audio.",
            "The animal making this sound is a frog.",
            "This is the sound of a frog.",
        ],
        "hen": [
            "The sound in the audio is from a hen.",
            "This audio contains the sound of a hen clucking.",
            "A hen is making the sound in this audio.",
            "The animal making this sound is a hen.",
            "This is the sound of a hen.",
        ],
        "pig": [
            "The sound in the audio is from a pig.",
            "This audio contains the sound of a pig oinking.",
            "A pig is making the sound in this audio.",
            "The animal making this sound is a pig.",
            "This is the sound of a pig.",
        ],
        "rooster": [
            "The sound in the audio is from a rooster.",
            "This audio contains the sound of a rooster crowing.",
            "A rooster is making the sound in this audio.",
            "The animal making this sound is a rooster.",
            "This is the sound of a rooster.",
        ],
        "sheep": [
            "The sound in the audio is from a sheep.",
            "This audio contains the sound of a sheep bleating.",
            "A sheep is making the sound in this audio.",
            "The animal making this sound is a sheep.",
            "This is the sound of a sheep.",
        ],
    },
    "gender": {
        "female": [
            "The speaker in the audio is female.",
            "This is a female voice.",
            "The voice belongs to a woman.",
            "The speaker is a woman.",
            "The gender of the speaker is female.",
        ],
        "male": [
            "The speaker in the audio is male.",
            "This is a male voice.",
            "The voice belongs to a man.",
            "The speaker is a man.",
            "The gender of the speaker is male.",
        ],
    },
    "language": {
        "Chinese": [
            "The speaker is speaking Chinese.",
            "This audio is in the Chinese language.",
            "The language spoken in the audio is Chinese.",
            "The speaker is using Chinese.",
            "Chinese is being spoken in this audio.",
        ],
        "English": [
            "The speaker is speaking English.",
            "This audio is in the English language.",
            "The language spoken in the audio is English.",
            "The speaker is using English.",
            "English is being spoken in this audio.",
        ],
        "French": [
            "The speaker is speaking French.",
            "This audio is in the French language.",
            "The language spoken in the audio is French.",
            "The speaker is using French.",
            "French is being spoken in this audio.",
        ],
        "German": [
            "The speaker is speaking German.",
            "This audio is in the German language.",
            "The language spoken in the audio is German.",
            "The speaker is using German.",
            "German is being spoken in this audio.",
        ],
        "Italian": [
            "The speaker is speaking Italian.",
            "This audio is in the Italian language.",
            "The language spoken in the audio is Italian.",
            "The speaker is using Italian.",
            "Italian is being spoken in this audio.",
        ],
        "Japanese": [
            "The speaker is speaking Japanese.",
            "This audio is in the Japanese language.",
            "The language spoken in the audio is Japanese.",
            "The speaker is using Japanese.",
            "Japanese is being spoken in this audio.",
        ],
        "Korean": [
            "The speaker is speaking Korean.",
            "This audio is in the Korean language.",
            "The language spoken in the audio is Korean.",
            "The speaker is using Korean.",
            "Korean is being spoken in this audio.",
        ],
        "Spanish": [
            "The speaker is speaking Spanish.",
            "This audio is in the Spanish language.",
            "The language spoken in the audio is Spanish.",
            "The speaker is using Spanish.",
            "Spanish is being spoken in this audio.",
        ],
    },
}


def main(args: argparse.Namespace) -> None:
    os.makedirs(args.output_dir, exist_ok=True)

    dataset_cls = SAKURA_TASKS[args.task]
    prompts_by_label = TEXT_PROMPTS[args.task]

    print(f"Task: {args.task} | labels: {dataset_cls.LABELS}")
    print(f"Loading {args.model} ...")
    model_wrapper = build_model(args.model, device="cuda")
    extractor = MLPActivationExtractor(model_wrapper.model)
    layer_names = extractor.layer_names
    print(f"Registered hooks on {len(layer_names)} MLP layers")

    with open(os.path.join(args.output_dir, "layers.json"), "w") as f:
        json.dump(layer_names, f, indent=2)

    metadata: list[dict] = []

    for label in dataset_cls.LABELS:
        prompts = prompts_by_label[label]
        keywords = LABEL_KEYWORDS[args.task][label] if args.label_word_only else None
        mode_desc = f"keyword {keywords}" if args.label_word_only else "full text"
        print(f"\n[{label}] processing {len(prompts)} prompts ({mode_desc}) ...")

        prompt_acts: list[dict[int, np.ndarray]] = []

        for text in tqdm(prompts, desc=f"  {label}"):
            extractor.clear()
            with torch.inference_mode():
                inputs = model_wrapper.prepare_text_inputs(text)
                model_wrapper.model(**inputs)

            if args.label_word_only:
                text_positions, matched_kw = _find_label_word_positions(
                    model_wrapper, inputs["input_ids"], keywords
                )
                if not text_positions:
                    print(f"  [warn] no label keyword found in '{text}' — skipping")
                    continue
            else:
                text_positions = model_wrapper.find_text_content_positions(inputs["input_ids"], text)
                if not text_positions:
                    print(f"  [warn] could not locate tokens for '{text}' — skipping")
                    continue

            frame_acts = extractor.get_audio_frame_activations(text_positions)
            acts = {
                li: frame_acts[lname].mean(dim=0).numpy().astype(np.float32)
                for li, lname in enumerate(layer_names)
                if lname in frame_acts
            }
            prompt_acts.append(acts)

        if not prompt_acts:
            print(f"  [warn] no valid prompts for '{label}' — skipping")
            continue

        save_dict: dict[str, np.ndarray] = {}
        for li in range(len(layer_names)):
            vecs = [pa[li] for pa in prompt_acts if li in pa]
            if vecs:
                save_dict[f"layer_{li}"] = np.stack(vecs).mean(axis=0).astype(np.float16)

        out_path = os.path.join(args.output_dir, f"{label}.npz")
        np.savez_compressed(out_path, **save_dict)
        metadata.append({"file_id": label, "label": label})

    with open(os.path.join(args.output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    extractor.remove_hooks()
    print(f"\nDone. {len(metadata)} labels saved to '{args.output_dir}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract text-only MLP activations for each label of a SAKURA task"
    )
    parser.add_argument("--task", choices=list(SAKURA_TASKS.keys()), default="emotion",
                        help="SAKURA task to process (default: emotion)")
    parser.add_argument("--model", choices=list(MODEL_REGISTRY.keys()), default="qwen2.5-omni-3b",
                        help="Model to load (default: qwen2.5-omni-3b)")
    parser.add_argument("-o", "--output_dir", default=None,
                        help="Output directory (default: ./activations/text_<task> or "
                             "./activations/text_<task>_labelword with --label_word_only)")
    parser.add_argument("--label_word_only", action="store_true",
                        help="Extract activations at label keyword token(s) only "
                             "(e.g. 'happy' instead of the full sentence). "
                             "Keywords are defined in LABEL_KEYWORDS.")
    args = parser.parse_args()

    if args.output_dir is None:
        suffix = "_labelword" if args.label_word_only else ""
        args.output_dir = f"./activations/{args.model}_text_{args.task}{suffix}"

    main(args)
