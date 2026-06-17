"""
extract_activations.py
======================
Run a SAKURA dataset split through an audio-LALM (see src/models/registry.py
for the supported --model choices) and save the MLP intermediate activations
at audio token positions.

For each sample the script saves one .npz file containing, for every
transformer layer, the *mean* activation across audio frames:
    layer_0, layer_1, …  shape (intermediate_dim,)  dtype float16

A layers.json and metadata.json index file are also written so that
analyze_neurons.py / compare_modalities.py can load and interpret the data.

Usage
-----
    python extract_activations.py --task emotion --model qwen2.5-omni-3b -o ./activations/sakura_emotion
    python extract_activations.py --task animal   --model qwen2.5-omni-7b
    python extract_activations.py --task gender   --model desta2.5-audio
    python extract_activations.py --task language --model audio-flamingo-3
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

PROBE_TEXTS = {
    "emotion":  "What is the emotional tone of the speaker in the audio?",
    "animal":   "What animal is making the sound in the audio?",
    "gender":   "What is the gender of the speaker in the audio?",
    "language": "What language is the speaker speaking in the audio?",
}


def safe_file_id(path: str) -> str:
    return path.replace(os.sep, "_").replace(".", "_")


def main(args: argparse.Namespace) -> None:
    os.makedirs(args.output_dir, exist_ok=True)

    dataset_cls = SAKURA_TASKS[args.task]
    probe_text = PROBE_TEXTS[args.task]

    dataset = dataset_cls()
    print(f"Task: {args.task} | {len(dataset)} samples | labels: {dataset_cls.LABELS}")

    print(f"Loading {args.model} …")
    model_wrapper = build_model(args.model, device="cuda")

    extractor = MLPActivationExtractor(model_wrapper.model)
    layer_names = extractor.layer_names
    print(f"Registered hooks on {len(layer_names)} MLP layers")

    with open(os.path.join(args.output_dir, "layers.json"), "w") as f:
        json.dump(layer_names, f, indent=2)

    metadata: list[dict] = []

    for idx in tqdm(range(len(dataset)), desc="Extracting"):
        sample = dataset[idx]
        audio: np.ndarray = sample["audio"]
        label: str = sample["label"]
        file_id: str = safe_file_id(sample["id"])

        out_path = os.path.join(args.output_dir, f"{file_id}.npz")

        if os.path.isfile(out_path) and not args.overwrite:
            metadata.append({"file_id": file_id, "label": label})
            continue

        extractor.clear()

        with torch.inference_mode():
            inputs = model_wrapper.prepare_inputs(audio, probe_text)
            if idx == 0:
                response = model_wrapper.generate(inputs)
                print(f"\n[sanity check] sample 0 response:\n{response}\n")
                extractor.clear()
            model_wrapper.model(**inputs)

        audio_positions = model_wrapper.find_audio_positions(inputs["input_ids"])

        if not audio_positions:
            print(f"  [warn] no audio tokens found for sample {idx} – skipping")
            continue

        frame_acts = extractor.get_audio_frame_activations(audio_positions)

        save_dict = {}
        for li, lname in enumerate(layer_names):
            if lname not in frame_acts:
                continue
            mean_act = frame_acts[lname].mean(dim=0).numpy().astype(np.float16)
            save_dict[f"layer_{li}"] = mean_act

        np.savez_compressed(out_path, **save_dict)
        metadata.append({"file_id": file_id, "label": label})

    with open(os.path.join(args.output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    extractor.remove_hooks()
    print(f"\nDone. {len(metadata)} samples saved to '{args.output_dir}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract MLP activations for a SAKURA task")
    parser.add_argument("--task", choices=list(SAKURA_TASKS.keys()), default="emotion",
                        help="SAKURA task to process (default: emotion)")
    parser.add_argument("--model", choices=list(MODEL_REGISTRY.keys()), default="qwen2.5-omni-3b",
                        help="Model to extract activations from (default: qwen2.5-omni-3b)")
    parser.add_argument("-o", "--output_dir", default=None,
                        help="Output directory (default: ./activations/<model>_sakura_<task>)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-extract even if the output file already exists")
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = f"./activations/{args.model}_sakura_{args.task}"

    main(args)
