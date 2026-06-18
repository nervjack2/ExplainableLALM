#!/usr/bin/env bash
# Run extract_activations.py + extract_text_activations.py + compare_modalities.py
# for all 4 SAKURA tasks, for audio-flamingo-3.
set -euo pipefail

TASKS=(emotion animal gender language)
MODEL="audio-flamingo-3"

for TASK in "${TASKS[@]}"; do
    echo "=== [$MODEL] task=$TASK ==="

    AUDIO_DIR="./activations/${MODEL}_sakura_${TASK}"
    TEXT_DIR="./activations/${MODEL}_text_${TASK}"
    OUT_DIR="./results/${MODEL}_modality_comparison_${TASK}"

    python extract_activations.py \
        --task "$TASK" --model "$MODEL" -o "$AUDIO_DIR"

    python extract_text_activations.py \
        --task "$TASK" --model "$MODEL" -o "$TEXT_DIR"

    python compare_modalities.py \
        --audio_dir "$AUDIO_DIR" \
        --text_dir "$TEXT_DIR" \
        --out_dir "$OUT_DIR" \
        --label_name "$TASK" \
        --use_specificity
done
