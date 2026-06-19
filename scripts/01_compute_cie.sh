#!/bin/bash
# Step 1 — Causal head selection.
# Computes the causal indirect effect (CIE) of every attention head, then selects the
# top-k heads used by the self-function-vector experiments.
#
# main_icl/main.py --cie 1 computes the CIE of ONE layer per run (--cie_layer L) and saves
# {L}_mean_effect.pt. We therefore loop over all layers, then aggregate with
# select_top_heads.py -> top_heads.json (+ mean_head_act.pt).
#
# Run from the repository root.
set -e

MODEL=${MODEL:-meta-llama/Llama-2-7b-hf}
DATA=${DATA:-wordnet}
ANSWER_TYPE=${ANSWER_TYPE:-choice}
NUM_SHOTS=${NUM_SHOTS:-10}
NUM_LAYERS=${NUM_LAYERS:-32}     # 32 for 7B, 40 for 13B, 80 for 70B
OUTPUT_DIR=${OUTPUT_DIR:-results}

# 1a. Compute CIE for every layer (each writes ${OUTPUT_DIR}/.../cie_seed24/{L}_mean_effect.pt).
#     The per-layer loop is embarrassingly parallel: dispatch each layer to its own
#     GPU/cluster job to speed this up.
for ((L=0; L<NUM_LAYERS; L++)); do
  python main_icl/main.py \
    --model "$MODEL" \
    --data_path dataset_files/icl \
    --data "$DATA" \
    --answer_type "$ANSWER_TYPE" \
    --num_shots "$NUM_SHOTS" \
    --cie 1 \
    --cie_layer "$L" \
    --layer_end "$NUM_LAYERS" \
    --output_dir "$OUTPUT_DIR" \
    --use_wandb 0
done

# 1b. Aggregate per-layer CIE and select top-k heads -> top_heads.json.
python select_top_heads.py \
  --output_dir "$OUTPUT_DIR" \
  --num_layers "$NUM_LAYERS" \
  --top_k 100
