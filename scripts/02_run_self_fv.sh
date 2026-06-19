#!/bin/bash
# Step 2 — Self-function-vector uncertainty decomposition (main experiment).
# main_icl/main.py builds a self-function vector from the prompt's own activations at the
# selected heads (--use_self_fv 1, needs only --top_heads_path) and decomposes predictive
# uncertainty. The same run also evaluates the function-vector baseline for direct
# comparison (--use_fv 1, reads mean_head_act.pt). (Total entropy is always computed.)
#
# The intervention layer defaults to model depth // 3 (the paper's setting) when
# --interv_layer is not given.
#
# Run from the repository root, after scripts/01_compute_cie.sh.
set -e

MODEL=${MODEL:-meta-llama/Llama-2-7b-hf}
DATA=${DATA:-wordnet}
ANSWER_TYPE=${ANSWER_TYPE:-choice}
NUM_SHOTS=${NUM_SHOTS:-10}
NUM_LAYERS=${NUM_LAYERS:-32}        # 32 for 7B, 40 for 13B, 80 for 70B
NUM_TOP_HEADS=${NUM_TOP_HEADS:-20}
OUTPUT_DIR=${OUTPUT_DIR:-results}
SEED=${SEED:-42}
CIE_SEED=${CIE_SEED:-24}

CIE_DIR=${OUTPUT_DIR}/${DATA}/${ANSWER_TYPE}/${NUM_SHOTS}_shot/seed${SEED}/cie_seed${CIE_SEED}
TOP_HEADS=${TOP_HEADS:-${CIE_DIR}/top_heads.json}
MEAN_ACT=${MEAN_ACT:-${CIE_DIR}/mean_head_act.pt}

python main_icl/main.py \
  --model "$MODEL" \
  --data_path dataset_files/icl \
  --data "$DATA" \
  --answer_type "$ANSWER_TYPE" \
  --num_shots "$NUM_SHOTS" \
  --seed "$SEED" \
  --cie_seed "$CIE_SEED" \
  --layer_end "$NUM_LAYERS" \
  --num_top_heads "$NUM_TOP_HEADS" \
  --top_heads_path "$TOP_HEADS" \
  --mean_head_act_path "$MEAN_ACT" \
  --use_self_fv 1 \
  --use_fv 1 \
  --use_wandb 0
