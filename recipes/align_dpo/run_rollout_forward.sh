#!/bin/bash
set -eu  # Exit on error or undefined variables

# -------- Configuration --------
PROJ_DIR='/GLOBALFS/gznwp_3/qxj/shitianyuan/Model-Fusion-public'
DATE='0525'
MODEL_ID='Llama-3.2-3B-Instruct'
VERSION='iterative_reverse_segment_dpo_w_accu_q'
SEGMENT_NUM=3

ROLLOUT_INPUT="$PROJ_DIR/data/FuseChat-3.0-DPO-Data/openmathinstruct2-Qwen2.5-72B-Instruct-2527.json"
ROLLOUT_OUTPUT="$PROJ_DIR/data/FuseChat-3.0-DPO-Data/$DATE/$MODEL_ID/$VERSION/rollout_forward"
MODEL="$PROJ_DIR/saves/model_fusion/$MODEL_ID/0525/Llama-3.2-3B-Instruct-sft-warmup"
LOG_DIR="$PROJ_DIR/log/$DATE/$VERSION/rollout_forward"
SEEDS=(100 13 21 42 79)
GPUS=(0 1 2 3 4 5 6 7)

NUM_GPUS=${#GPUS[@]}

mkdir -p "$LOG_DIR"
cd "$PROJ_DIR"

# -------- Functions --------
activate_env() {
    eval "$(conda shell.bash hook)"
    conda activate "$1"
}

run_rollout_chunk() {
    local seed=$1
    local chunk_id=$2
    local gpu=${GPUS[$((chunk_id % NUM_GPUS))]}
    local log_file="$LOG_DIR/rollout_chunk_${chunk_id}_seed_${seed}.log"

    CUDA_VISIBLE_DEVICES=$gpu python $PROJ_DIR/rollout_for_math/rollout_forward_0520.py \
        --input_file "$ROLLOUT_INPUT" \
        --model_name_or_path "$MODEL" \
        --model_id "$MODEL_ID" \
        --output_dir "$ROLLOUT_OUTPUT" \
        --seed "$seed" \
        --chunk_num "$NUM_GPUS" \
        --max_segment_num "$SEGMENT_NUM" \
        --chunk_id "$chunk_id" > "$log_file" 2>&1 &
}

# -------- Rollout Execution --------
# activate_env vllm_zlg
# for seed in "${SEEDS[@]}"; do
#     for chunk_id in "${!GPUS[@]}"; do
#         run_rollout_chunk "$seed" "$chunk_id"
#     done
#     wait  # Wait for all GPU tasks of current seed
# done

# -------- Post-Processing --------
POST_PROCESS_TYPE="merge_multi_seeds_completion_chunks_v2"
POST_LOG="$LOG_DIR/post_merge.log"
# python $PROJ_DIR/rollout_for_math/build_dataset_0520.py \
#     --completion_file_dir "$ROLLOUT_OUTPUT" \
#     --post_process_type "$POST_PROCESS_TYPE" > "$POST_LOG" 2>&1

# -------- RM Scoring (Rule) --------
activate_env llama_factory_sty

RM_SCORING_INPUT="$ROLLOUT_OUTPUT/all_generated_completions.json"
RM_SCORED_OUTPUT_DIR="$ROLLOUT_OUTPUT/completion_with_rm_score"
SCORING_LOG="$LOG_DIR/rule_verify.log"

# python $PROJ_DIR/rollout_for_math/rollout_verification.py \
#     --input_file "$RM_SCORING_INPUT" \
#     --output_dir "$RM_SCORED_OUTPUT_DIR" \
#     --verfiy_mode Rule \
#     --chunk_num 1 > "$SCORING_LOG" 2>&1

# -------- RM Scoring (ORM) --------
ORM_PATH="/GLOBALFS/gznwp_3/qxj/models/ArmoRM-Llama3-8B-v0.1"
ORM_VERIFY_DIR="$LOG_DIR/orm_verify"
ORM_SCORED_OUTPUT_DIR="$RM_SCORED_OUTPUT_DIR/orm_verify"
mkdir -p "$ORM_VERIFY_DIR"

for ((j = 0; j < NUM_GPUS; j++)); do
    gpu=${GPUS[$j]}
    log_file="$ORM_VERIFY_DIR/scoring_chunk_${j}.log"

    CUDA_VISIBLE_DEVICES=$gpu python $PROJ_DIR/rollout_for_math/rollout_verification.py \
        --input_file "$RM_SCORED_OUTPUT_DIR/all_step_rollout_rule_verified.json" \
        --model_name_or_path "$ORM_PATH" \
        --output_dir "$ORM_SCORED_OUTPUT_DIR" \
        --verfiy_mode ORM \
        --chunk_num "$NUM_GPUS" \
        --chunk_id "$j" > "$log_file" 2>&1 &
done
wait

# -------- Final Merge --------
FINAL_POST_PROCESS_TYPE="merge_completion_chunks"
python $PROJ_DIR/rollout_for_math/build_dataset_0520.py \
    --completion_file_dir "$ORM_SCORED_OUTPUT_DIR" \
    --post_process_type "$FINAL_POST_PROCESS_TYPE"