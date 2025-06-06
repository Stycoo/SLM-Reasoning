#!/bin/bash
set -eu  # Exit on error or undefined variables

# -------- Configuration --------
PROJ_DIR="/GLOBALFS/gznwp_3/qxj/shitianyuan/Model-Fusion-public"
DATE="0525"
MODEL_ID="Llama-3.2-3B-Instruct"
VERSION="iterative_reverse_segment_dpo_w_accu_q_hybrid_w_seg_sft"
SEGMENT_NUM=3

ROLLOUT_INPUT="$PROJ_DIR/data/FuseChat-3.0-DPO-Data/openmathinstruct2-Qwen2.5-72B-Instruct-2527.json"
ROLLOUT_OUTPUT="$PROJ_DIR/data/FuseChat-3.0-DPO-Data/$DATE/$MODEL_ID/$VERSION/rollout_forward"
MODEL_ROOT_DIR="$PROJ_DIR/saves/model_fusion/$MODEL_ID/0525/sft_segment"
MODELS=('seg_1' 'seg_2' 'seg_3')
LOG_DIR="$PROJ_DIR/log/$DATE/$VERSION/rollout_forward"
SEEDS=(100 13 21 42 79)
GPUS=(0 1 2 3 4 5 6 7)
NUM_GPUS=${#GPUS[@]}

mkdir -p "$LOG_DIR"
cd "$PROJ_DIR" || { echo "Failed to cd to $PROJ_DIR"; exit 1; }

# -------- Functions --------
activate_env() {
    eval "$(conda shell.bash hook)"
    conda activate "$1"
}

run_rollout_inference() {
    local seed="$1" chunk_num="$2" chunk_id="$3" input_file="$4" model="$5"
    local output_dir="$6" iter_time="$7" completion_id="$8" log_file="$9"

    local gpu_index=$((chunk_id % NUM_GPUS))
    local gpu="${GPUS[$gpu_index]}"

    CUDA_VISIBLE_DEVICES="$gpu" python "$PROJ_DIR/rollout_for_math/rollout_in_current_iter_seg_index.py" \
        --forward_or_reverse reverse \
        --input_file "$input_file" \
        --model_name_or_path "$model" \
        --model_id "$MODEL_ID" \
        --output_dir "$output_dir" \
        --seed "$seed" \
        --segment_num "$iter_time" \
        --chunk_num "$chunk_num" \
        --chunk_id "$chunk_id" \
        --iteration_time "$iter_time" \
        --completion_id "$completion_id" > "$log_file" 2>&1 &
}

# -------- Rollout Execution --------
activate_env vllm_zlg

for ((i = 0; i < SEGMENT_NUM; i++)); do
    model="${MODEL_ROOT_DIR}/${MODELS[$i]}"

    ROLLOUT_LOG_DIR="${LOG_DIR}/segment_$((i+1))"
    mkdir -p "$ROLLOUT_LOG_DIR"

    ROLLOUT_OUTPUT_DIR="${ROLLOUT_OUTPUT}/segment_$((i+1))"
    mkdir -p "$ROLLOUT_OUTPUT_DIR"

    for seed in "${SEEDS[@]}"; do
        for chunk_id in "${!GPUS[@]}"; do
            log_file="${ROLLOUT_LOG_DIR}/infer_chunk_${chunk_id}_seed_${seed}.log"
            run_rollout_inference "$seed" "$NUM_GPUS" "$chunk_id" "$ROLLOUT_INPUT" "$model" "$ROLLOUT_OUTPUT_DIR" "$SEGMENT_NUM" "$((i+1))" "$log_file"
        done
        wait
    done

    # -------- Post-Processing --------
    POST_PROCESS_TYPE="merge_multi_seeds_completion_chunks_v2"
    POST_LOG="$ROLLOUT_LOG_DIR/post_merge.log"
    python $PROJ_DIR/rollout_for_math/build_dataset_0520.py \
        --completion_file_dir "$ROLLOUT_OUTPUT_DIR" \
        --post_process_type "$POST_PROCESS_TYPE" > "$POST_LOG" 2>&1

    # -------- RM Scoring (Rule) --------
    activate_env llama_factory_sty

    RM_SCORING_INPUT="$ROLLOUT_OUTPUT_DIR/all_generated_completions.json"
    RM_SCORED_OUTPUT_DIR="$ROLLOUT_OUTPUT_DIR/completion_with_rm_score"
    SCORING_LOG="$ROLLOUT_LOG_DIR/rule_verify.log"

    python $PROJ_DIR/rollout_for_math/rollout_verification.py \
        --input_file "$RM_SCORING_INPUT" \
        --output_dir "$RM_SCORED_OUTPUT_DIR" \
        --verfiy_mode Rule \
        --chunk_num 1 > "$SCORING_LOG" 2>&1

    # -------- RM Scoring (ORM) --------
    ORM_PATH="/GLOBALFS/gznwp_3/qxj/models/ArmoRM-Llama3-8B-v0.1"
    ORM_VERIFY_DIR="$ROLLOUT_LOG_DIR/orm_verify"
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

done

FINAL_POST_PROCESS_TYPE="create_segment_rollout_input_dataset_w_seg_sft"
python $PROJ_DIR/rollout_for_math/build_dataset_0520.py \
    --completion_file_dir "$ROLLOUT_OUTPUT" \
    --post_process_type "$FINAL_POST_PROCESS_TYPE" \
    --segment_num "$SEGMENT_NUM"