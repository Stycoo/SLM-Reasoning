#!/bin/bash
set -euo pipefail

# ------------------- Config -------------------
PROJ_DIR="/GLOBALFS/gznwp_3/qxj/shitianyuan/Model-Fusion-public"
DATE="0522"
MODEL_ID="Llama-3.2-3B-Instruct"
VERSION="iterative_reverse_segment_dpo_w_accu_q"
ITERATION_TIME=3
SEGMENT_NUM=3
LOG_DIR="${PROJ_DIR}/log/${DATE}/${VERSION}"
BASE_MODEL="${PROJ_DIR}/saves/model_fusion/${MODEL_ID}/0501/Llama-3.2-3B-Instruct-math-multi-src-csft"
GPUS=(0 1 2 3 4 5 6 7)
NUM_GPUS=${#GPUS[@]}
mkdir -p "$LOG_DIR"
cd "$PROJ_DIR" || { echo "Failed to cd to ${PROJ_DIR}"; exit 1; }

# ------------------- Functions -------------------
activate_env() {
    eval "$(conda shell.bash hook)"
    conda activate "$1"
}

run_rollout_inference() {
    activate_env vllm_zlg

    local seed="$1" chunk_num="$2" chunk_id="$3" input_file="$4" model="$5" output_dir="$6" iter_time="$7" completion_id="$8" log_file="$9"
    local gpu_index=$((chunk_id % NUM_GPUS))
    local gpu="${GPUS[$gpu_index]}"
    
    CUDA_VISIBLE_DEVICES=$gpu python "$PROJ_DIR/rollout_for_math/rollout_for_accumulate_q_v2.py" \
        --input_file "$input_file" \
        --model_name_or_path "$model" \
        --model_id "$MODEL_ID" \
        --output_dir "$output_dir" \
        --seed "$seed" \
        --chunk_num "$chunk_num" \
        --chunk_id "$chunk_id" \
        --iteration_time "$iter_time" \
        --completion_id "$completion_id" > "$log_file" 2>&1 &
}

# ------------------- Iterative Rollout Loop -------------------

for ((i = 3; i <= ITERATION_TIME; i++)); do
    echo "=== Iteration $i started ==="
    ROLLOUT_LOG_DIR="${LOG_DIR}/iter_${i}_rollout"
    mkdir -p "$ROLLOUT_LOG_DIR"

    if (( i == 1 )); then
        activate_env llama_factory_sty

        ORM_SCORED_OUTPUT_DIR="$PROJ_DIR/data/FuseChat-3.0-DPO-Data/$DATE/$MODEL_ID/$VERSION/rollout_forward/completion_with_rm_score/orm_verify"
            
        POST_PROCESS_LOG_FILE="${ROLLOUT_LOG_DIR}/create_align_segment_dpo_dataset_iter_${i}.log"
        python "$PROJ_DIR/rollout_for_math/build_dataset_0520.py" \
            --completion_file_dir "$ORM_SCORED_OUTPUT_DIR" \
            --post_process_type create_align_segment_dpo_dataset \
            --iteration_id "$i" \
            --segment_num "$SEGMENT_NUM" \
            --save_dir $PROJ_DIR/data/FuseChat-3.0-DPO-Data/$DATE/$MODEL_ID/$VERSION/iter_${i} > "$POST_PROCESS_LOG_FILE" 2>&1
        
        # ---------- DPO Training ----------
        TRAINING_LOG_DIR="${LOG_DIR}/training_iter_${i}"
        mkdir -p "$TRAINING_LOG_DIR"
        LOG_FILE="${TRAINING_LOG_DIR}/run_dpo_iter_${i}.log"

        echo -e "\n--- Starting DPO Training: Iteration $i ---"
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 FORCE_TORCHRUN=1 llamafactory-cli train \
            "$PROJ_DIR/recipes/align_dpo/iterative_reverse_segment_dpo_w_accu_q_0520/iter_${i}.yaml" > "$LOG_FILE" 2>&1
        echo "Iteration ${i} completed. Log saved to ${LOG_FILE}"
        
    else
        ROLLOUT_OUTPUT_DIR="$PROJ_DIR/data/FuseChat-3.0-DPO-Data/$DATE/$MODEL_ID/$VERSION/iter_${i}/rollout"
        POST_PROCESS_LOG_FILE="${ROLLOUT_LOG_DIR}/create_rollout_input_data_iter_${i}.log"

        ROLLOUT_INPUT_FILE="$PROJ_DIR/data/FuseChat-3.0-DPO-Data/0520/$MODEL_ID/$VERSION/rollout_forward/completion_with_rm_score/orm_verify/all_step_rollout_orm_verified.json"

        python "$PROJ_DIR/rollout_for_math/build_dataset_0520.py" \
            --segment_rollout_input_file "$ROLLOUT_INPUT_FILE" \
            --completion_file_dir "$ROLLOUT_OUTPUT_DIR" \
            --iteration_id "$i" \
            --post_process_type create_segment_rollout_input_dataset > "$POST_PROCESS_LOG_FILE" 2>&1

        SEEDS=(100 13 21 42 79)
        MODEL="$PROJ_DIR/saves/model_fusion/$MODEL_ID/$DATE/$VERSION/iter_$((i - 1))" 
        ROLLOUT_INPUT_JSON="$ROLLOUT_OUTPUT_DIR/input.json"
        ROLLOUT_i_LOG_DIR="${ROLLOUT_LOG_DIR}/rollout"
        mkdir -p "$ROLLOUT_i_LOG_DIR"

        for seed in "${SEEDS[@]}"; do
            for ((chunk_id = 0; chunk_id < NUM_GPUS; chunk_id++)); do
                log_file="${ROLLOUT_i_LOG_DIR}/run_infer_chunk_${chunk_id}_seed_${seed}.log"
                run_rollout_inference "$seed" "$NUM_GPUS" "$chunk_id" "$ROLLOUT_INPUT_JSON" "$MODEL" "$ROLLOUT_OUTPUT_DIR" "$ITERATION_TIME" "$i" "$log_file"
            done
            wait
        done
        
        python "$PROJ_DIR/rollout_for_math/build_dataset_0520.py" \
            --completion_file_dir "$ROLLOUT_OUTPUT_DIR" \
            --post_process_type merge_multi_seeds_segment_rollout_chunks        
    fi

    POST_PROCESS_LOG_FILE="${ROLLOUT_LOG_DIR}/create_segment_rollout_scoring_input_data_iter_${i}.log"
    python "$PROJ_DIR/rollout_for_math/build_dataset_0520.py" \
        --completion_file_dir "$ROLLOUT_OUTPUT_DIR" \
        --post_process_type create_segment_rollout_scoring_input_dataset \
        --iteration_id "$i" > "$POST_PROCESS_LOG_FILE" 2>&1

    # ---------- Rule Verification ----------
    activate_env llama_factory_sty
    SCORING_INPUT_FILE="$ROLLOUT_OUTPUT_DIR/iter_${i}_rollout_scoring_input.json"
    RM_SCORED_OUTPUT_DIR="$ROLLOUT_OUTPUT_DIR/completion_with_rm_score"
    RULE_LOG="${ROLLOUT_LOG_DIR}/rule_verify_iter_${i}.log"

    python "$PROJ_DIR/rollout_for_math/rollout_verification.py" \
        --input_file "$SCORING_INPUT_FILE" \
        --output_dir "$RM_SCORED_OUTPUT_DIR" \
        --verfiy_mode Rule \
        --chunk_num 1 --chunk_id 1 > "$RULE_LOG" 2>&1

    # ---------- ORM Verification ----------
    ORM_VERIFY_DIR="${ROLLOUT_LOG_DIR}/orm_verify"
    ORM_PATH="/GLOBALFS/gznwp_3/qxj/models/ArmoRM-Llama3-8B-v0.1"
    ORM_SCORED_OUTPUT_DIR="$RM_SCORED_OUTPUT_DIR/orm_verify"
    mkdir -p "$ORM_VERIFY_DIR"

    for ((j = 0; j < NUM_GPUS; j++)); do
        log_file="${ORM_VERIFY_DIR}/scoring_iter_${i}_chunk_${j}.log"
        CUDA_VISIBLE_DEVICES=${GPUS[$j % NUM_GPUS]} python "$PROJ_DIR/rollout_for_math/rollout_verification.py" \
            --input_file "$RM_SCORED_OUTPUT_DIR/all_step_rollout_rule_verified.json" \
            --model_name_or_path "$ORM_PATH" \
            --output_dir "$ORM_SCORED_OUTPUT_DIR" \
            --verfiy_mode ORM \
            --chunk_num "$NUM_GPUS" \
            --chunk_id "$j" > "$log_file" 2>&1 &
    done
    wait

    # ---------- Post-processing ----------
    python "$PROJ_DIR/rollout_for_math/build_dataset_0520.py" \
        --completion_file_dir "$ORM_SCORED_OUTPUT_DIR" \
        --post_process_type merge_completion_chunks

    POST_PROCESS_LOG_FILE="${ROLLOUT_LOG_DIR}/create_align_segment_dpo_dataset_iter_${i}.log"
    python "$PROJ_DIR/rollout_for_math/build_dataset_0520.py" \
        --completion_file_dir "$ORM_SCORED_OUTPUT_DIR" \
        --post_process_type create_align_segment_dpo_dataset \
        --iteration_id "$i" \
        --save_dir $PROJ_DIR/data/FuseChat-3.0-DPO-Data/$DATE/$MODEL_ID/$VERSION/iter_${i} > "$POST_PROCESS_LOG_FILE" 2>&1
    
    # ---------- DPO Training ----------
    TRAINING_LOG_DIR="${LOG_DIR}/training_iter_${i}"
    mkdir -p "$TRAINING_LOG_DIR"
    LOG_FILE="${TRAINING_LOG_DIR}/run_dpo_iter_${i}.log"

    echo -e "\n--- Starting DPO Training: Iteration $i ---"
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 FORCE_TORCHRUN=1 llamafactory-cli train \
        "$PROJ_DIR/recipes/align_dpo/iterative_reverse_segment_dpo_w_accu_q_inherit_0520/iter_${i}.yaml" > "$LOG_FILE" 2>&1
    echo "Iteration ${i} completed. Log saved to ${LOG_FILE}"

    # exit 0

done