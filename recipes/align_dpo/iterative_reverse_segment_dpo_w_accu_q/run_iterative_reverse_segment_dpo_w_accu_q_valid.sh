#!/bin/bash
set -euo pipefail

# ------------------- Config -------------------
PROJ_DIR="/GLOBALFS/gznwp_3/qxj/shitianyuan/Model-Fusion-public"
DATE="0514"
MODEL_ID="Llama-3.2-3B-Instruct"
VERSION="iterative_reverse_segment_dpo_w_accu_q"
ITERATION_TIME=2
LOG_DIR="${PROJ_DIR}/log/${DATE}/${VERSION}"
BASE_MODEL="${PROJ_DIR}/saves/model_fusion/${MODEL_ID}/0501/Llama-3.2-3B-Instruct-math-multi-src-csft"
GPUS=(4 5 6 7)
NUM_GPUS=${#GPUS[@]}
mkdir -p "$LOG_DIR"
cd "$PROJ_DIR" || { echo "Failed to cd to ${PROJ_DIR}"; exit 1; }

# ------------------- Functions -------------------

activate_env() {
    eval "$(conda shell.bash hook)"
    conda activate "$1"
}

run_rollout_inference() {
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
activate_env vllm_zlg

for ((i = 2; i <= ITERATION_TIME; i++)); do
    echo "=== Iteration $i started ==="
    ROLLOUT_LOG_DIR="${LOG_DIR}/valid_iter_${i}_rollout"
    mkdir -p "$ROLLOUT_LOG_DIR"

    for ((r_id = 2; r_id <= i + 1; r_id++)); do
        ROLLOUT_OUTPUT_DIR="$PROJ_DIR/data/FuseChat-3.0-DPO-Data/$DATE/$MODEL_ID/$VERSION/iter_${i}/valid_rollout_${r_id}"
        POST_PROCESS_LOG_FILE="${ROLLOUT_LOG_DIR}/create_rollout_input_data_iter_${i}_rollout_${r_id}.log"

        if (( r_id == 1 )); then
            ROLLOUT_INPUT_FILE="$PROJ_DIR/data/FuseChat-3.0-DPO-Data/$DATE/$MODEL_ID/hard_prompt/valid_rollout_input.json"
        else
            ROLLOUT_INPUT_FILE="$PROJ_DIR/data/FuseChat-3.0-DPO-Data/$DATE/$MODEL_ID/$VERSION/iter_${i}/valid_rollout_$((r_id-1))/all_generated_completions.json"
        fi

        python "$PROJ_DIR/rollout_for_math/build_dataset.py" \
            --segment_rollout_input_file "$ROLLOUT_INPUT_FILE" \
            --completion_file_dir "$ROLLOUT_OUTPUT_DIR" \
            --iteration_time "$ITERATION_TIME" \
            --rollout_id "$r_id" \
            --iteration_id "$i" \
            --post_process_type create_segment_rollout_input_valid_dataset > "$POST_PROCESS_LOG_FILE" 2>&1

        SEEDS=( $( (( r_id == 1 )) && echo "100" || echo "10" ) ) # 100 13 21 42 79

        if [[ $((i + 1 - r_id)) -eq 0 ]]; then
            MODEL="$BASE_MODEL"
        else
            MODEL="$PROJ_DIR/saves/model_fusion/$MODEL_ID/$DATE/$VERSION/iter_$((i + 1 - r_id))"
        fi

        ROLLOUT_INPUT_JSON="$ROLLOUT_OUTPUT_DIR/input.json"
        ROLLOUT_i_LOG_DIR="${ROLLOUT_LOG_DIR}/rollout_${r_id}"
        mkdir -p "$ROLLOUT_i_LOG_DIR"

        for seed in "${SEEDS[@]}"; do
            for ((chunk_id = 0; chunk_id < NUM_GPUS; chunk_id++)); do
                log_file="${ROLLOUT_i_LOG_DIR}/run_infer_chunk_${chunk_id}_seed_${seed}.log"
                run_rollout_inference "$seed" "$NUM_GPUS" "$chunk_id" "$ROLLOUT_INPUT_JSON" "$MODEL" "$ROLLOUT_OUTPUT_DIR" "$ITERATION_TIME" "$i" "$log_file"
            done
            wait
        done
        
        python "$PROJ_DIR/rollout_for_math/build_dataset.py" \
            --completion_file_dir "$ROLLOUT_OUTPUT_DIR" \
            --post_process_type merge_multi_seeds_segment_rollout_chunks

        POST_PROCESS_LOG_FILE="${ROLLOUT_LOG_DIR}/create_segment_rollout_scoring_input_data_iter_${i}.log"
        python "$PROJ_DIR/rollout_for_math/build_dataset.py" \
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

        exit 0

    done

    # POST_PROCESS_LOG_FILE="${ROLLOUT_LOG_DIR}/create_segment_rollout_scoring_input_data_iter_${i}.log"
    # python "$PROJ_DIR/rollout_for_math/build_dataset.py" \
    #     --completion_file_dir "$ROLLOUT_OUTPUT_DIR" \
    #     --post_process_type create_segment_rollout_scoring_input_dataset \
    #     --iteration_id "$i" > "$POST_PROCESS_LOG_FILE" 2>&1

    # # ---------- Rule Verification ----------
    # activate_env llama_factory_sty

    # SCORING_INPUT_FILE="$ROLLOUT_OUTPUT_DIR/iter_${i}_rollout_scoring_input.json"
    # RM_SCORED_OUTPUT_DIR="$ROLLOUT_OUTPUT_DIR/completion_with_rm_score"
    # RULE_LOG="${ROLLOUT_LOG_DIR}/rule_verify_iter_${i}.log"

    # python "$PROJ_DIR/rollout_for_math/rollout_verification.py" \
    #     --input_file "$SCORING_INPUT_FILE" \
    #     --output_dir "$RM_SCORED_OUTPUT_DIR" \
    #     --verfiy_mode Rule \
    #     --chunk_num 1 --chunk_id 1 > "$RULE_LOG" 2>&1

done