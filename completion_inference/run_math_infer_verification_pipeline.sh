#!/bin/bash

# ----------------------- ENV SETUP -----------------------
activate_env() {
    eval "$(conda shell.bash hook)"
    conda activate "$1"
}

activate_env vllm_zlg

PROJ_DIR="/GLOBALFS/gznwp_3/qxj/shitianyuan/Model-Fusion-public"
cd "$PROJ_DIR" || exit 1

# INPUT_FILE="$PROJ_DIR/data/FuseChat-3.0-DPO-Data/0514/Llama-3.2-3B-Instruct/hard_prompt/valid_rollout_input.json"
INPUT_FILE=$PROJ_DIR/data/FuseChat-3.0-DPO-Data/0525/Llama-3.2-3B-Instruct/iterative_reverse_segment_dpo_w_accu_q_hybrid/rollout_forward/completion_with_rm_score/all_step_rollout_rule_verified.json
LOG_DIR="$PROJ_DIR/log/0525"
mkdir -p "$LOG_DIR"
exec > "$LOG_DIR/run_math_infer_verification_pipeline.log" 2>&1

# ----------------------- CONFIG --------------------------
SEEDS=(100 13 21 42 79)
GPUS=(0 1 2 3 4 5 6 7)
NUM_GPUS=${#GPUS[@]}
CHUNK_NUM=$NUM_GPUS

MODEL=${PROJ_DIR}/saves/model_fusion/Llama-3.2-3B-Instruct/0525/Llama-3.2-3B-Instruct-sft-warmup
CHECKPOINTS=("checkpoint-50" "checkpoint-100" "checkpoint-104")
OUTPUT_BASE="$PROJ_DIR/data/FuseChat-3.0-DPO-Data/0525/Llama-3.2-3B-Instruct/on_policy"

# # ----------------------- INFERENCE LOOP ------------------
# for MODEL_CKPT in "${CHECKPOINTS[@]}"; do
#     MODEL="$MODEL_BASE/$MODEL_CKPT"
#     OUTPUT_DIR="$OUTPUT_BASE/$MODEL_CKPT"
#     mkdir -p "$OUTPUT_DIR"

#     for SEED in "${SEEDS[@]}"; do
#         echo "Running inference for MODEL=$MODEL_CKPT SEED=$SEED"

#         for ((CHUNK_ID = 0; CHUNK_ID < CHUNK_NUM; CHUNK_ID++)); do
#             GPU_INDEX=$((CHUNK_ID % NUM_GPUS))
#             GPU=${GPUS[$GPU_INDEX]}

#             CUDA_VISIBLE_DEVICES=$GPU python "$PROJ_DIR/completion_inference/math_infer.py" \
#                 --input_file "$INPUT_FILE" \
#                 --model_name_or_path "$MODEL" \
#                 --output_dir "$OUTPUT_DIR" \
#                 --temperature 0.9 \
#                 --seed "$SEED" \
#                 --chunk_num "$CHUNK_NUM" \
#                 --chunk_id "$CHUNK_ID" &
#         done

#         wait
#     done

#     # ------------------- POST-PROCESS ---------------------
#     echo "Post-processing responses for $MODEL_CKPT"
#     python "$PROJ_DIR/completion_inference/build_dataset_v2.py" \
#         --response_file_dir "$OUTPUT_DIR" \
#         --post_process_type merge_multi_seeds_response_chunks

#     # ------------------- SCORING --------------------------
#     echo "Running RM scoring for $MODEL_CKPT"
#     activate_env llama_factory_sty

#     RM_SCORING_INPUT="$OUTPUT_DIR/all_generated_responses.json"
#     RM_SCORED_OUTPUT_DIR="$OUTPUT_DIR/completion_with_rm_score"
#     mkdir -p "$RM_SCORED_OUTPUT_DIR"

#     python "$PROJ_DIR/completion_inference/math_infer_verification.py" \
#         --input_file "$RM_SCORING_INPUT" \
#         --output_dir "$RM_SCORED_OUTPUT_DIR" \
#         --verfiy_mode Rule \
#         --chunk_num 1 \
#         --chunk_id 1
# done


# ----------------------- INFERENCE LOOP ------------------
OUTPUT_DIR="$OUTPUT_BASE"
mkdir -p "$OUTPUT_DIR"

# for SEED in "${SEEDS[@]}"; do
#     echo "Running inference for MODEL=$MODEL SEED=$SEED"

#     for ((CHUNK_ID = 0; CHUNK_ID < CHUNK_NUM; CHUNK_ID++)); do
#         GPU_INDEX=$((CHUNK_ID % NUM_GPUS))
#         GPU=${GPUS[$GPU_INDEX]}

#         CUDA_VISIBLE_DEVICES=$GPU python "$PROJ_DIR/completion_inference/math_infer.py" \
#             --input_file "$INPUT_FILE" \
#             --model_name_or_path "$MODEL" \
#             --output_dir "$OUTPUT_DIR" \
#             --temperature 0.9 \
#             --seed "$SEED" \
#             --chunk_num "$CHUNK_NUM" \
#             --chunk_id "$CHUNK_ID" &
#     done

#     wait
# done

# ------------------- POST-PROCESS ---------------------
echo "Post-processing responses for $MODEL"
# python "$PROJ_DIR/completion_inference/build_dataset_v2.py" \
#     --response_file_dir "$OUTPUT_DIR" \
#     --post_process_type merge_multi_seeds_response_chunks

# ------------------- SCORING --------------------------
echo "Running RM scoring for $MODEL"
activate_env llama_factory_sty

SCORING_INPUT="$OUTPUT_DIR/all_generated_responses.json"
RULE_SCORED_OUTPUT_DIR="$OUTPUT_DIR/completion_with_rm_score"
mkdir -p "$RULE_SCORED_OUTPUT_DIR"

# python "$PROJ_DIR/completion_inference/math_infer_verification.py" \
#     --input_file "$SCORING_INPUT" \
#     --output_dir "$RULE_SCORED_OUTPUT_DIR" \
#     --verfiy_mode Rule \
#     --chunk_num 1 \
#     --chunk_id 1


ORM_PATH="/GLOBALFS/gznwp_3/qxj/models/ArmoRM-Llama3-8B-v0.1"
CHUNK_NUM=$NUM_GPUS

RM_SCORING_INPUT="$RULE_SCORED_OUTPUT_DIR/all_generated_response_rule_verified.json"
RM_SCORED_OUTPUT_DIR=$RULE_SCORED_OUTPUT_DIR/orm_verify

# for ((i=0; i<NUM_GPUS; i++))
# do
#     # 计算当前任务分配的 GPU 索引
#     GPU_INDEX=$((i % NUM_GPUS))
#     GPU=${GPUS[$GPU_INDEX]}

#     # 使用 & 符号将任务放到后台运行
#     CUDA_VISIBLE_DEVICES=$GPU python $PROJ_DIR/completion_inference/response_scoring.py \
#         --input_file "$RM_SCORING_INPUT" \
#         --model_name_or_path "$ORM_PATH" \
#         --output_dir $RM_SCORED_OUTPUT_DIR \
#         --chunk_num $CHUNK_NUM \
#         --chunk_id $i &
# done
# wait

python "$PROJ_DIR/rollout_for_math/build_dataset_0520.py" \
        --completion_file_dir "$RM_SCORED_OUTPUT_DIR" \
        --post_process_type merge_completion_chunks