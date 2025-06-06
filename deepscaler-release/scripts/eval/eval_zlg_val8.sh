#!/bin/bash

set -x

export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7


# Default values
MODEL_ROOT_DIR=/GLOBALFS/gznwp_3/qxj/lgzhong/LLaMA-Factory/saves/fuser1
OUTPUT_ROOT_DIR=/GLOBALFS/gznwp_3/qxj/lgzhong/deepscaler-release/eval_outputs

# Sampling params
GPU_NUM=8
TP=1
TEMP=0.6
TOP_P=0.95
MAX_LEN=32768
for MODEL_NAME in DeepSeek-R1-Distill-Qwen-1.5B-fuserl-pref-v3.3.3-dpo-beta0.3-lr5e-7-ep1-bs16-cutoff-16k
do
MODEL_PATH="$MODEL_ROOT_DIR/$MODEL_NAME"
OUTPUT_DIR="$OUTPUT_ROOT_DIR/$MODEL_NAME"  # Add default output directory
# DATA_TYPE="math"
# N=1

# # Echo the values for verification
# echo "Model Path: ${MODEL_PATH}"
# echo "Datasets: ${DATATYPES[@]}"
# echo "Output Directory: ${OUTPUT_DIR}"

# python3 -m verl.trainer.main_generation \
#     trainer.nnodes=1 \
#     trainer.n_gpus_per_node=${GPU_NUM} \
#     data.path=/GLOBALFS/gznwp_3/qxj/lgzhong/deepscaler-release/processed_data/${DATA_TYPE}.parquet \
#     data.output_path=${OUTPUT_DIR}/${DATA_TYPE}/${DATA_TYPE}_n_${N}_temp_${TEMP}_topp_${TOP_P}_maxlen_${MAX_LEN}.json \
#     data.n_samples=${N} \
#     data.batch_size=2048 \
#     model.path=${MODEL_PATH} \
#     rollout.temperature=${TEMP} \
#     rollout.response_length=${MAX_LEN} \
#     rollout.top_k=-1 \
#     rollout.top_p=${TOP_P} \
#     rollout.gpu_memory_utilization=0.95 \
#     rollout.tensor_model_parallel_size=${TP} \
#     +data.skip_format_reward=True


DATA_TYPE="aime"
N=16

# Echo the values for verification
echo "Model Path: ${MODEL_PATH}"
echo "Datasets: ${DATATYPES[@]}"
echo "Output Directory: ${OUTPUT_DIR}"

python3 -m verl.trainer.main_generation \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=${GPU_NUM} \
    data.path=/GLOBALFS/gznwp_3/qxj/lgzhong/deepscaler-release/processed_data/${DATA_TYPE}.parquet \
    data.output_path=${OUTPUT_DIR}/${DATA_TYPE}/${DATA_TYPE}_n_${N}_temp_${TEMP}_topp_${TOP_P}_maxlen_${MAX_LEN}.json \
    data.n_samples=${N} \
    data.batch_size=2048 \
    model.path=${MODEL_PATH} \
    rollout.temperature=${TEMP} \
    rollout.response_length=${MAX_LEN} \
    rollout.top_k=-1 \
    rollout.top_p=${TOP_P} \
    rollout.gpu_memory_utilization=0.95 \
    rollout.tensor_model_parallel_size=${TP} \
    +data.skip_format_reward=True

DATA_TYPE="aime25"
N=16

# Echo the values for verification
echo "Model Path: ${MODEL_PATH}"
echo "Datasets: ${DATATYPES[@]}"
echo "Output Directory: ${OUTPUT_DIR}"

python3 -m verl.trainer.main_generation \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=${GPU_NUM} \
    data.path=/GLOBALFS/gznwp_3/qxj/lgzhong/deepscaler-release/processed_data/${DATA_TYPE}.parquet \
    data.output_path=${OUTPUT_DIR}/${DATA_TYPE}/${DATA_TYPE}_n_${N}_temp_${TEMP}_topp_${TOP_P}_maxlen_${MAX_LEN}.json \
    data.n_samples=${N} \
    data.batch_size=2048 \
    model.path=${MODEL_PATH} \
    rollout.temperature=${TEMP} \
    rollout.response_length=${MAX_LEN} \
    rollout.top_k=-1 \
    rollout.top_p=${TOP_P} \
    rollout.gpu_memory_utilization=0.95 \
    rollout.tensor_model_parallel_size=${TP} \
    +data.skip_format_reward=True

done