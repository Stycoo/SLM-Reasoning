#!/bin/bash

source ~/.bashrc
conda activate 360_llama_fac

PROJ_DIR=/GLOBALFS/gznwp_3/qxj/lgzhong/deepscaler-release-qxj
cd $PROJ_DIR

set -x

export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export CUDA_VISIBLE_DEVICES=0,1

# Default values
MODEL_ROOT_DIR=/GLOBALFS/gznwp_3/qxj/shitianyuan/Model-Fusion/saves/model_fusion/FuseChat-Llama-3.2-3B-SFT/0412/iterative_misalign_dpo_v2 # iteration_order_ablation_2
# /GLOBALFS/gznwp_3/qxj/models/FuseChat-Llama-3.2-3B-SFT
# /GLOBALFS/gznwp_3/qxj/shitianyuan/Model-Fusion/saves/model_fusion/FuseChat-Llama-3.2-3B-SFT/0405
# /GLOBALFS/gznwp_3/qxj/models/Llama-3.2-3B-Instruct
# /GLOBALFS/gznwp_3/qxj/shitianyuan/Model-Fusion/saves/model_fusion/Llama-3.2-3B-Instruct/0322/openr1_solution_hard_sft
# /GLOBALFS/gznwp_3/qxj/models/DeepSeek-R1-Distill-Qwen-1.5B
# /GLOBALFS/gznwp_3/qxj/shitianyuan/Model-Fusion/saves/model_fusion/DeepSeek-R1-Distill-Qwen-1.5B/0319
# /GLOBALFS/gznwp_3/qxj/models/Qwen2.5-3B-Instruct
# /GLOBALFS/gznwp_3/qxj/shitianyuan/Model-Fusion/saves/Qwen2.5-3B-Instruct/0316
OUTPUT_ROOT_DIR=/GLOBALFS/gznwp_3/qxj/shitianyuan/Model-Fusion/benchmark_inference_results/FuseChat-Llama-3.2-3B-SFT/0412/iterative_misalign_dpo_v2
# 0322/openr1_solution_hard_sft
# /GLOBALFS/gznwp_3/qxj/shitianyuan/Model-Fusion/benchmark_inference_results/Qwen2.5-3B-Instruct/base
# /GLOBALFS/gznwp_3/qxj/shitianyuan/Model-Fusion/benchmark_inference_results/Qwen2.5-3B-Instruct/0316
mkdir -p $OUTPUT_ROOT_DIR

# Sampling params
GPU_NUM=2
TP=1
TEMP=0.6
TOP_P=0.95
MAX_LEN=32768

MODEL_NAMES=(iter_1 iter_2 iter_3)
DATATYPES=(aime math) # "math" aime 
N=(16 1) #(1 16)

for MODEL_NAME in "${MODEL_NAMES[@]}";do

MODEL_PATH=$MODEL_ROOT_DIR/$MODEL_NAME
# MODEL_PATH=$MODEL_ROOT_DIR
echo "Model Path: ${MODEL_PATH}"

# Loop through all datatypes
for ((i=0; i<${#DATATYPES[@]}; i++)); do
    DATA_TYPE=${DATATYPES[i]}
    N_VALUE=${N[i]}

    OUTPUT_DIR=$OUTPUT_ROOT_DIR/$MODEL_NAME/$DATA_TYPE
    # OUTPUT_DIR=$OUTPUT_ROOT_DIR/$DATA_TYPE
    echo "Output Directory: ${OUTPUT_DIR}"

    python -m verl.trainer.main_generation \
        trainer.nnodes=1 \
        trainer.n_gpus_per_node=${GPU_NUM} \
        data.path=/GLOBALFS/gznwp_3/qxj/lgzhong/deepscaler-release-qxj/processed_data/${DATA_TYPE}.parquet \
        data.output_path=${OUTPUT_DIR}/n_${N}_temp_${TEMP}_topp_${TOP_P}_maxlen_${MAX_LEN}.json \
        data.n_samples=${N_VALUE} \
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
done


# ------------------------------------------------------

# for MODEL_NAME in DeepSeek-R1-Distill-Qwen-7B-s1k-math-qwq-bs16-ep5-lr1e-5-cutoff-16k
# do
# MODEL_PATH="$MODEL_ROOT_DIR/$MODEL_NAME"
# OUTPUT_DIR="$OUTPUT_ROOT_DIR/$MODEL_NAME"  # Add default output directory
# DATATYPES=("aime")
# N=16

# # Echo the values for verification
# echo "Model Path: ${MODEL_PATH}"
# echo "Datasets: ${DATATYPES[@]}"
# echo "Output Directory: ${OUTPUT_DIR}"

# # Loop through all datatypes
# for DATA_TYPE in "${DATATYPES[@]}"; do
#     python3 -m verl.trainer.main_generation \
#         trainer.nnodes=1 \
#         trainer.n_gpus_per_node=${GPU_NUM} \
#         data.path=/GLOBALFS/gznwp_3/qxj/lgzhong/deepscaler-release/processed_data/${DATA_TYPE}.parquet \
#         data.output_path=${OUTPUT_DIR}/${DATA_TYPE}/${DATA_TYPE}_n_${N}_temp_${TEMP}_topp_${TOP_P}_maxlen_${MAX_LEN}.json \
#         data.n_samples=${N} \
#         data.batch_size=2048 \
#         model.path=${MODEL_PATH} \
#         rollout.temperature=${TEMP} \
#         rollout.response_length=${MAX_LEN} \
#         rollout.top_k=-1 \
#         rollout.top_p=${TOP_P} \
#         rollout.gpu_memory_utilization=0.95 \
#         rollout.tensor_model_parallel_size=${TP} \
#         +data.skip_format_reward=True
# done
# done