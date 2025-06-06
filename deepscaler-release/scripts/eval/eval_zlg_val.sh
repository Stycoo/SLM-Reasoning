set -x

export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export CUDA_VISIBLE_DEVICES=4,5,6,7


# Default values
MODEL_ROOT_DIR=/data/wanfq/fuse3/model_ckpt
# MODEL_ROOT_DIR=/data/wanfq/yangzy/model_ckpt
# MODEL_ROOT_DIR=/data/wanfq/yangzy/models
OUTPUT_ROOT_DIR=/data/wanfq/fuse3/Light-R1/deepscaler-release/eval_outputs

# Sampling params
GPU_NUM=4
TP=4
TEMP=0.6
TOP_P=0.95
# MAX_LEN=16384
# MAX_LEN=30000
MAX_LEN=32768
for MODEL_NAME in DeepSeek-R1-Distill-Qwen-7B-fuserl-sft-v3-bs16-ep5-lr1e-5-cutoff-16k Qwen2.5-7B-Instruct-fuserl-sft-v3-fuserl-pref-v3.3-dpo-beta0.1-lr3e-7-ep1-bs16-cutoff-16k Qwen2.5-7B-Instruct-fuserl-sft-v3-bs16-ep5-lr1e-5-cutoff-16k
do
# MODEL_NAME=DeepScaleR-1.5B-Preview
# MODEL_NAME=DeepSeek-R1-Distill-Qwen-1.5B

MODEL_PATH="$MODEL_ROOT_DIR/$MODEL_NAME"
# Possible values: aime, amc, math, minerva, olympiad_bench
# DATATYPES=("aime")
OUTPUT_DIR="$OUTPUT_ROOT_DIR/$MODEL_NAME"  # Add default output directory


DATATYPES=("aime25")
N=16

# Echo the values for verification
echo "Model Path: ${MODEL_PATH}"
echo "Datasets: ${DATATYPES[@]}"
echo "Output Directory: ${OUTPUT_DIR}"

# Loop through all datatypes
for DATA_TYPE in "${DATATYPES[@]}"; do
    python3 -m verl.trainer.main_generation \
        trainer.nnodes=1 \
        trainer.n_gpus_per_node=${GPU_NUM} \
        data.path=/data/wanfq/fuse3/Light-R1/deepscaler-release/processed_data/${DATA_TYPE}.parquet \
        data.output_path=${OUTPUT_DIR}/${DATA_TYPE}/${DATA_TYPE}_n_${N}_temp_${TEMP}_topp_${TOP_P}_maxlen_${MAX_LEN}.json \
        data.n_samples=${N} \
        data.batch_size=2048 \
        model.path=${MODEL_PATH} \
        rollout.temperature=${TEMP} \
        rollout.response_length=${MAX_LEN} \
        rollout.top_k=-1 \
        rollout.top_p=${TOP_P} \
        rollout.gpu_memory_utilization=0.85 \
        rollout.tensor_model_parallel_size=${TP} \
        +data.skip_format_reward=True
done
done
# +data.skip_format_reward=True是默认行为，跳过校验答案正确性时的格式检查，没<think>也没事
# nnodes增大，则可增大gpu_memory_utilization至0.9-0.95
# 如遇OOM，一般减小gpu_memory_utilization即可


MODEL_ROOT_DIR=/data/wanfq/yangzy/models
# Sampling params
GPU_NUM=4
TP=4
TEMP=0.6
TOP_P=0.95
# MAX_LEN=16384
# MAX_LEN=30000
MAX_LEN=32768
# DeepSeek-R1-Distill-Qwen-7B DeepSeek-R1-Distill-Qwen-1.5B  Qwen2.5-7B-Instruct Qwen2.5-7B-Instruct-s1k-qwq-v0-bs16-ep5-lr1e-5-cutoff-16k
for MODEL_NAME in Qwen2.5-7B-Instruct
do
# MODEL_NAME=DeepScaleR-1.5B-Preview
# MODEL_NAME=DeepSeek-R1-Distill-Qwen-1.5B

MODEL_PATH="$MODEL_ROOT_DIR/$MODEL_NAME"
# Possible values: aime, amc, math, minerva, olympiad_bench
# DATATYPES=("aime")
OUTPUT_DIR="$OUTPUT_ROOT_DIR/$MODEL_NAME"  # Add default output directory

DATATYPES=("aimea5")
N=16

# Echo the values for verification
echo "Model Path: ${MODEL_PATH}"
echo "Datasets: ${DATATYPES[@]}"
echo "Output Directory: ${OUTPUT_DIR}"

# Loop through all datatypes
for DATA_TYPE in "${DATATYPES[@]}"; do
    python3 -m verl.trainer.main_generation \
        trainer.nnodes=1 \
        trainer.n_gpus_per_node=${GPU_NUM} \
        data.path=/data/wanfq/fuse3/Light-R1/deepscaler-release/processed_data/${DATA_TYPE}.parquet \
        data.output_path=${OUTPUT_DIR}/${DATA_TYPE}/${DATA_TYPE}_n_${N}_temp_${TEMP}_topp_${TOP_P}_maxlen_${MAX_LEN}.json \
        data.n_samples=${N} \
        data.batch_size=2048 \
        model.path=${MODEL_PATH} \
        rollout.temperature=${TEMP} \
        rollout.response_length=${MAX_LEN} \
        rollout.top_k=-1 \
        rollout.top_p=${TOP_P} \
        rollout.gpu_memory_utilization=0.85 \
        rollout.tensor_model_parallel_size=${TP} \
        +data.skip_format_reward=True
done
done



# for MODEL_NAME in DeepSeek-R1-Distill-Qwen-1.5B-s1k-math-qwq-bs16-ep5-lr1e-5-cutoff-16k
# do
# # MODEL_NAME=DeepScaleR-1.5B-Preview
# # MODEL_NAME=DeepSeek-R1-Distill-Qwen-1.5B

# MODEL_PATH="$MODEL_ROOT_DIR/$MODEL_NAME"
# # Possible values: aime, amc, math, minerva, olympiad_bench
# # DATATYPES=("aime")
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
#         data.path=/data/wanfq/fuse3/Light-R1/deepscaler-release/processed_data/${DATA_TYPE}.parquet \
#         data.output_path=${OUTPUT_DIR}/${DATA_TYPE}/${DATA_TYPE}_n_${N}_temp_${TEMP}_topp_${TOP_P}_maxlen_${MAX_LEN}.json \
#         data.n_samples=${N} \
#         data.batch_size=2048 \
#         model.path=${MODEL_PATH} \
#         rollout.temperature=${TEMP} \
#         rollout.response_length=${MAX_LEN} \
#         rollout.top_k=-1 \
#         rollout.top_p=${TOP_P} \
#         rollout.gpu_memory_utilization=0.85 \
#         rollout.tensor_model_parallel_size=${TP} \
#         +data.skip_format_reward=True
# done
# done