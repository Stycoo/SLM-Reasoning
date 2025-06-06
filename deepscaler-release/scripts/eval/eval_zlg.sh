set -x

export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Default values
MODEL_ROOT_DIR=/GLOBALFS/gznwp_3/qxj/lgzhong/LLaMA-Factory/saves/fuser1
OUTPUT_ROOT_DIR=/GLOBALFS/gznwp_3/qxj/lgzhong/deepscaler-release/eval_outputs

# Sampling params
GPU_NUM=8
N=16
TP=1
TEMP=0.6
TOP_P=0.95
# MAX_LEN=16384
MAX_LEN=32768
# DeepSeek-R1-Distill-Qwen-7B DeepSeek-R1-Distill-Qwen-1.5B  Qwen2.5-7B-Instruct 
for MODEL_NAME in DeepSeek-R1-Distill-Qwen-7B-s1k-math-qwq-bs16-ep5-lr1e-5-cutoff-16k 
do
# MODEL_NAME=DeepScaleR-1.5B-Preview
# MODEL_NAME=DeepSeek-R1-Distill-Qwen-1.5B

MODEL_PATH="$MODEL_ROOT_DIR/$MODEL_NAME"
# Possible values: aime, amc, math, minerva, olympiad_bench
# DATATYPES=("aime")
DATATYPES=("math" "aime" "amc")
OUTPUT_DIR="$OUTPUT_ROOT_DIR/$MODEL_NAME"  # Add default output directory

# Parse named arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_PATH="$2"
            shift 2
            ;;
        --datasets)
            # Convert space-separated arguments into array
            shift
            DATATYPES=()
            while [[ $# -gt 0 && ! $1 =~ ^-- ]]; do
                DATATYPES+=("$1")
                shift
            done
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 --model <model_path> --datasets dataset1 dataset2 ... --output-dir <output_directory>"
            exit 1
            ;;
    esac
done

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