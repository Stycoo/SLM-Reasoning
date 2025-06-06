### MATH EVALUATION
activate_env() {
    eval "$(conda shell.bash hook)"
    conda activate "$1"
}
activate_env 360_llama_fac

PROJ_DIR='/GLOBALFS/gznwp_3/qxj/shitianyuan/Model-Fusion-public'

cd $PROJ_DIR

set -x

export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export CUDA_VISIBLE_DEVICES=2,3

MODEL_ID="Llama-3.2-3B-Instruct"
DATA="0523"
VERSION="on_policy_dpo"

# Default values
MODEL_ROOT_DIR="$PROJ_DIR/saves/model_fusion/$MODEL_ID/$DATA/$VERSION"
OUTPUT_ROOT_DIR="$PROJ_DIR/benchmark_inference_results/$MODEL_ID/$DATA/$VERSION"
mkdir -p $OUTPUT_ROOT_DIR

LOG_DIR="${PROJ_DIR}/log/${DATA}/${VERSION}"
mkdir -p "$LOG_DIR"

# Sampling params
GPU_NUM=2
TP=1
TEMP=0.6
TOP_P=0.95
MAX_LEN=32768
 
MODEL_NAMES=(on_policy_avg_acc_0.4)
DATATYPES=(aime math)
N=(16 1)

for MODEL_NAME in "${MODEL_NAMES[@]}";do

MODEL_PATH=$MODEL_ROOT_DIR/$MODEL_NAME
echo "Model Path: ${MODEL_PATH}"

EVAL_LOG_DIR="${LOG_DIR}/${MODEL_NAME}"
mkdir -p $EVAL_LOG_DIR

# Loop through all datatypes
for ((i=0; i<${#DATATYPES[@]}; i++)); do
    DATA_TYPE=${DATATYPES[i]}
    N_VALUE=${N[i]}

    OUTPUT_DIR=$OUTPUT_ROOT_DIR/$MODEL_NAME/$DATA_TYPE
    echo "Output Directory: ${OUTPUT_DIR}"
    
    python -m verl.trainer.main_generation \
        trainer.nnodes=1 \
        trainer.n_gpus_per_node=${GPU_NUM} \
        data.path=$PROJ_DIR/deepscaler-release/processed_data/${DATA_TYPE}.parquet \
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
        +data.skip_format_reward=True > "${EVAL_LOG_DIR}/${DATA_TYPE}.log" 2>&1
done
done