#!/bin/bash
set -e  # 出现错误时退出
set -u  # 使用未定义变量时报错

# 环境激活函数，确保每次循环时激活环境
activate_env() {
    eval "$(conda shell.bash hook)"
    conda activate "$1"
}

# 项目根目录
PROJ_DIR='/GLOBALFS/gznwp_3/qxj/shitianyuan/Model-Fusion-public'
cd "$PROJ_DIR" || { echo "Failed to cd to ${PROJ_DIR}"; exit 1; }

# 设置数据和版本
DATE='0501'
MODEL_ID="Llama-3.2-3B-Instruct"
VERSION='iterative_misalign_dpo_forward_seg_4'
ITERATION_TIME=4
FORWARD_OR_REVERSE="forward"

LOG_DIR="${PROJ_DIR}/log/${DATE}/${VERSION}"
mkdir -p "$LOG_DIR"

for (( i=1; i<=ITERATION_TIME; i++ ))
do  
    COMPLETION_INFER_LOG_DIR="${LOG_DIR}/completion_infer_iter_${i}"
    mkdir -p "$COMPLETION_INFER_LOG_DIR"

    # 激活环境（每次迭代激活）
    activate_env vllm_zlg

    echo "Starting Iteration $i at $(date)"

    # 输入文件和模型设置
    INPUT_FILE="$PROJ_DIR/data/FuseChat-3.0-DPO-Data/openmathinstruct2_validation.json"
    OUTPUT_DIR="$PROJ_DIR/data/FuseChat-3.0-DPO-Data/$DATE/$MODEL_ID/$VERSION/iter_$i"
    ITER_SEG_INDEX_FILE="$PROJ_DIR/data/FuseChat-3.0-DPO-Data/$DATE/$MODEL_ID/$VERSION/segment_index.json"

    if [ $i -eq 1 ]; then
        MODEL=/GLOBALFS/gznwp_3/qxj/models/$MODEL_ID
    else
        MODEL="$PROJ_DIR/saves/model_fusion/$MODEL_ID/$DATE/$VERSION/iter_$((i-1))"
    fi

    SEEDS=(100 13 21 42 79) # 13 21 42 79
    GPUS=(4 5 6 7) # 1 2 3 4 5 6 7
    NUM_SEEDS=${#SEEDS[@]}  # seed 的个数
    NUM_GPUS=${#GPUS[@]}    # 可用的 GPU 数量
    CHUNK_NUM=$NUM_GPUS
    JOB_INDEX=0

    while [ $JOB_INDEX -lt $NUM_SEEDS ]; do
        SEED=${SEEDS[$JOB_INDEX]}
        
        # 分配 CHUNK_NUM 块数据到 GPU
        for (( CHUNK_ID=0; CHUNK_ID<CHUNK_NUM; CHUNK_ID++ )); do
            TASK_LOG_FILE="${COMPLETION_INFER_LOG_DIR}/run_infer_chunk_${CHUNK_ID}_seed_${SEED}.log"

            GPU_INDEX=$((CHUNK_ID % NUM_GPUS))
            GPU=${GPUS[$GPU_INDEX]}  # 获取当前 GPU
            CUDA_VISIBLE_DEVICES=$GPU python $PROJ_DIR/rollout_for_math/rollout_in_last_and_current_iter_seg_index.py \
                    --forward_or_reverse $FORWARD_OR_REVERSE \
                    --input_file $INPUT_FILE \
                    --last_iteration_segment_index_file $ITER_SEG_INDEX_FILE \
                    --model_name_or_path $MODEL \
                    --model_id $MODEL_ID \
                    --output_dir $OUTPUT_DIR \
                    --seed $SEED \
                    --chunk_num $CHUNK_NUM \
                    --chunk_id $CHUNK_ID \
                    --iteration_time $ITERATION_TIME \
                    --completion_id $i > "$TASK_LOG_FILE" 2>&1 &
        done

        # 等待所有 GPU 的任务完成
        wait

        JOB_INDEX=$((JOB_INDEX + 1))
    done

    ### COMPLETION POST-PROCESS
    POST_PROCESS_TYPE=merge_multi_seeds_completion_chunks_v2
    POST_PROCESS_LOG_FILE="${LOG_DIR}/post_process_iter_${i}.log"
    python $PROJ_DIR/rollout_for_math/build_dataset.py \
        --completion_file_dir $OUTPUT_DIR \
        --post_process_type $POST_PROCESS_TYPE > "$POST_PROCESS_LOG_FILE" 2>&1

    ### COMPLETION RM SCORING
    # 激活新的环境
    activate_env llama_factory_sty

    RM_SCORING_INPUT=$OUTPUT_DIR/all_generated_completions.json
    REWARD_MODEL=/GLOBALFS/gznwp_3/qxj/models/Qwen2.5-Math-PRM-7B
    RM_SCORED_OUTPUT_DIR=$OUTPUT_DIR/completion_with_rm_score

    VERIFY_MODE=Rule # PRM or Rule or LLM-as-a-judge

    if [ "$VERIFY_MODE" = "Rule" ]; then
        CHUNK_NUM=1
    elif [ "$VERIFY_MODE" = "PRM" ]; then
        CHUNK_NUM=$NUM_GPUS
    else
        echo "Invalid VERIFY_MODE: $VERIFY_MODE"
        exit 1
    fi

    for ((j=0; j<CHUNK_NUM; j++))
    do
        # 计算当前任务分配的 GPU 索引
        GPU_INDEX=$((j % NUM_GPUS))
        GPU=${GPUS[$GPU_INDEX]}
        SCORING_LOG_FILE="${LOG_DIR}/scoring_iter_${i}_chunk_${j}.log"
        # 使用 & 符号将任务放到后台运行
        CUDA_VISIBLE_DEVICES=$GPU python $PROJ_DIR/rollout_for_math/rollout_verification.py \
            --input_file "$RM_SCORING_INPUT" \
            --model_name_or_path "$REWARD_MODEL" \
            --output_dir $RM_SCORED_OUTPUT_DIR \
            --verfiy_mode $VERIFY_MODE \
            --chunk_num $CHUNK_NUM \
            --chunk_id $j > "$SCORING_LOG_FILE" 2>&1 &
    done
    wait

    ### CREATE MISALINGED DPO DATA
    POST_PROCESS_TYPE=create_misaligned_segment_dpo_dataset
    COMPLETION_DIR=$RM_SCORED_OUTPUT_DIR
    CREATE_DPO_LOG_FILE="${LOG_DIR}/create_misaligned_dpo_iter_${i}.log"
    python $PROJ_DIR/rollout_for_math/build_dataset.py \
        --forward_or_reverse $FORWARD_OR_REVERSE \
        --completion_file_dir $RM_SCORED_OUTPUT_DIR \
        --last_iteration_segment_index_file $ITER_SEG_INDEX_FILE \
        --post_process_type $POST_PROCESS_TYPE \
        --iteration_time $ITERATION_TIME \
        --iteration_id $i > "$CREATE_DPO_LOG_FILE" 2>&1
        
    ### MISALIGNED DPO TRAINING
    TRAINING_LOG_DIR="${LOG_DIR}/training_iter_${i}"
    mkdir -p "$TRAINING_LOG_DIR"
    LOG_FILE="${TRAINING_LOG_DIR}/run_dpo_iter_${i}.log"

    echo "-------------------------------"
    echo "Iteration ${i}:"
    echo "  Log file: ${LOG_FILE}"
    echo "-------------------------------"

    CUDA_VISIBLE_DEVICES=4,5,6,7 FORCE_TORCHRUN=1 llamafactory-cli train \
        "$PROJ_DIR/recipes/misalign_dpo/iterative_misalign_dpo_${FORWARD_OR_REVERSE}_seg_4/iter_$i.yaml" > "$LOG_FILE" 2>&1

    echo "Iteration ${i} completed. Log saved to ${LOG_FILE}"
    # exit 0
done

exit 0
### MATH EVALUATION
activate_env 360_llama_fac

PROJ_DIR="$PROJ_DIR"
cd $PROJ_DIR

set -x

export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export CUDA_VISIBLE_DEVICES=6,7

# Default values
MODEL_ROOT_DIR="$PROJ_DIR/saves/model_fusion/$MODEL_ID/$DATE/$VERSION"
OUTPUT_ROOT_DIR="$PROJ_DIR/benchmark_inference_results/$MODEL_ID/$DATE/$VERSION"

mkdir -p $OUTPUT_ROOT_DIR

# Sampling params
GPU_NUM=2
TP=1
TEMP=0.6
TOP_P=0.95
MAX_LEN=32768

MODEL_NAMES=(iter_4) # iter_2 iter_3
DATATYPES=(aime)
N=(16)

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