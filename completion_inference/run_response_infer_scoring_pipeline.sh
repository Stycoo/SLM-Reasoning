#!/bin/sh
### RUN RESPONSE INFER
# source ~/.bashrc
# module unload CUDA/11.8
# module load CUDA/12.2
# conda activate vllm_zlg

INPUT_FILE=/data/sty/model_fusion/zlg_1118/source_responses_top_4/source_model_response_filtered_28806_over_8_steps.json
# /GLOBALFS/gznwp_3/qxj/shitianyuan/data/model_fusion_0226/source_responses_top_4/source_model_response_filtered_28806_over_8_steps.json
MODEL=/data/wanfq/fuse3/models/Llama-3.2-3B-Instruct
# /GLOBALFS/gznwp_3/qxj/yangzy/models/Llama-3.2-3B-Instruct

DATASET_ID=Llama-3.2-3B-Instruct-sampling-5-5k
OUTPUT_DIR=/data/sty/model_fusion/zlg_1118/source_responses_top_4/$DATASET_ID

PROJ_DIR=/home/sty/Model-Fusion
cd $PROJ_DIR

LOG_DIR=$PROJ_DIR/log/0304
mkdir -p $LOG_DIR
exec > $LOG_DIR/run_response_infer_scoring_pipeline.log 2>&1

# SEEDS=(13 21 42 79 100)
# GPUS_PER_NODE=1
# NUM_SEEDS=${#SEEDS[@]}  # seed的个数
# JOB_INDEX=0
# while [ $JOB_INDEX -lt $NUM_SEEDS ]; do
#     for (( GPU=0; GPU<$GPUS_PER_NODE; GPU++ )); do
#         if [ $JOB_INDEX -ge $NUM_SEEDS ]; then
#             break
#         fi
        
#         SEED=${SEEDS[$JOB_INDEX]}
        
#         # 使用 & 符号将任务放到后台运行
#         CUDA_VISIBLE_DEVICES=$GPU python $PROJ_DIR/completion_inference/response_infer.py \
#             --input_file $INPUT_FILE \
#             --model_name_or_path $MODEL \
#             --output_dir $OUTPUT_DIR \
#             --temperature 0.8 \
#             --top_p 0.95 \
#             --seed $SEED &

#         JOB_INDEX=$((JOB_INDEX + 1))
#     done

#     # 等待所有后台任务完成
#     wait
# done

SEEDS=(13 21 42 79 100)
GPUS=(0 1 2 3)
NUM_SEEDS=${#SEEDS[@]}  # seed 的个数
NUM_GPUS=${#GPUS[@]}    # 可用的 GPU 数量
CHUNK_NUM=$NUM_GPUS
JOB_INDEX=0

# while [ $JOB_INDEX -lt $NUM_SEEDS ]; do
#     SEED=${SEEDS[$JOB_INDEX]}
#     # 分配 CHUNK_NUM 块数据到 GPU
#     for (( CHUNK_ID=0; CHUNK_ID<CHUNK_NUM; CHUNK_ID++ )); do
#         GPU_INDEX=$((CHUNK_ID % NUM_GPUS))
#         GPU=${GPUS[$GPU_INDEX]}  # 获取当前 GPU
#         CUDA_VISIBLE_DEVICES=$GPU python $PROJ_DIR/completion_inference/response_infer.py \
#                 --input_file $INPUT_FILE \
#                 --model_name_or_path $MODEL \
#                 --output_dir $OUTPUT_DIR \
#                 --temperature 0.8 \
#                 --top_p 0.95 \
#                 --seed $SEED \
#                 --chunk_num $CHUNK_NUM \
#                 --chunk_id $CHUNK_ID &
#     done
    
#     # 等待所有 GPU 的任务完成
#     wait

#     JOB_INDEX=$((JOB_INDEX + 1))
# done


### RESPONSE POST-PROCESS
POST_PROCESS_TYPE=merge_multi_seeds_response_chunks #merge_completion_chunks 
RESPONSE_DIR=$OUTPUT_DIR

python $PROJ_DIR/completion_inference/build_dataset_v2.py \
    --response_file_dir $RESPONSE_DIR \
    --post_process_type $POST_PROCESS_TYPE


### RESPONSE RM SCORING
RM_SCORING_INPUT=$OUTPUT_DIR/all_generated_response.json
REWARD_MODEL=/data/chenrj/7b_model/ArmoRM-Llama3-8B-v0.1

# CHUNK_NUM=8
# ACTUAL_SAMPLING_TIME=$(( CHUNK_NUM < GPUS_PER_NODE ? CHUNK_NUM : GPUS_PER_NODE ))

# RM_SCORED_OUTPUT_DIR=$OUTPUT_DIR/multi_responses_with_rm_score

# for ((i=0; i<ACTUAL_SAMPLING_TIME; i++))
# do
#     CUDA_VISIBLE_DEVICES=$i python $PROJ_DIR/completion_inference/response_scoring.py \
#         --input_file "$RM_SCORING_INPUT" \
#         --model_name_or_path "$REWARD_MODEL" \
#         --output_dir $RM_SCORED_OUTPUT_DIR \
#         --chunk_num $CHUNK_NUM \
#         --chunk_id $i &
# done

# wait

CHUNK_NUM=$NUM_GPUS
RM_SCORED_OUTPUT_DIR=$OUTPUT_DIR/multi_responses_with_rm_score

for ((i=0; i<NUM_GPUS; i++))
do
    # 计算当前任务分配的 GPU 索引
    GPU_INDEX=$((i % NUM_GPUS))
    GPU=${GPUS[$GPU_INDEX]}

    # 使用 & 符号将任务放到后台运行
    CUDA_VISIBLE_DEVICES=$GPU python $PROJ_DIR/completion_inference/response_scoring.py \
        --input_file "$RM_SCORING_INPUT" \
        --model_name_or_path "$REWARD_MODEL" \
        --output_dir $RM_SCORED_OUTPUT_DIR \
        --chunk_num $CHUNK_NUM \
        --chunk_id $i &
done

# 等待所有后台任务完成
wait


### RESPONSE POST-PROCESS
POST_PROCESS_TYPE=merge_response_scoring_chunks

python $PROJ_DIR/completion_inference/build_dataset_v2.py \
    --response_file_dir $RM_SCORED_OUTPUT_DIR \
    --post_process_type $POST_PROCESS_TYPE