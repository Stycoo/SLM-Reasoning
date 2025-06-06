#!/bin/sh
### RUN COMPLETION INFER
# source ~/.bashrc
module unload CUDA/11.8
module load CUDA/12.2
conda activate vllm_zlg

PROJ_DIR=/home/sty/Model-Fusion
# /GLOBALFS/gznwp_3/qxj/shitianyuan/Model-Fusion
cd $PROJ_DIR

LOG_DIR=$PROJ_DIR/log/0304
mkdir -p $LOG_DIR
exec > $LOG_DIR/run_completion_infer_scoring_pipeline_v6.log 2>&1

#### v2: 依次遍历source model response中切分位置，不考虑target model response质量
### RUN COMPLETION INFER
INPUT_FILE=/data/sty/model_fusion/zlg_1118/source_responses_top_4/source_model_response_filtered_28806_over_8_steps.json
# /GLOBALFS/gznwp_3/qxj/shitianyuan/data/model_fusion_0226/source_responses_top_4/source_model_response_filtered_28806_over_8_steps.json
MODEL=/data/wanfq/fuse3/models/Llama-3.2-3B-Instruct

VERSION_NAME=data_select_based_on_adv_score
OUTPUT_DIR=/data/sty/model_fusion/zlg_1118/source_responses_top_4/$VERSION_NAME
MAX_STEPS=16
DATASET_ID=max_step_$MAX_STEPS #chosen_sft_iter1 #completion_iter2

# SEEDS=(13 21 42 79 100)
# GPUS_PER_NODE=8

# NUM_SEEDS=${#SEEDS[@]}  # seed的个数
# JOB_INDEX=0
# CHUNK_NUM=8

# while [ $JOB_INDEX -lt $NUM_SEEDS ]; do
#     SEED=${SEEDS[$JOB_INDEX]}
    
#     # 分配 CHUNK_NUM 块数据到 GPU
#     for (( CHUNK_ID=0; CHUNK_ID<CHUNK_NUM; CHUNK_ID++ )); do
#         GPU=$(( CHUNK_ID % GPUS_PER_NODE ))  # 循环分配 GPU
#         CUDA_VISIBLE_DEVICES=$GPU python $PROJ_DIR/completion_inference/completion_infer_v6.py \
#                 --input_file $INPUT_FILE \
#                 --model_name_or_path "$MODEL" \
#                 --model_id $DATASET_ID \
#                 --output_dir $OUTPUT_DIR \
#                 --seed $SEED \
#                 --max_steps $MAX_STEPS \
#                 --chunk_num $CHUNK_NUM \
#                 --chunk_id $CHUNK_ID &
#     done
    
#     # 等待所有 GPU 的任务完成
#     wait
    
#     JOB_INDEX=$((JOB_INDEX + 1))
# done


SEEDS=(100) # 13 21 42 79 
GPUS=(4 5 6 7)
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
#         CUDA_VISIBLE_DEVICES=$GPU python $PROJ_DIR/completion_inference/completion_infer_v6.py \
#                 --input_file $INPUT_FILE \
#                 --model_name_or_path $MODEL \
#                 --model_id $DATASET_ID \
#                 --output_dir $OUTPUT_DIR \
#                 --seed $SEED \
#                 --max_steps $MAX_STEPS \
#                 --chunk_num $CHUNK_NUM \
#                 --chunk_id $CHUNK_ID &
#     done
    
#     # 等待所有 GPU 的任务完成
#     wait
    
#     JOB_INDEX=$((JOB_INDEX + 1))
# done


### COMPLETION POST-PROCESS
POST_PROCESS_TYPE=merge_multi_seeds_completion_chunks_v2 #merge_completion_chunks
COMPLETION_DIR=$OUTPUT_DIR/$DATASET_ID

# python $PROJ_DIR/completion_inference/build_dataset_v2.py \
#     --completion_file_dir $COMPLETION_DIR \
#     --post_process_type $POST_PROCESS_TYPE


### COMPLETION RM SCORING
RM_SCORING_INPUT=$COMPLETION_DIR/all_generated_completions.json
REWARD_MODEL=/data/chenrj/7b_model/ArmoRM-Llama3-8B-v0.1
# /GLOBALFS/gznwp_3/qxj/models/ArmoRM-Llama3-8B-v0.1

RM_SCORED_OUTPUT_DIR=$COMPLETION_DIR/completion_with_rm_score

# CHUNK_NUM=8
# ACTUAL_LOOPING_TIME=$(( CHUNK_NUM < GPUS_PER_NODE ? CHUNK_NUM : GPUS_PER_NODE ))

# for ((i=0; i<ACTUAL_LOOPING_TIME; i++))
# do
#     CUDA_VISIBLE_DEVICES=$i python $PROJ_DIR/completion_inference/completion_scoring_v2.py \
#         --input_file $RM_SCORING_INPUT \
#         --model_name_or_path $REWARD_MODEL \
#         --output_dir $RM_SCORED_OUTPUT_DIR \
#         --chunk_num $CHUNK_NUM \
#         --chunk_id $i &
# done

# wait

CHUNK_NUM=$NUM_GPUS

# for ((i=0; i<NUM_GPUS; i++))
# do
#     # 计算当前任务分配的 GPU 索引
#     GPU_INDEX=$((i % NUM_GPUS))
#     GPU=${GPUS[$GPU_INDEX]}

#     # 使用 & 符号将任务放到后台运行
#     CUDA_VISIBLE_DEVICES=$GPU python $PROJ_DIR/completion_inference/completion_scoring_v2.py \
#         --input_file "$RM_SCORING_INPUT" \
#         --model_name_or_path "$REWARD_MODEL" \
#         --output_dir $RM_SCORED_OUTPUT_DIR \
#         --chunk_num $CHUNK_NUM \
#         --chunk_id $i &
# done
# wait


### MERGE COMPLETION SCORING CHUNKS
POST_PROCESS_TYPE=merge_completion_scoring_chunks
COMPLETION_DIR=$RM_SCORED_OUTPUT_DIR

# python $PROJ_DIR/completion_inference/build_dataset_v2.py \
#     --completion_file_dir $COMPLETION_DIR \
#     --post_process_type $POST_PROCESS_TYPE


### DATA SELECT BASED ON ADV SCORE
POST_PROCESS_TYPE=data_select_based_on_adv_score
TRAIN_DATA_SAVE_DIR=data/model_fusion/$VERSION_NAME/0304/$DATASET_ID
SOURCE_MODEL_RESPONSE_FILE=$INPUT_FILE
TARGET_MODEL_RESPINSER_FILE=/data/sty/model_fusion/zlg_1118/source_responses_top_4/Llama-3.2-3B-Instruct-sampling-5/multi_responses_with_rm_score/multi_responses_with_rm_score.json
# /data/sty/model_fusion/zlg_1118/source_responses_top_4/Llama-3.2-3B-Instruct-sampling-5/multi_responses_with_rm_score/multi_responses_with_rm_score.json

python $PROJ_DIR/completion_inference/build_dataset_v2.py \
    --completion_file_dir $COMPLETION_DIR \
    --post_process_type $POST_PROCESS_TYPE \
    --source_model_response_file $SOURCE_MODEL_RESPONSE_FILE \
    --target_model_response_file $TARGET_MODEL_RESPINSER_FILE \
    --train_data_save_dir $TRAIN_DATA_SAVE_DIR
