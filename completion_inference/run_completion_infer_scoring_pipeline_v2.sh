#### v2: 依次遍历source model response中切分位置，不考虑target model response质量
### RUN COMPLETION INFER
INPUT_FILE=/GLOBALFS/gznwp_3/qxj/shitianyuan/data/model_fusion_0226/source_responses_top_4/source_model_response_filtered_28806_over_8_steps.json
MODEL=/GLOBALFS/gznwp_3/qxj/yangzy/models/Llama-3.2-3B-Instruct

VERSION_NAME=data_select_based_on_adv_score
OUTPUT_DIR=/GLOBALFS/gznwp_3/qxj/shitianyuan/data/model_fusion/0226/$VERSION_NAME
DATASET_ID=completion_iter2 #chosen_sft_iter1 #completion_iter2

SEEDS=(13 21 42 79 100) # 21 42 79 100
GPUS_PER_NODE=2
COMPLETION_START_INDEX=1

NUM_SEEDS=${#SEEDS[@]}  # seed的个数
JOB_INDEX=0

while [ $JOB_INDEX -lt $NUM_SEEDS ]; do
    for (( GPU=0; GPU<$GPUS_PER_NODE; GPU++ )); do
        if [ $JOB_INDEX -ge $NUM_SEEDS ]; then
            break
        fi
        
        SEED=${SEEDS[$JOB_INDEX]}
        
        # 使用 & 符号将任务放到后台运行
        CUDA_VISIBLE_DEVICES=$GPU python ./completion_inference/completion_infer_v2.py \
            --input_file $INPUT_FILE \
            --model_name_or_path "$MODEL" \
            --model_id $DATASET_ID \
            --output_dir $OUTPUT_DIR \
            --completion_start_index $COMPLETION_START_INDEX \
            --seed $SEED &

        JOB_INDEX=$((JOB_INDEX + 1))
    done

    # 等待所有后台任务完成
    wait
done


### COMPLETION POST-PROCESS
POST_PROCESS_TYPE=merge_completion_chunks
COMPLETION_DIR=$OUTPUT_DIR/$DATASET_ID

python ./completion_inference/build_dataset_v2.py \
    --completion_file_dir $COMPLETION_DIR \
    --post_process_type $POST_PROCESS_TYPE


### COMPLETION RM SCORING
RM_SCORING_INPUT=$COMPLETION_DIR/all_generated_completions.json
REWARD_MODEL=/GLOBALFS/gznwp_3/qxj/chenrj/llm_model/ArmoRM-Llama3-8B-v0.1

RM_SCORED_OUTPUT_DIR=$COMPLETION_DIR/completion_with_rm_score

CHUNK_NUM=2
ACTUAL_SAMPLING_TIME=$(( CHUNK_NUM < GPUS_PER_NODE ? CHUNK_NUM : GPUS_PER_NODE ))

for ((i=0; i<ACTUAL_SAMPLING_TIME; i++))
do
    CUDA_VISIBLE_DEVICES=$i python ./completion_inference/completion_scoring_v2.py \
        --input_file $RM_SCORING_INPUT \
        --model_name_or_path $REWARD_MODEL \
        --output_dir $RM_SCORED_OUTPUT_DIR \
        --chunk_num $CHUNK_NUM \
        --chunk_id $i &
done

wait


### BUILD DPO DATASET
POST_PROCESS_TYPE=merge_completion_scoring_chunks
PRE_DATA_SAVE_DIR=data/model_fusion/zlg_1118/subset_sequential_ratio_0.4/$VERSION_NAME
# /nas-wulanchabu/shitianyuan.sty/LLaMA-Factory/data/model_fusion/zlg_1118/subset_sequential_ratio_0.4/$VERSION_NAME

python ./completion_inference/build_dataset_v2.py \
    --completion_file_dir $COMPLETION_DIR \
    --post_process_type $POST_PROCESS_TYPE \
    --preference_data_save_dir $PRE_DATA_SAVE_DIR \
    --completion_start_index $COMPLETION_START_INDEX


### BUILD CHOSEN SFT DATASET
POST_PROCESS_TYPE=build_sft_data_sharegpt_format

python ./completion_inference/build_dataset_v2.py \
    --completion_file_dir $COMPLETION_DIR \
    --post_process_type $POST_PROCESS_TYPE \
    --preference_data_save_dir $PRE_DATA_SAVE_DIR \
    --completion_start_index $COMPLETION_START_INDEX