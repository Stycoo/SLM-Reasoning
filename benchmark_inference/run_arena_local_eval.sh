export CUDA_VISIBLE_DEVICES=1

### benchmark infer
OUTPUT_DIR=/nas-wulanchabu/hongzhan.chz/tmp/fuse/eval/arena-hard/arena-hard/results/ModelFusion/model_answer
QUESTION_FILE=/nas-wulanchabu/hongzhan.chz/tmp/fuse/eval/arena-hard/arena-hard/data/arena-hard-v0.1/question.jsonl
REF_FILE=/nas-wulanchabu/hongzhan.chz/tmp/fuse/eval/arena-hard/arena-hard/data/arena-hard-v0.1/model_answer/gpt-4-0314.jsonl

VERSION=0103
MODEL_ID=stage2_dpo_completion_iter3_epoch3 #stage2_dpo_completion_iter5 stage2_dpo_source stage2_dpo_source_epoch3
MODEL=/nas-wulanchabu/shitianyuan.sty/LLaMA-Factory/saves/model_fusion/Llama-3.2-3B-Instruct/$VERSION/$MODEL_ID 

OUTPUT_DIR=$OUTPUT_DIR/$VERSION

python ./benchmark_inference/arena_hard_infer.py \
    --question_file "$QUESTION_FILE" \
    --model "$MODEL" \
    --model_id "$MODEL_ID" \
    --output_dir $OUTPUT_DIR/$MODEL_ID

### local eval
GENERATION_FILE=$OUTPUT_DIR/$MODEL_ID/$MODEL_ID.jsonl
REWARD_MODEL=/nas-wulanchabu/shitianyuan.sty/huggingface_model/reward_model/models--RLHFlow--ArmoRM-Llama3-8B-v0.1

python ./benchmark_inference/arena_hard_local_eval.py --reward_model $REWARD_MODEL --question_file $QUESTION_FILE --generation_file_1 $GENERATION_FILE --generation_file_2 $REF_FILE