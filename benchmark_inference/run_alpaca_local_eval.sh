export CUDA_VISIBLE_DEVICES=5
# conda activate /nas-wulanchabu/miniconda3/envs/tianyuan_handbook_update

### benchmark infer
# OUTPUT_DIR=/nas-wulanchabu/shitianyuan.sty/alpaca_eval/results/ModelFusion
# REF_FILE=/nas-wulanchabu/shitianyuan.sty/alpaca_eval/results/Mutual-Taught/reference_outputs.json

# VERSION=0217
# MODEL_ID=chosen_sft_iteration_3 # dpo_completion_iteration_1
# MODEL=/nas-wulanchabu/shitianyuan.sty/LLaMA-Factory/saves/model_fusion/Llama-3.2-3B-Instruct/$VERSION/$MODEL_ID 

# OUTPUT_DIR=$OUTPUT_DIR/$VERSION

PROJ_DIR=/home/sty/Model-Fusion
cd $PROJ_DIR

VERSION=0304
MODEL_ID=sft_max_q_5000 #sft_adv_best_5000

REF_FILE=data/alpaca_eval_reference_outputs.json
MODEL=/data/sty/model_fusion/save_models/Llama-3.2-3B-Instruct/0304/$MODEL_ID
OUTPUT_DIR=benchmark_inference_results/alpaca_eval/$VERSION/$MODEL_ID
mkdir -p $OUTPUT_DIR

python $PROJ_DIR/benchmark_inference/alpaca_eval_infer.py \
    --question_file $REF_FILE \
    --model $MODEL \
    --model_id $MODEL_ID \
    --output_dir $OUTPUT_DIR
### local eval
GENERATION_FILE=$OUTPUT_DIR/model_outputs.json

REWARD_MODEL=/data/chenrj/7b_model/ArmoRM-Llama3-8B-v0.1
# /nas-wulanchabu/shitianyuan.sty/huggingface_model/reward_model/models--RLHFlow--ArmoRM-Llama3-8B-v0.1

python $PROJ_DIR/benchmark_inference/alpaca_local_eval.py --reward_model $REWARD_MODEL --generation_file_1 $GENERATION_FILE --generation_file_2 $REF_FILE