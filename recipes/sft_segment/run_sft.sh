#!/bin/bash

activate_env() {
    eval "$(conda shell.bash hook)"
    conda activate "$1"
}
activate_env llama_factory_sty

PROJ_DIR=/GLOBALFS/gznwp_3/qxj/shitianyuan/Model-Fusion-public
cd $PROJ_DIR

LOG_DIR=$PROJ_DIR/log/0525
mkdir -p $LOG_DIR

# exec > $LOG_DIR/run_sft_segment_1.log 2>&1
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 FORCE_TORCHRUN=1 llamafactory-cli train $PROJ_DIR/recipes/sft_segment/llama3_full_sft_ds3_1.yaml

# exec > $LOG_DIR/run_sft_segment_2.log 2>&1
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 FORCE_TORCHRUN=1 llamafactory-cli train $PROJ_DIR/recipes/sft_segment/llama3_full_sft_ds3_2.yaml

exec > $LOG_DIR/run_sft_segment_3.log 2>&1
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 FORCE_TORCHRUN=1 llamafactory-cli train $PROJ_DIR/recipes/sft_segment/llama3_full_sft_ds3_3.yaml