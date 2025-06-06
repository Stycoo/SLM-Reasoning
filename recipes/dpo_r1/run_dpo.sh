#!/bin/bash

activate_env() {
    eval "$(conda shell.bash hook)"
    conda activate "$1"
}
activate_env llama_factory_sty

### DPO TRAINING
PROJ_DIR=/GLOBALFS/gznwp_3/qxj/shitianyuan/Model-Fusion-public
cd $PROJ_DIR

LOG_DIR=$PROJ_DIR/log/0424
mkdir -p $LOG_DIR
exec > $LOG_DIR/run_Light_R1_dpo.log 2>&1

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 FORCE_TORCHRUN=1 llamafactory-cli train $PROJ_DIR/recipes/dpo_r1/llama3_full_dpo_ds3.yaml