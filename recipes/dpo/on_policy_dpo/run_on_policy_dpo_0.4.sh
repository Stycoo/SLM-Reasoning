#!/bin/bash

activate_env() {
    eval "$(conda shell.bash hook)"
    conda activate "$1"
}
activate_env llama_factory_sty

### DPO TRAINING
PROJ_DIR=/GLOBALFS/gznwp_3/qxj/shitianyuan/Model-Fusion-public
cd $PROJ_DIR

DATA='0523'
VERSION='on_policy'

LOG_DIR="${PROJ_DIR}/log/${DATA}/${VERSION}"
mkdir -p "$LOG_DIR"

LOG_FILE="${LOG_DIR}/avg_acc_0.4.log"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 FORCE_TORCHRUN=1 llamafactory-cli train $PROJ_DIR/recipes/dpo/on_policy_dpo/llama3_full_dpo_ds3_0.4.yaml > "$LOG_FILE" 2>&1