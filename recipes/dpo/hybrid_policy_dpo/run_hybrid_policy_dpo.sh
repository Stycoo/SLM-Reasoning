#!/bin/bash

activate_env() {
    eval "$(conda shell.bash hook)"
    conda activate "$1"
}
activate_env llama_factory_sty

### DPO TRAINING
PROJ_DIR=/GLOBALFS/gznwp_3/qxj/shitianyuan/Model-Fusion-public
cd $PROJ_DIR

DATA='0430'
VERSION='hybrid_policy'

LOG_DIR="${PROJ_DIR}/log/${DATA}/${VERSION}"
mkdir -p "$LOG_DIR"

LOG_FILE="${LOG_DIR}/training_iter_1.log"

CUDA_VISIBLE_DEVICES=0,1,2,3 FORCE_TORCHRUN=1 llamafactory-cli train $PROJ_DIR/recipes/dpo/hybrid_policy_dpo/llama3_full_dpo_ds3.yaml #> "$LOG_FILE" 2>&1