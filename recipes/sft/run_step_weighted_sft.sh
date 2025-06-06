#!/bin/bash
# source /opt/conda/etc/profile.d/conda.sh
# conda activate /nas-wulanchabu/miniconda3/envs/tianyuan_llama_factory/

# module unload CUDA/11.8
# module load CUDA/12.2
source ~/.bashrc
conda activate llama_factory_sty

PROJ_DIR=/GLOBALFS/gznwp_3/qxj/shitianyuan/Model-Fusion
# /home/sty/Model-Fusion
# /GLOBALFS/gznwp_3/qxj/shitianyuan/Model-Fusion
cd $PROJ_DIR

export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"

LOG_DIR=$PROJ_DIR/log/0408
mkdir -p $LOG_DIR
exec > $LOG_DIR/run_openmathinstruct2_StepWeighted_CSFT_v3_3.log 2>&1

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 FORCE_TORCHRUN=1 llamafactory-cli train $PROJ_DIR/recipes/sft/llama3_full_step_weighted_sft_ds3.yaml