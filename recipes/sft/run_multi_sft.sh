source /opt/conda/etc/profile.d/conda.sh
conda activate /nas-wulanchabu/miniconda3/envs/tianyuan_llama_factory/

PROJ_BASE=/nas-wulanchabu/shitianyuan.sty/LLaMA-Factory
cd $PROJ_BASE

LOG_DIR=$PROJ_BASE/log/1124
if [ ! -d "$LOG_DIR" ]; then
    mkdir -p "$LOG_DIR"
    echo "Directory created: $LOG_DIR"
else
    echo "Directory already exists: $LOG_DIR"
fi

FORCE_TORCHRUN=1 llamafactory-cli train recipes/sft/llama3_full_multi_sft_ds3.yaml > $LOG_DIR/multi_sft_wo_overlap_v2_2.log