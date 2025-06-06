LOG_DIR=/nas-wulanchabu/shitianyuan.sty/LLaMA-Factory/log

# nvidia-smi > $LOG_DIR/gpu_status.log
ps aux|grep python > $LOG_DIR/gpu_status.log