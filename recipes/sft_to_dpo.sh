PROJ_BASE=/nas-wulanchabu/shitianyuan.sty/LLaMA-Factory
cd $PROJ_BASE

bash recipes/sft/run_sft.sh
bash recipes/dpo/run_dpo.sh