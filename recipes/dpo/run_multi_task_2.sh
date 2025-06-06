#!/bin/bash

set -e  # 出现错误时退出

PROJ_DIR=/GLOBALFS/gznwp_3/qxj/shitianyuan/Model-Fusion-public
cd $PROJ_DIR

bash recipes/dpo/iterative_misalign_dpo_v2_chosen_fixed_re/run_iterative_misalign_dpo_v2_chosen_fixed_re.sh
bash recipes/dpo/iterative_misalign_dpo_forward/run_iterative_misalign_dpo_forward.sh