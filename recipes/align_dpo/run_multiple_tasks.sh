#!/bin/bash

set -e  # 出现错误时退出

PROJ_DIR=/GLOBALFS/gznwp_3/qxj/shitianyuan/Model-Fusion-public
cd $PROJ_DIR

# bash recipes/align_dpo/run_rollout_forward.sh
bash $PROJ_DIR/recipes/align_dpo/iterative_reverse_segment_dpo_w_accu_q_inherit_0520/run_iterative_reverse_segment_dpo_w_accu_q.sh
bash $PROJ_DIR/recipes/align_dpo/iterative_reverse_segment_dpo_w_accu_q_inherit_0520/run_iterative_reverse_segment_dpo_w_accu_q_cp.sh