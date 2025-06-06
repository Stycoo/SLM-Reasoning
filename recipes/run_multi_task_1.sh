#!/bin/bash

set -e  # 出现错误时退出

PROJ_DIR=/GLOBALFS/gznwp_3/qxj/shitianyuan/Model-Fusion-public
cd $PROJ_DIR

# bash recipes/dpo/iterative_misalign_dpo_reverse_seg_4/run_iterative_misalign_dpo_reverse_seg_4.sh &
# bash recipes/dpo/iterative_misalign_dpo_forward_seg_4/run_iterative_misalign_dpo_forward_seg_4.sh
# bash recipes/misalign_dpo/iterative_misalign_dpo_reverse_seg_4/run_iterative_misalign_dpo_reverse_seg_4.sh &
# bash recipes/misalign_dpo/iterative_misalign_dpo_forward_seg_4/run_iterative_misalign_dpo_forward_seg_4.sh
# bash recipes/dpo/on_policy_dpo/run_on_policy_dpo.sh


bash recipes/sft_segment/math_eval_1.sh &
bash recipes/sft_segment/math_eval_2.sh &
bash recipes/sft_segment/math_eval_3.sh