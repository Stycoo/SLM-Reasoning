#!/bin/bash

set -e  # 出现错误时退出

PROJ_DIR=/GLOBALFS/gznwp_3/qxj/shitianyuan/Model-Fusion-public
cd $PROJ_DIR

bash recipes/dpo/on_policy_dpo/math_eval_0.2.sh &
bash recipes/dpo/on_policy_dpo/math_eval_0.4.sh &
bash recipes/dpo/on_policy_dpo/math_eval_0.6.sh 