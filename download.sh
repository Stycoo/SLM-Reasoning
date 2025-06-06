#!/bin/bash
export HF_ENDPOINT=https://hf-mirror.com

PROJ_DIR=
# /home/sty/Model-Fusion-public
cd $PROJ_DIR

huggingface-cli download --resume-download FuseAI/FuseChat-Llama-3.2-3B-SFT \
--token "" \
--local-dir $PROJ_DIR/base_models/FuseChat-Llama-3.2-3B-SFT \
--exclude "*.pth" "*.pt" "consolidated*"  \
--cache-dir $PROJ_DIR/base_models/hugging_cache \
--local-dir-use-symlinks False
