PROJ_DIR=
# /home/sty/Model-Fusion-public

activate_env() {
    eval "$(conda shell.bash hook)"
    conda activate "$1"
}

### vllm envs
conda env create -f $PROJ_DIR/vllm_environment.yml -y

### llama_fac envs
conda env create -f $PROJ_DIR/llama_fac_environment.yml -y
activate_env llama_factory_sty
cd "$PROJ_DIR"
pip install -e .

### math eval envs
conda env create -f $PROJ_DIR/360_llama_fac_environment.yml -y
activate_env 360_llama_fac

cd "$PROJ_DIR/360-LLaMA-Factory"
pip install -e .

cd "$PROJ_DIR/deepscaler-release"
pip install -e .

cd "$PROJ_DIR/deepscaler-release/verl"
pip install -e .