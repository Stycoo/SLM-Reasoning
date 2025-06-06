#!/bin/bash

# 指定Miniconda的下载链接，这将随版本更新而变。
# CONDA_INSTALLER_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
CONDA_INSTALLER_URL="https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-latest-Linux-x86_64.sh"
INSTALLER_NAME="Miniconda3-latest-Linux-x86_64.sh"
INSTALL_PATH="$HOME/miniconda3"

# 下载Miniconda安装脚本
wget "$CONDA_INSTALLER_URL" -O "$INSTALLER_NAME"

# 给安装脚本执行权限
chmod +x "$INSTALLER_NAME"

# 使用非交互模式运行安装脚本并且指定安装路径
./"$INSTALLER_NAME" -b -p "$INSTALL_PATH"

# 初始化conda环境
eval "$INSTALL_PATH"/bin/conda init

# 更新.bashrc，使得conda命令可用
source "$HOME/.bashrc"

# 更新conda到最新版
"$INSTALL_PATH"/bin/conda update -y conda

# 清理安装包
rm -f "$INSTALLER_NAME"
