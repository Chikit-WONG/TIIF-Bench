#!/bin/bash

module load cuda/12.1

source /hpc2hdd/home/ckwong627/miniconda3/etc/profile.d/conda.sh
conda activate TIIF-Bench

echo "🔍 正在检测 cuDNN 加载环境..."
echo

# Step 1: 确认是否在 Conda 环境中
if [ -z "$CONDA_PREFIX" ]; then
    echo "❌ 当前不在 Conda 环境中，请先 conda activate <env>"
    exit 1
fi

echo "✅ 当前 Conda 环境: $CONDA_PREFIX"

# Step 2: 检查 cuDNN .so 文件是否存在于 Conda 环境
echo "🔍 正在查找 cuDNN 库文件..."
CUDNN_FILES=$(find "$CONDA_PREFIX/lib" -name "libcudnn*.so*" 2>/dev/null)

if [ -z "$CUDNN_FILES" ]; then
    echo "❌ 未在 Conda 环境中找到 cuDNN 库文件！"
    echo "你可能需要运行: conda install -c nvidia cudnn=9.1"
    exit 1
fi

echo "✅ 找到以下 cuDNN 文件:"
echo "$CUDNN_FILES"
echo

# Step 3: 设置 LD_LIBRARY_PATH（追加，不覆盖）
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
echo "✅ 已设置 LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
echo

# Step 4: 验证 Python 是否已链接到 Conda 的 cuDNN
echo "🔍 检查当前 Python 动态库链接:"
ldd $(which python) | grep cudnn || echo "⚠️ 当前 Python 进程未链接 libcudnn（这是正常现象，只有使用时才加载）"

echo
echo "✅ cuDNN 检查与路径设置完成。你现在可以重新运行 vLLM 了。"
