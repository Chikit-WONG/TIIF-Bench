#!/bin/bash

# 设置你想迁移到的目标目录
TARGET_DIR="/hpc2hdd/home/ckwong627/workspace/paddleocr_models"

# 创建目标目录（如果不存在）
mkdir -p "$TARGET_DIR"

# 移动 ~/.paddleocr 内容
if [ -d "$HOME/.paddleocr" ]; then
    echo ">>> 正在移动 ~/.paddleocr 到 $TARGET_DIR ..."
    mv "$HOME/.paddleocr/"* "$TARGET_DIR/" 2>/dev/null
    rmdir "$HOME/.paddleocr"
fi

# 移动 ~/.paddlex/official_models 内容
if [ -d "$HOME/.paddlex/official_models" ]; then
    echo ">>> 正在移动 ~/.paddlex/official_models 到 $TARGET_DIR ..."
    mv "$HOME/.paddlex/official_models/"* "$TARGET_DIR/" 2>/dev/null
    rmdir "$HOME/.paddlex/official_models"
fi

# 写入环境变量到 ~/.bashrc 并立即生效
echo "export PADDLE_OCR_BASE_DIR=$TARGET_DIR" >> ~/.bashrc
export PADDLE_OCR_BASE_DIR="$TARGET_DIR"
source ~/.bashrc

echo "✅ 模型文件已迁移完成，当前模型目录：$PADDLE_OCR_BASE_DIR"
