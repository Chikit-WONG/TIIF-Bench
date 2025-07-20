#!/bin/bash
#SBATCH -p debug                  # 指定GPU队列
#SBATCH -o output-run_eval.txt             # 指定作业标准输出文件
#SBATCH -e err-run_eval.txt                # 指定作业标准错误输出文件
#SBATCH -n 1                      # 指定CPU总核心数
#SBATCH --gres=gpu:1              # 指定GPU卡数
#SBATCH -D .                      # 指定作业执行路径为当前目录
#SBATCH --time=00:30:00           # 设置作业的最大运行时间为30分钟

echo "[INFO] 当前节点：$(hostname)"
echo "[INFO] 启动时间：$(date)"

# ===== 0. 设置工作目录 =====
cd /hpc2hdd/home/ckwong627/workspace/TIIF-Bench

# 加载CUDA模块（如需）
module load cuda/12.1

# export LD_PRELOAD=""
# export LD_LIBRARY_PATH=$CONDA_PREFIX/lib

# ===== 1. 激活 Conda 环境 =====
source /hpc2hdd/home/ckwong627/miniconda3/etc/profile.d/conda.sh
conda activate TIIF-Bench

source check_and_fix_cudnn.sh

# ===== 2. 设置变量 =====
export JSONL_DIR=data/testmoremini_eval_prompts # 这里可以替换为你想要评估的 JSONL 文件目录
export IMAGE_DIR=output
export EVALUATED_MODEL=flux1_dev  # 这里可以替换为你想要评估的模型名称
export OUTPUT_DIR=eval_results
export MODEL=qwen7b

# ===== 3. 运行本地推理评估脚本（不再依赖 vllm 服务）=====
if [[ "$DEBUG" == "1" ]]; then
    echo "[INFO] 正在本地评估图文模型：$EVALUATED_MODEL"
    python -m debugpy --listen 0.0.0.0:5678 --wait-for-client eval/eval_with_vlm.py \
      --jsonl_dir $JSONL_DIR \
      --image_dir $IMAGE_DIR \
      --evaluated_model $EVALUATED_MODEL \
      --output_dir $OUTPUT_DIR
    echo "[INFO] 评估完成于：$(date)"
else
    echo "[INFO] 正在本地评估图文模型：$EVALUATED_MODEL"
    python eval/eval_with_vlm.py \
      --jsonl_dir $JSONL_DIR \
      --image_dir $IMAGE_DIR \
      --evaluated_model $EVALUATED_MODEL \
      --output_dir $OUTPUT_DIR \
    echo "[INFO] 评估完成于：$(date)"
fi
