#!/bin/bash
#SBATCH -p debug        # 指定GPU队列
#SBATCH -o output.txt   # 指定作业标准输出文件
#SBATCH -e err.txt      # 指定作业标准错误输出文件
#SBATCH -n 1            # 指定CPU总核心数
#SBATCH --gres=gpu:1    # 指定GPU卡数
#SBATCH -D .            # 指定作业执行路径为当前目录
#SBATCH --time=00:30:00 # 设置作业的最大运行时间为30分钟

echo "[INFO] 当前节点：$(hostname)"
echo "[INFO] 启动时间：$(date)"

# ===== 0. 设置工作目录 =====
cd /hpc2hdd/home/ckwong627/workspace/TIIF-Bench

# 加载CUDA模块（如果需要）
module load cuda/12.1  

# 激活 Conda 环境
source /hpc2hdd/home/ckwong627/miniconda3/etc/profile.d/conda.sh
conda activate TIIF-Bench

# 设置 LD_LIBRARY_PATH，只使用 conda 下的 .so 文件
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
echo "[INFO] LD_LIBRARY_PATH=$LD_LIBRARY_PATH"

# 可选调试：查看 .so 是否在位
echo "[DEBUG] Verifying libcudnn_graph.so presence:"
ls -l $CONDA_PREFIX/lib/libcudnn_graph.so* || echo "[DEBUG] 未找到 libcudnn_graph.so，跳过"

# ===== 2. 设置变量 =====
export JSONL_DIR=data/testmini_eval_prompts
export IMAGE_DIR=output
export MODEL_NAME=flux1_dev
export OUTPUT_DIR=eval_results
export BASE_URL=http://localhost:8000
export MODEL="Qwen2.5-VL-7B-Instruct"

# ===== 3. 后台启动 vllm 模型服务 =====
echo "[INFO] 启动 vllm 模型服务..."
vllm serve \
  ./eval_models/Qwen2.5-VL-7B-Instruct \
  --port 8000 \
  --host 0.0.0.0 \
  --dtype bfloat16 \
  --serve-mode openai > vllm_server.log 2>&1 &

# ===== 4. 等待服务完全启动（嵌入检测逻辑）=====
echo "[INFO] 等待模型服务启动..."
MAX_RETRIES=60
SLEEP_SECONDS=5
URL="$BASE_URL/v1/models"
for ((i=1; i<=MAX_RETRIES; i++)); do
  if LD_LIBRARY_PATH= curl -s --connect-timeout 2 "$URL" | grep -q '"data"'; then
    echo "[INFO] 模型服务已就绪！"
    break
  else
    echo "[INFO] 第 $i 次检测失败，继续等待..."
    sleep $SLEEP_SECONDS
  fi
  if [ "$i" -eq "$MAX_RETRIES" ]; then
    echo "[ERROR] 超过最大等待次数，模型服务仍未就绪。"
    exit 1
  fi
done

# ===== 5. 开始运行评估 =====
python eval/eval_with_vlm.py \
  --jsonl_dir $JSONL_DIR \
  --image_dir $IMAGE_DIR \
  --eval_model $MODEL_NAME \
  --output_dir $OUTPUT_DIR \
  --base_url $BASE_URL \
  --model "$MODEL"

echo "[INFO] 评估完成于：$(date)"
