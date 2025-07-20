#!/bin/bash
#SBATCH -p debug
#SBATCH -o ./temp/output.txt
#SBATCH -e ./temp/err.txt
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -D .

module load cuda/12.1

source /hpc2hdd/home/ckwong627/miniconda3/etc/profile.d/conda.sh
conda activate TIIF-Bench

source ./scripts/check_and_fix_cudnn.sh

echo "Job started at $(date)"

# 启动 vLLM 服务并放入后台，同时记录进程号
vllm serve \
  ./eval_models/Qwen2.5-VL-7B-Instruct \
  --port 8000 \
  --host 0.0.0.0 \
  --served-model-name qwen7b \
  --dtype float16 \
  > ./temp/vllm.log 2>&1 &

VLLM_PID=$!

# 判断是否成功
if ! ps -p $VLLM_PID > /dev/null; then
  echo "❌ vLLM 启动失败！PID 无效，检查 vllm 是否安装"
  exit 1
else
  echo "✅ vLLM 启动成功，PID=$VLLM_PID"
fi

echo -e "\nvLLM 服务已启动, PID=$VLLM_PID"
sleep 60  # 给 vLLM 一点时间启动完成（可调整）

# 执行测试脚本
# python vllm_qwen_test.py
# ===== 设置变量 =====
export JSONL_DIR=data/testmoremini_eval_prompts # 这里可以替换为你想要评估的 JSONL 文件目录
export IMAGE_DIR=output
export EVAL_MODEL=flux1_dev  # 这里可以替换为你想要评估的模型名称
export OUTPUT_DIR=eval_results
export MODEL=qwen7b
export BASE_URL=http://localhost:8000/v1

python eval/eval_with_vlm-qwen.py \
      --jsonl_dir $JSONL_DIR \
      --image_dir $IMAGE_DIR \
      --eval_model $EVAL_MODEL \
      --output_dir $OUTPUT_DIR \
      --base_url $BASE_URL \
      --model $MODEL

# 测试完成后杀掉 vLLM 服务进程
kill $VLLM_PID
echo -e "\nvLLM 服务已关闭, 作业结束时间：$(date)"

conda deactivate
