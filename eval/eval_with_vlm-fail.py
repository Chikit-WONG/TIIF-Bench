import argparse
import os
import json
import glob
import time
import re
from tqdm import tqdm
import random
from PIL import Image
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

# —— 初始化模型与处理器（加载一次）——
print("[INFO] Loading Qwen2.5-VL model…")
model_path = "./eval_models/Qwen2.5-VL-7B-Instruct"
processor = AutoProcessor.from_pretrained(model_path)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
)
model.eval()
print("[INFO] Model loaded.")

# —— Prompt 模板 —— 
raw_prompt = '''
You are tasked with carefully examining the provided image and answering the following yes or no questions:

Questions:
##YNQuestions##

Instructions:

1. Answer each question on a separate line, starting with "yes" or "no", followed by a brief reason.
2. Maintain the exact order of the questions in your answers.
3. Provide only one answer per question.
4. Return only the answers—no additional commentary.
5. Each answer must be on its own line.
6. Ensure the number of answers matches the number of questions.
'''


def format_prompt(questions):
    qs = "\n".join(q.strip() for q in questions)
    return raw_prompt.replace("##YNQuestions##", qs)

# —— 图像 resize 校正函数 —— 
patch_size = model.config.vision_config.patch_size

def resize_to_patch_multiple(img: Image.Image) -> Image.Image:
    w, h = img.size
    new_w = (w // patch_size) * patch_size
    new_h = (h // patch_size) * patch_size
    if new_w == 0 or new_h == 0:
        raise ValueError(f"Image {img.size} too small for patch_size {patch_size}")
    return img.resize((new_w, new_h), Image.LANCZOS)

# —— 生成函数 —— 
def generate_with_prompt(prompt: str, img_path: str) -> str:
    img = Image.open(img_path).convert("RGB")
    img = resize_to_patch_multiple(img)
    inputs = processor(text=prompt, images=img, return_tensors="pt").to(model.device)

    #  Debug: 检查 tokens 是否匹配
    if inputs.get("image_token_ids") is not None:
        t = inputs["image_token_ids"].shape[-1]
        f = inputs["pixel_values"].shape[-2] * inputs["pixel_values"].shape[-1] // (patch_size ** 2)
        if t != f:
            raise ValueError(f"tokens {t} != features {f}")

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=512)
    return processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()

# —— 输出解析 —— 
def extract_yes_no(output: str, questions: list[str]) -> list[str]:
    lines = [l.strip() for l in output.splitlines() if l.strip()]
    preds = [m.group(1).lower() for m in (re.match(r'^(yes|no)', l, re.I) for l in lines) if m]
    if len(preds) != len(questions):
        raise ValueError(f"Pred len={len(preds)} != questions len={len(questions)}\nOutput:\n{output}")
    return preds

# —— 主流程 —— 
def main(args):
    tasks = []
    for jf in glob.glob(os.path.join(args.jsonl_dir, "*.jsonl")):
        attr = os.path.splitext(os.path.basename(jf))[0]
        with open(jf, "r", encoding="utf-8") as f:
            items = [json.loads(l) for l in f if l.strip()]
        for desc in ["long_description", "short_description"]:
            for idx, entry in enumerate(items):
                img_path = os.path.join(args.image_dir, attr, args.evaluated_model, desc, f"{idx}.png")
                out_path = os.path.join(args.output_dir, attr, args.evaluated_model, desc, f"{idx}.json")
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                if os.path.exists(out_path):
                    continue
                questions = entry.get("yn_question_list", [])
                gt = entry.get("yn_answer_list", [])
                prompt = format_prompt(questions)
                try:
                    output = generate_with_prompt(prompt, img_path)
                    pred = extract_yes_no(output, questions)
                    json.dump({"questions": questions, "gt": gt, "pred": pred, "output": output},
                               open(out_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
                    print(f"Saved {out_path}")
                    print(f"Try:{pred}")
                except Exception as e:
                    print(f"[Error] {img_path} : {e}")
                    print("Exception")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl_dir", type=str, required=True)
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--evaluated_model", type=str, required=False, help="Name of the eval model")
    args = parser.parse_args()
    main(args)
