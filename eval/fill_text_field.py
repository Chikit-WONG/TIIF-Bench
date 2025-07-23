import os
import json
import re
from tqdm import tqdm

PROMPT_PATH = "./data/testmini_prompts/text_prompts.jsonl"
OCR_RESULTS_BASE_DIR = "./eval_results/paddleocr_results"

def extract_words(text):
    """提取英文关键词并统一为小写"""
    return re.findall(r'"([a-zA-Z0-9\-]+)"', text)

def load_ground_truth(prompt_path):
    """加载 text_prompts.jsonl，建立 image_name -> text 列表的映射"""
    mapping = {}
    with open(prompt_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            obj = json.loads(line)
            short = extract_words(obj.get("short_description", ""))  # 只提取 short_description
            mapping[f"{idx}.png"] = list(set(short))  # 去重
    return mapping

def process_all_models(ocr_base_dir, prompt_mapping):
    for model_name in os.listdir(ocr_base_dir):
        model_dir = os.path.join(ocr_base_dir, model_name)
        text_json_path = os.path.join(model_dir, "text.json")
        if not os.path.exists(text_json_path):
            print(f"[Skip] No text.json in {model_dir}")
            continue

        with open(text_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        updated_count = 0
        for item in data:
            image_name = item["image_name"]
            if "text" not in item or not item["text"]:
                item["text"] = prompt_mapping.get(image_name, [])
                updated_count += 1

        with open(text_json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

        print(f"[Updated] {model_name}/text.json — {updated_count} entries patched")

if __name__ == "__main__":
    prompt_mapping = load_ground_truth(PROMPT_PATH)
    process_all_models(OCR_RESULTS_BASE_DIR, prompt_mapping)
