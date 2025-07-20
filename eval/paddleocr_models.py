import os
import json
from paddleocr import PaddleOCR
from tqdm import tqdm

# def extract_text_with_paddleocr(image_path, ocr_engine):
#     # result = ocr_engine.ocr(image_path, cls=False)
#     result = ocr_engine.predict(image_path)
#     ocr_results = []
#     for idx in range(len(result)):
#         res = result[idx]
#     return res

# def extract_text_with_paddleocr(image_path, ocr_engine):
#     results = ocr_engine.predict(image_path)

#     ocr_results = []
#     for res in results:
#         ocr_results.append({
#             "text": res.text,             # 识别的文字
#             "score": res.score,           # 置信度
#             "box": res.box,               # 文本框坐标 [4个点]
#         })

#         # # 可选：保存图像和json
#         # res.save_to_img("output")         # 可加命名
#         # res.save_to_json("output")        # 可加命名

#     return ocr_results

def extract_text_with_paddleocr(image_path, ocr_engine):
    results = ocr_engine.predict(image_path)

    for res in results:
        res_json = res.json
        res_text = res_json['res']['rec_texts'] if 'res' in res_json and 'rec_texts' in res_json['res'] else []
        return res_text





def process_images(short_prompt_dir, long_prompt_dir, output_json_path, model_name):

    ocr = PaddleOCR(lang='en')

    image_extensions = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}
    image_names = sorted(
        [f for f in os.listdir(short_prompt_dir) if os.path.splitext(f.lower())[1] in image_extensions],
        key=lambda x: int(os.path.splitext(x)[0])
    )

    results = []

    for img_name in tqdm(image_names, desc=f"Processing images for {model_name}"):
        short_img_path = os.path.join(short_prompt_dir, img_name)
        long_img_path = os.path.join(long_prompt_dir, img_name)

        simple_results = extract_text_with_paddleocr(short_img_path, ocr)
        enhanced_results = extract_text_with_paddleocr(long_img_path, ocr)

        result_dict = {
            "image_name": img_name,
            "short_image_ocr_results": simple_results,
            "long_image_ocr_results": enhanced_results
        }
        results.append(result_dict)

    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print(f"OCR results saved: {output_json_path}")

def process_all_folders(base_dir, output_base_dir):
    """
    Process all folders under base_dir
    
    base_dir 应该是 output/ 目录，包含各类 <data_type>/<model_name>/short_description 图像
    
    Process all folders under base_dir and group outputs by model_name
    """

    # os.makedirs(output_base_dir, exist_ok=True)

    # folders = [f for f in os.listdir(base_dir) 
    #           if os.path.isdir(os.path.join(base_dir, f))]

    # for folder in folders:
    #     print(f"\n Processing: {folder}")

    #     short_prompt_dir = os.path.join(base_dir, folder, "short_description")
    #     long_prompt_dir = os.path.join(base_dir, folder, "long_description")
    #     output_json_path = os.path.join(output_base_dir, f"ocr_results_{folder}.json")

    #     if not os.path.exists(short_prompt_dir) or not os.path.exists(long_prompt_dir):
    #         print(f"Warning: {folder} is missing either the short_description or long_description directory, skipping.")
    #         continue

    #     process_images(short_prompt_dir, long_prompt_dir, output_json_path)
    
    for data_type in os.listdir(base_dir):
        if data_type != 'text':
            continue
        type_path = os.path.join(base_dir, data_type)
        if not os.path.isdir(type_path):
            continue

        for model_name in os.listdir(type_path):
            model_path = os.path.join(type_path, model_name)
            if not os.path.isdir(model_path):
                continue

            print(f"\nProcessing: {model_name}/{data_type}")
            short_prompt_dir = os.path.join(model_path, "short_description")
            long_prompt_dir = os.path.join(model_path, "long_description")

            if not os.path.exists(short_prompt_dir) or not os.path.exists(long_prompt_dir):
                print(f"Warning: {model_path} missing expected subfolders.")
                continue

            # ✅ 创建每个模型的输出目录
            output_model_dir = os.path.join(output_base_dir, model_name)
            os.makedirs(output_model_dir, exist_ok=True)

            # ✅ 保存为 <data_type>.json
            output_json_path = os.path.join(output_model_dir, f"{data_type}.json")

            process_images(short_prompt_dir, long_prompt_dir, output_json_path, model_name)

if __name__ == "__main__":
    base_dir = "./output"
    output_base_dir="./eval_results/paddleocr_results"

    process_all_folders(base_dir,output_base_dir)