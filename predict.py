import torch
import re
import json
from PIL import Image
from transformers import DonutProcessor, VisionEncoderDecoderModel

# Hàm dự đoán thông tin hóa đơn
def extract_invoice_info(image_path, model_dir="./donut-finetuned-invoice"):
    # Load ảnh
    image = Image.open(image_path).convert("RGB")

    # Load processor và mô hình
    processor_inf = DonutProcessor.from_pretrained(model_dir)
    model_inf = VisionEncoderDecoderModel.from_pretrained(model_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_inf.to(device)

    # Tạo prompt và input cho mô hình
    task_prompt = "<s_invoice>"
    decoder_input_ids = processor_inf.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids
    pixel_values = processor_inf(image, return_tensors="pt").pixel_values

    # Sinh output
    outputs = model_inf.generate(
        pixel_values.to(device),
        decoder_input_ids=decoder_input_ids.to(device),
        max_length=512,
        pad_token_id=processor_inf.tokenizer.pad_token_id,
        eos_token_id=processor_inf.tokenizer.eos_token_id,
        bad_words_ids=[[processor_inf.tokenizer.unk_token_id]]
    )

    # Giải mã kết quả
    result = processor_inf.batch_decode(outputs, skip_special_tokens=True)[0]
    result = re.sub(r"<.*?>", "", result).strip()

    try:
        return json.loads(result)
    except:
        return {"output": result}

# Chạy thử dự đoán
if __name__ == "__main__":
    test_image = "./images/BILL_1.jpg" 
    print("\n=== Kết quả dự đoán ===")
    output = extract_invoice_info(test_image)
    print(json.dumps(output, indent=2, ensure_ascii=False))
