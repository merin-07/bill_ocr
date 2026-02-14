import json
from paddleocr import PaddleOCR
from ai_categorizer import categorize_receipt

ocr = PaddleOCR(lang='en')

def extract_text_from_image(image_path: str) -> str:
    result = ocr.ocr(image_path)

    extracted_text = []

    for line in result:
        for word_info in line:
            text = word_info[1][0]
            extracted_text.append(text)

    return "\n".join(extracted_text)


if __name__ == "__main__":
    print("\n--- PADDLE OCR OUTPUT ---\n")

    ocr_text = extract_text_from_image("bill.png")
    print(ocr_text)

    print("\n--- AI STRUCTURED OUTPUT ---\n")

    structured_data = categorize_receipt(ocr_text)
    print(json.dumps(structured_data, indent=2))
