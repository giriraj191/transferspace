import os
import shutil
import json
from PIL import Image
import pytesseract
from cleantext import clean  # using clean-text 0.6.0
import warnings
warnings.simplefilter("ignore")

# Set Tesseract path
pytesseract.pytesseract.tesseract_cmd = "C:\\Users\\giriraj\\AppData\\Local\\Programs\\Tesseract-OCR\\tesseract.exe"


def clean_json(data):
    if isinstance(data, dict):
        return {k: clean_json(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [clean_json(item) for item in data]
    elif isinstance(data, str):
        # Use clean-text to ensure text is ASCII and clean.
        return clean(data, fix_unicode=True, to_ascii=True, lower=False)
    else:
        return data


def cleaner():
    raw_dir =  os.path.join("artifacts", "raw")
    cleaned_dir = os.path.join("artifacts", "cleaned")

    # Walk through the raw directory
    for root, dirs, files in os.walk(raw_dir):
        # Compute relative path to maintain folder structure
        relative_path = os.path.relpath(root, raw_dir)
        target_dir = os.path.join(cleaned_dir, relative_path)
        os.makedirs(target_dir, exist_ok=True)

        for file in files:
            source_file_path = os.path.join(root, file)
            # Determine target file path (we may change the extension later for images)
            target_file_path = os.path.join(target_dir, file)
            ext = os.path.splitext(file)[1].lower()

            if ext in ['.png', '.jpg', '.jpeg', '.bmp', '.gif']:
                try:
                    image = Image.open(source_file_path)
                    ocr_text = pytesseract.image_to_string(image)
                    if len(ocr_text) > 3:
                        base_name = os.path.splitext(file)[0]
                        target_txt_path = os.path.join(target_dir, base_name + ".txt")
                        with open(target_txt_path, "w", encoding="utf-8") as f:
                            f.write(ocr_text)
                        print(f"OCR text saved for image: {source_file_path}")
                except Exception as e:
                    print(f"Error processing image {source_file_path}: {e}")

            elif ext == ".json":
                try:
                    with open(source_file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    cleaned_data = clean_json(data)
                    with open(target_file_path, "w", encoding="utf-8") as f:
                        json.dump(cleaned_data, f, indent=4)
                    print(f"Cleaned JSON saved for file: {source_file_path}")
                except Exception as e:
                    print(f"Error processing JSON {source_file_path}: {e}")

            elif ext == ".csv":
                try:
                    shutil.copy2(source_file_path, target_file_path)
                    print(f"Saved CSV file: {source_file_path}")
                except Exception as e:
                    print(f"Error copying CSV file {source_file_path}: {e}")

            else:
                try:
                    shutil.copy2(source_file_path, target_file_path)
                    print(f"Saved Unsupported file: {source_file_path}")
                except Exception as e:
                    print(f"Error copying file {source_file_path}: {e}")