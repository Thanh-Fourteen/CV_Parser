import os
import cv2
import numpy as np
import tensorflow as tf
import pytesseract
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.model import DBNet
from config import DBConfig
from inference import polygons_from_bitmap


cfg = DBConfig()

def resize_image(image, image_short_side=736):
    h, w, _ = image.shape
    if h < w:
        new_h = image_short_side
        new_w = int((new_h / h * w // 32) * 32)
    else:
        new_w = image_short_side
        new_h = int((new_w / w * h // 32) * 32)
    return cv2.resize(image, (new_w, new_h))

def process_image(img_name, img_dir, output_text_dir, model, box_thresh=0.5):
    """Xử lý một ảnh và trích xuất văn bản."""
    img_path = os.path.join(img_dir, img_name)
    image = cv2.imread(img_path)
    if image is None:
        print(f"✘ Failed to load image: {img_path}")
        return

    src_img, (h, w) = image.copy(), image.shape[:2]
    image = resize_image(image).astype(np.float32) - np.array([103.939, 116.779, 123.68])
    pred = model.predict(tf.convert_to_tensor(np.expand_dims(image, axis=0)))[0]

    boxes, _ = polygons_from_bitmap(pred, pred > 0.3, w, h, box_thresh=box_thresh)

    text_file_path = os.path.join(output_text_dir, f"{os.path.splitext(img_name)[0]}.txt")
    with open(text_file_path, "w", encoding="utf-8") as text_file:
        for box in boxes:
            x_min, y_min = np.min(box, axis=0).astype(int)
            x_max, y_max = np.max(box, axis=0).astype(int)
            roi = src_img[y_min:y_max, x_min:x_max]

            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            roi_thresh = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

            text = pytesseract.image_to_string(roi_thresh, lang="vie+eng", config="--psm 6").strip()
            if text:
                text_file.write(text + "\n")

    print(f"✔ Extracted text saved: {text_file_path}")

def extract_text_from_cv(model_path, img_dir, output_text_dir, backbone, box_thresh=0.5, max_workers=None):
    """Trích xuất văn bản từ CV sử dụng đa luồng."""
    os.makedirs(output_text_dir, exist_ok=True)
    
    # Khởi tạo mô hình
    model = DBNet(cfg, model='inference',backbone=backbone)
    model.load_weights(model_path, by_name=True, skip_mismatch=True)

    # Lấy danh sách ảnh
    img_names = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg','webp'))]
    if not img_names:
        print("✘ No images found in the input directory!")
        return

    # Đặt số luồng mặc định dựa trên CPU nếu không chỉ định
    if max_workers is None:
        max_workers = min(os.cpu_count() or 4, len(img_names))  # Không vượt quá số ảnh
    
    print(f"Processing {len(img_names)} images with {max_workers} threads...")

    # Sử dụng ThreadPoolExecutor để xử lý đa luồng
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(process_image, img_name, img_dir, output_text_dir, model, box_thresh)
            for img_name in img_names
        ]
        # Theo dõi tiến độ với tqdm
        for future in tqdm(futures, total=len(img_names), desc="Extracting text"):
            try:
                future.result()  # Chờ kết quả từ mỗi luồng và xử lý lỗi nếu có
            except Exception as e:
                print(f"✘ Error processing an image: {e}")

if __name__ == '__main__':
    extract_text_from_cv(
        model_path = os.path.join(os.getcwd(), "db_19_2.6895_3.3356.h5"),
        img_dir=os.path.join(os.getcwd(), "Arts resumes"),
        output_text_dir="output_text",
        box_thresh=0.5,
        max_workers=None  # Sẽ tự động chọn số luồng tối ưu
    )