import os
import sqlite3
import numpy as np
import cv2
import torch
import pytesseract
from tqdm import tqdm
from transformers import pipeline, BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import time
import random
import re
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import matplotlib.pyplot as plt
import multiprocessing
from datatools.extract2txt import extract_text_from_cv
class CVPipeline:
    def __init__(self, image_input_dir,text_output_dir,summary_output_dir,db_path, model_path):
        self.image_input_dir = image_input_dir
        self.text_output_dir = text_output_dir
        self.summary_output_dir = summary_output_dir
        self.db_path = db_path
        self.model_path = model_path  # Đường dẫn tới trọng số mô hình DBNet

        # Tạo các thư mục đầu ra nếu chưa tồn tại
        os.makedirs(self.text_output_dir, exist_ok=True)
        os.makedirs(self.summary_output_dir, exist_ok=True)

        # Khởi tạo các công cụ
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
        self.model = BertModel.from_pretrained('bert-base-multilingual-cased')
        self.summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", 
                                  device=0 if torch.cuda.is_available() else -1, truncation=True)
        self.conn, self.cursor = self.init_database()
        self.lock = threading.Lock()
        self.max_workers = multiprocessing.cpu_count()
        print(f"Using device: {'GPU' if torch.cuda.is_available() else 'CPU'}, Max workers: {self.max_workers}")

    def init_database(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS cvs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_name TEXT NOT NULL,
            embedding BLOB
        )''')
        conn.commit()
        return conn, cursor

    def clear_database(self):
        self.cursor.execute("DELETE FROM cvs")
        self.conn.commit()
        print("🗑️ Database đã được xóa.")

    # Bước 1: Trích xuất văn bản từ ảnh bằng DBNet
    def process_images(self):
        print("📸 Bắt đầu trích xuất văn bản từ ảnh bằng DBNet...")
        extract_text_from_cv(
            model_path=self.model_path,
            img_dir=self.image_input_dir,
            output_text_dir=self.text_output_dir,
            box_thresh=0.5,
            max_workers=self.max_workers
        )
        print(f"✔ Văn bản đã được trích xuất và lưu tại: {self.text_output_dir}")

    # Bước 2: Tóm tắt văn bản
    def clean_text(self, text):
        text = re.sub(r'[B][a-zA-Z]*[B][a-zA-Z]*\s*', '', text)
        text = re.sub(r'\|.*$', '', text, flags=re.MULTILINE)
        text = re.sub(r'[-=]+', '', text)
        text = re.sub(r'(\+\d{1,3}\s?)?(\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4})', '', text)
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text)
        text = re.sub(r'\d+\s+[A-Za-z\s]+(St|Street|Ave|Avenue|Rd|Road|Blvd|Boulevard|Apt|Unit)\b', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def summarize_text(self, text):
        cleaned_text = self.clean_text(text)
        relevant_sections = []
        lines = cleaned_text.split('\n')
        for line in lines:
            line = line.strip()
            if any(keyword in line.lower() for keyword in [
                'experience', 'skills', 'certification', 'achievement', 'work', 'job', 
                'training', 'project', 'role', 'responsibility', 'technical', 'language'
            ]):
                relevant_sections.append(line)
        
        relevant_text = '\n'.join(relevant_sections)
        if not relevant_text:
            return "No relevant experience or skills found."
        
        max_chunk_length = 1000
        chunks = [relevant_text[i:i + max_chunk_length] for i in range(0, len(relevant_text), max_chunk_length)]
        summaries = []
        try:
            for chunk in chunks:
                summary = self.summarizer(chunk, max_length=150, min_length=50, do_sample=False)
                summaries.append(summary[0]['summary_text'])
            return ' '.join(summaries)
        except Exception as e:
            return f"Error summarizing text: {str(e)}"

    def process_summary(self, file_info):
        root, file = file_info
        input_path = os.path.join(root, file)
        with open(input_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        summary = self.summarize_text(content)
        relative_path = os.path.relpath(root, self.text_output_dir)
        output_subdir = os.path.join(self.summary_output_dir, relative_path)
        os.makedirs(output_subdir, exist_ok=True)
        
        output_path = os.path.join(output_subdir, file)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(summary)

    def summarize_files(self):
        all_txt_files = [(root, file) for root, _, files in os.walk(self.text_output_dir) for file in files if file.endswith(".txt")]
        if not all_txt_files:
            print(f"⚠️ Không tìm thấy file .txt trong {self.text_output_dir}")
            return
        print(f"✂️ Đang tóm tắt {len(all_txt_files)} file...")
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            list(tqdm(executor.map(self.process_summary, all_txt_files), total=len(all_txt_files), desc="✂️ Tóm tắt"))

    # Bước 3: Xử lý CV và xếp hạng
    def load_cv_data(self):
        cv_data = {}
        txt_files = [os.path.join(root, f) for root, _, files in os.walk(self.summary_output_dir) for f in files if f.endswith(".txt")]
        if not txt_files:
            print(f"⚠️ Không tìm thấy file .txt trong {self.summary_output_dir}")
            return cv_data

        print(f"📖 Đang đọc {len(txt_files)} file CV...")
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(lambda p: (os.path.relpath(p, self.summary_output_dir).replace('.txt', ''), open(p, 'r', encoding='utf-8').read().strip()), path) for path in txt_files]
            for future in tqdm(as_completed(futures), total=len(txt_files), desc="📖 Đang đọc CV"):
                file_name, content = future.result()
                cv_data[file_name] = content
        return cv_data

    def store_candidate(self, file_name):
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("INSERT INTO cvs (file_name) VALUES (?)", (file_name,))
            conn.commit()
            conn.close()

    def store_candidates(self, cv_data):
        print(f"📂 Đang lưu {len(cv_data)} hồ sơ...")
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            list(tqdm(executor.map(self.store_candidate, cv_data.keys()), total=len(cv_data), desc="📂 Đang lưu CV"))

    def get_bert_embedding(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", max_length=512, truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state[:, 0, :].squeeze().numpy()

    def process_embedding(self, cv_id, file_name, cv_text):
        embedding = self.get_bert_embedding(cv_text)
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("UPDATE cvs SET embedding = ? WHERE id = ?", (embedding.tobytes(), cv_id))
            conn.commit()
            conn.close()

    def add_embeddings(self, cv_data):
        self.cursor.execute("SELECT id, file_name FROM cvs WHERE embedding IS NULL")
        cv_list = [(cv_id, file_name, cv_data[file_name]) for cv_id, file_name in self.cursor.fetchall() if file_name in cv_data]
        if not cv_list:
            print("⚠️ Không có CV nào cần tạo embedding!")
            return
        print(f"🔄 Đang tạo embeddings cho {len(cv_list)} CV...")
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            list(tqdm(executor.map(lambda x: self.process_embedding(*x), cv_list), total=len(cv_list), desc="🔄 Đang xử lý embeddings"))

    def compare_job_description(self, job_description, top_candidate):
        job_embedding = self.get_bert_embedding(job_description)
        self.cursor.execute("SELECT id, file_name, embedding FROM cvs")
        cvs = self.cursor.fetchall()
        if not cvs:
            print("⚠️ Không có dữ liệu trong database!")
            return [], []
        
        cv_embeddings = np.vstack([np.frombuffer(cv[2], dtype=np.float32) for cv in cvs])
        similarities = np.dot(cv_embeddings, job_embedding) / (np.linalg.norm(cv_embeddings, axis=1) * np.linalg.norm(job_embedding))
        
        top_n = min(top_candidate, len(cvs))
        top_indices = np.argsort(similarities)[-top_n:][::-1]
        
        print(f"\n🎯 **Top {top_n} ứng viên phù hợp nhất:**")
        for idx in top_indices:
            print(f"⭐ **ID:** {cvs[idx][0]} | 📝 **File:** {cvs[idx][1]} | 🔥 **Độ tương đồng:** {similarities[idx]:.4f}")
        return cvs, similarities

    def check_image(self, image_path):
        if os.path.exists(image_path):
            img = cv2.imread(image_path)
            if img is not None:
                return image_path, img
        return None, None

    def visualize_best_candidate(self, cvs, similarities):
        if not cvs:
            return
        best_idx = np.argmax(similarities)
        cv_id, file_name, _ = cvs[best_idx]
        similarity = similarities[best_idx]

        image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"]
        image_paths = [os.path.join(self.image_input_dir, file_name + ext) for ext in image_extensions]

        print(f"\n🖼️ Đang kiểm tra ảnh cho ứng viên tốt nhất: {file_name}")
        with ThreadPoolExecutor(max_workers=len(image_extensions)) as executor:
            futures = [executor.submit(self.check_image, path) for path in image_paths]
            for future in as_completed(futures):
                image_path, img = future.result()
                if img is not None:
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    print(f"\n🖼️ **Ứng viên xuất sắc nhất:**\n🆔 **ID:** {cv_id} | 📄 **File:** {file_name} | 🔥 **Độ tương đồng:** {similarity:.4f}")
                    print(f"📷 Tìm thấy ảnh tại: {image_path}")
                    plt.figure(figsize=(8, 6))
                    plt.imshow(img_rgb)
                    plt.title(f"Best Candidate: {os.path.basename(file_name)} (Similarity: {similarity:.4f})")
                    plt.axis('off')
                    plt.show()
                    return
        print(f"❌ Không tìm thấy ảnh cho {file_name} với các đuôi {image_extensions}")

    def run(self, job_description, top_candidate):
        start_time = time.time()
        print("🚀 Bắt đầu pipeline...")

        # Bước 1: Trích xuất văn bản từ ảnh bằng DBNet
        self.process_images()

        # Bước 2: Tóm tắt văn bản
        self.summarize_files()

        # Bước 3: Xử lý và xếp hạng CV
        self.clear_database()
        cv_data = self.load_cv_data()
        if not cv_data:
            print("⚠️ Không có dữ liệu CV, pipeline dừng.")
            return
        self.store_candidates(cv_data)
        self.add_embeddings(cv_data)
        cvs, similarities = self.compare_job_description(job_description, top_candidate)
        self.visualize_best_candidate(cvs, similarities)

        end_time = time.time()
        print(f"✅ Pipeline hoàn thành. Tổng thời gian: {end_time - start_time:.2f} giây")
        self.conn.close()