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
    def __init__(self, image_input_dir, text_output_dir, summary_output_dir, db_path, model_path):
        self.image_input_dir = image_input_dir
        self.text_output_dir = text_output_dir
        self.summary_output_dir = summary_output_dir
        self.db_path = db_path
        self.model_path = model_path

        os.makedirs(self.text_output_dir, exist_ok=True)
        os.makedirs(self.summary_output_dir, exist_ok=True)

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

    def process_images(self):
        print("📸 Bắt đầu trích xuất văn bản từ ảnh bằng DBNet...")
        extract_text_from_cv(
            model_path=self.model_path,
            img_dir=self.image_input_dir,
            backbone="EfficientNet",
            output_text_dir=self.text_output_dir,
            box_thresh=0.5,
            max_workers=8
        )
        print(f"✔ Văn bản đã được trích xuất và lưu tại: {self.text_output_dir}")

    def store_candidate(self, img_path):
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("INSERT INTO cvs (file_name) VALUES (?)", (img_path,))
            conn.commit()
            conn.close()

    def store_candidates(self, cv_data):
        print(f"📂 Đang lưu {len(cv_data)} hồ sơ...")
        # Lấy danh sách tất cả file ảnh từ image_input_dir
        image_files = [os.path.join(root, f) for root, _, files in os.walk(self.image_input_dir) 
                       for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'))]
        
        # Tạo mapping từ tên file không đuôi sang đường dẫn đầy đủ
        img_path_map = {os.path.splitext(os.path.basename(f))[0]: f for f in image_files}
        
        # Chỉ lưu các đường dẫn ảnh tương ứng với cv_data
        img_paths = []
        for file_name in cv_data.keys():
            base_name = file_name  # Tên file không đuôi từ cv_data
            if base_name in img_path_map:
                img_paths.append(img_path_map[base_name])
            else:
                print(f"⚠️ Không tìm thấy ảnh tương ứng cho {file_name}")
        
        if not img_paths:
            print("⚠️ Không có ảnh nào được tìm thấy để lưu!")
            return
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            list(tqdm(executor.map(self.store_candidate, img_paths), total=len(img_paths), desc="📂 Đang lưu CV"))

    def process_embedding(self, cv_id, img_path, cv_text):
        embedding = self.get_bert_embedding(cv_text)
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("UPDATE cvs SET embedding = ? WHERE id = ?", (embedding.tobytes(), cv_id))
            conn.commit()
            conn.close()

    def add_embeddings(self, cv_data):
        self.cursor.execute("SELECT id, file_name FROM cvs WHERE embedding IS NULL")
        # Sử dụng basename của file_name (đường dẫn đầy đủ) để ánh xạ với cv_data
        cv_list = [(cv_id, img_path, cv_data.get(os.path.splitext(os.path.basename(img_path))[0], "")) 
                   for cv_id, img_path in self.cursor.fetchall()]
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

    def visualize_candidates(self, job_description=None, top_candidate=1, cvs=None, similarities=None):
        """
        Hiển thị top N ứng viên từ cơ sở dữ liệu hoặc từ dữ liệu được cung cấp sẵn.
        - Nếu job_description được cung cấp: tính toán similarities từ cơ sở dữ liệu.
        - Nếu cvs và similarities được cung cấp: sử dụng trực tiếp để hiển thị.
        """
        if job_description is not None:  # Tính toán từ cơ sở dữ liệu
            job_embedding = self.get_bert_embedding(job_description)
            self.cursor.execute("SELECT id, file_name, embedding FROM cvs")
            cvs = self.cursor.fetchall()
            if not cvs:
                print("⚠️ Không có dữ liệu trong database!")
                return
            
            cv_embeddings = np.vstack([np.frombuffer(cv[2], dtype=np.float32) for cv in cvs])
            similarities = np.dot(cv_embeddings, job_embedding) / (np.linalg.norm(cv_embeddings, axis=1) * np.linalg.norm(job_embedding))
            
            top_n = min(top_candidate, len(cvs))
            top_indices = np.argsort(similarities)[-top_n:][::-1]
            
            print(f"\n🎯 **Top {top_n} ứng viên phù hợp nhất:**")
            for idx in top_indices:
                print(f"⭐ **ID:** {cvs[idx][0]} | 📝 **File:** {cvs[idx][1]} | 🔥 **Độ tương đồng:** {similarities[idx]:.4f}")
        else:
            if not cvs or similarities is None:
                return
            top_n = min(top_candidate, len(cvs))
            top_indices = np.argsort(similarities)[-top_n:][::-1]

        # Hiển thị ảnh cho tất cả top N ứng viên
        for idx in top_indices:
            cv_id, img_path, _ = cvs[idx]
            similarity = similarities[idx]

            print(f"\n🖼️ Đang kiểm tra ảnh cho ứng viên: {img_path}")
            if os.path.exists(img_path):
                img = cv2.imread(img_path)
                if img is not None:
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    print(f"\n🖼️ **Ứng viên:**\n🆔 **ID:** {cv_id} | 📄 **File:** {img_path} | 🔥 **Độ tương đồng:** {similarity:.4f}")
                    print(f"📷 Tìm thấy ảnh tại: {img_path}")
                    plt.figure(figsize=(8, 6))
                    plt.imshow(img_rgb)
                    plt.title(f"Candidate: {os.path.basename(img_path)} (Similarity: {similarity:.4f})")
                    plt.axis('off')
                    plt.show()
                else:
                    print(f"❌ Không thể đọc ảnh tại {img_path}")
            else:
                print(f"❌ Không tìm thấy ảnh tại {img_path}")

    def run(self, job_description, top_candidate):
        start_time = time.time()
        print("🚀 Bắt đầu pipeline...")

        self.process_images()
        self.summarize_files()
        self.clear_database()
        cv_data = self.load_cv_data()
        if not cv_data:
            print("⚠️ Không có dữ liệu CV, pipeline dừng.")
            return
        self.store_candidates(cv_data)
        self.add_embeddings(cv_data)
        # cvs, similarities = self.compare_job_description(job_description, top_candidate)
        # self.visualize_best_candidate(cvs, similarities)

        end_time = time.time()
        print(f"✅ Pipeline hoàn thành. Tổng thời gian: {end_time - start_time:.2f} giây")
        self.conn.close()

    # Các phương thức còn lại giữ nguyên: clean_text, summarize_text, process_summary, summarize_files, load_cv_data, get_bert_embedding
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

    def load_cv_data(self):
        cv_data = {}
        txt_files = [os.path.join(root, f) for root, _, files in os.walk(self.summary_output_dir) for f in files if f.endswith(".txt")]
        if not txt_files:
            print(f"⚠️ Không tìm thấy file .txt trong {self.summary_output_dir}")
            return cv_data

        print(f"📖 Đang đọc {len(txt_files)} file CV...")
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(lambda p: (os.path.splitext(os.path.basename(p))[0], open(p, 'r', encoding='utf-8').read().strip()), path) for path in txt_files]
            for future in tqdm(as_completed(futures), total=len(txt_files), desc="📖 Đang đọc CV"):
                file_name, content = future.result()
                cv_data[file_name] = content
        return cv_data

    def get_bert_embedding(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", max_length=512, truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state[:, 0, :].squeeze().numpy()