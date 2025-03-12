import os
import sqlite3
import numpy as np
import cv2
import torch
from tqdm import tqdm
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import matplotlib.pyplot as plt

class CVParser:
    def __init__(self, 
                 db_path=os.path.join(os.getcwd(), 'Dataset4DBNet', 'CV_parser_database.db'), 
                 image_folder=os.path.join(os.getcwd(), 'Dataset4DBNet', 'Extract')):
        self.db_path = db_path  
        self.image_folder = image_folder  
        if not os.path.exists(self.db_path):
            raise FileNotFoundError(f"Không tìm thấy file database tại {self.db_path}")
        if not os.path.exists(self.image_folder):
            os.makedirs(self.image_folder, exist_ok=True)
            print(f"📁 Đã tạo thư mục ảnh: {self.image_folder}")

        self.conn, self.cursor = self.init_database()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
        self.model = BertModel.from_pretrained('bert-base-multilingual-cased')
        self.lock = threading.Lock()

    def init_database(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        return conn, cursor

    def get_bert_embedding(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", max_length=512, truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state[:, 0, :].squeeze().numpy()

    def compare_job_description(self, job_description, top_candidate):
        start_time = time.time()
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

        end_time = time.time()
        execution_time = end_time - start_time

        print(f"\n🎯 **Top {top_n} ứng viên phù hợp nhất:**")
        for idx in top_indices:
            print(f"⭐ **ID:** {cvs[idx][0]} | 📝 **File:** {cvs[idx][1]} | 🔥 **Độ tương đồng:** {similarities[idx]:.4f}")
        
        print(f"\n⏳ **Thời gian truy xuất ứng viên:** {execution_time:.4f} giây")
        return cvs, similarities

    def check_image(self, image_path):
        """Hàm kiểm tra file ảnh, chạy trong luồng riêng"""
        if os.path.exists(image_path):
            img = cv2.imread(image_path)
            if img is not None:
                return image_path, img
        return None, None
    
    def visualize_best_candidate(self, cvs, similarities, top_candidate):
        if not cvs:
            return

        top_n = min(top_candidate, len(cvs))
        top_indices = np.argsort(similarities)[-top_n:][::-1]
        
        image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"]
        
        # Tạo subplot cho top_n ảnh
        fig, axes = plt.subplots(1, top_n, figsize=(top_n * 4, 5))
        if top_n == 1:
            axes = [axes]  # Đảm bảo axes luôn là list để duyệt

        print(f"\n🖼️ Đang kiểm tra ảnh cho {top_n} ứng viên hàng đầu:")
        for i, idx in enumerate(top_indices):
            cv_id, file_name, _ = cvs[idx]
            similarity = similarities[idx]
            
            # Sử dụng os.getcwd() làm root và kết hợp với file_name từ database
            full_file_path = os.path.join(os.getcwd(), 'Dataset4DBNet', file_name)
            
            # Tạo danh sách các đường dẫn thử nghiệm với các đuôi file
            image_paths = [full_file_path]
            if not any(full_file_path.endswith(ext) for ext in image_extensions):
                image_paths = [full_file_path + ext for ext in image_extensions]
            
            print(f"\n🔍 Kiểm tra ứng viên: {file_name}")
            with ThreadPoolExecutor(max_workers=len(image_extensions) + 1) as executor:
                futures = [executor.submit(self.check_image, path) for path in image_paths]
                found = False
                for future in as_completed(futures):
                    image_path, img = future.result()
                    if img is not None:
                        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        print(f"🖼️ **Ứng viên:**\n🆔 **ID:** {cv_id} | 📄 **File:** {file_name} | 🔥 **Độ tương đồng:** {similarity:.4f}")
                        print(f"📷 Tìm thấy ảnh tại: {image_path}")
                        axes[i].imshow(img_rgb)
                        axes[i].set_title(f"{os.path.basename(file_name)}\n(Similarity: {similarity:.4f})")
                        axes[i].axis('off')
                        found = True
                        break
                if not found:
                    print(f"❌ Lỗi: Không tìm thấy file ảnh nào cho {file_name} tại các đường dẫn:")
                    for path in image_paths:
                        print(f"  - {path}")
                    axes[i].text(0.5, 0.5, 'No Image', ha='center', va='center')
                    axes[i].set_title(f"{os.path.basename(file_name)}\n(Similarity: {similarity:.4f})")
                    axes[i].axis('off')

        plt.tight_layout()
        plt.show()

    def count_candidates(self):
        self.cursor.execute("SELECT COUNT(*) FROM cvs")
        total_candidates = self.cursor.fetchone()[0]
        print(f"📊 Tổng số ứng viên trong database: {total_candidates}")
        return total_candidates

    def print_top_candidates(self, limit=5):
        """Hàm in ra 5 ứng viên đầu tiên trong database"""
        self.cursor.execute("SELECT id, file_name FROM cvs LIMIT ?", (limit,))
        candidates = self.cursor.fetchall()
        
        if not candidates:
            print("⚠️ Database trống, không có ứng viên nào để hiển thị!")
            return
        
        print(f"\n📋 **Danh sách {min(limit, len(candidates))} ứng viên trong database:**")
        for candidate in candidates:
            print(f"🆔 **ID:** {candidate[0]} | 📝 **File:** {candidate[1]}")

    def run(self, job_description, top_candidate):
        self.count_candidates()
        self.print_top_candidates()  # Gọi hàm mới để in 5 ứng viên
        cvs, similarities = self.compare_job_description(job_description, top_candidate)
        self.visualize_best_candidate(cvs, similarities, top_candidate)
        self.conn.close()

# ================================
# **🏁 CHẠY CHƯƠNG TRÌNH**
# ================================
if __name__ == '__main__':
    # Đường dẫn dựa trên cwd
    db_path = os.path.join(os.getcwd(), 'Dataset4DBNet', 'CV_parser_database.db')
    image_folder = os.path.join(os.getcwd(), 'Dataset4DBNet', 'Extract')
    parser = CVParser(db_path=db_path, image_folder=image_folder)
    
    job_desc = """
    SKILLS
    Staff Training, schedule, Experience with Medical Records, General Knowledge Of Computer
    Software, On Time And Reliable, Weekend Availability, Works Well As Part OF A Team Or
    Individually, Excellent Multi-tasker, Rapid Order Processing, Conflict Resolution Techniques,
    Results-oriented, Marketing, And Advertising,
    """
    top_candidate = 5
    
    parser.run(job_desc, top_candidate)