import sys
import os
from PyQt6.QtWidgets import (QApplication, QMainWindow, QPushButton, QWidget, 
                            QVBoxLayout, QHBoxLayout, QTextEdit, QFileDialog, 
                            QProgressBar, QLabel, QMessageBox, QInputDialog)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QScreen

from Pipline.utils4zero2hero import CVPipeline

# Biến kích thước tùy chỉnh (width, height)
MAIN_WINDOW_SIZE = (400, 300)  # Kích thước cho MainWindow
WINDOW2_SIZE = (500, 400)     # Kích thước cho Window2
WINDOW3_SIZE = (600, 500)     # Kích thước cho Window3
RESULT_WINDOW_SIZE = (400, 300)  # Kích thước cho ResultWindow

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CV Parser")
        self.setFixedSize(*MAIN_WINDOW_SIZE)  # Sử dụng biến kích thước
        
        # Cố định ở giữa màn hình
        self.center_on_screen()
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        self.embedding_btn = QPushButton("Extract to Database")
        self.predict_btn = QPushButton("Search Candidates")
        
        layout.addWidget(self.embedding_btn)
        layout.addWidget(self.predict_btn)
        
        self.embedding_btn.clicked.connect(self.show_window2)
        self.predict_btn.clicked.connect(self.show_window3)
        
        self.window2 = None
        self.window3 = None
        
    def center_on_screen(self):
        screen = QApplication.primaryScreen().geometry()
        screen_width = screen.width()
        screen_height = screen.height()
        window_width = self.width()
        window_height = self.height()
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        self.move(x, y)
        
    def show_window2(self):
        if self.window2 is None:
            self.window2 = Window2()
        self.window2.show()
        self.hide()
        
    def show_window3(self):
        if self.window3 is None:
            self.window3 = Window3()
        self.window3.show()
        self.hide()

class Window2(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Extract CVs to Database")
        self.setFixedSize(*WINDOW2_SIZE)  # Sử dụng biến kích thước
        
        # Cố định ở giữa màn hình
        self.center_on_screen()
        
        main_layout = QVBoxLayout(self)
        
        self.upload_btn = QPushButton("Choose Image Folder")
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.status_label = QLabel("Status: Waiting for folder selection")
        
        bottom_layout = QHBoxLayout()
        self.back_btn = QPushButton("Back")
        self.continue_btn = QPushButton("Continue to Search")
        
        bottom_layout.addWidget(self.back_btn)
        bottom_layout.addStretch()
        bottom_layout.addWidget(self.continue_btn)
        
        main_layout.addWidget(self.upload_btn)
        main_layout.addWidget(self.progress_bar)
        main_layout.addWidget(self.status_label)
        main_layout.addStretch()
        # Xóa custom_layout vì chưa được định nghĩa
        main_layout.addLayout(bottom_layout)  # Nếu không cần, xóa dòng này
        
        self.upload_btn.clicked.connect(self.upload_folder_and_extract)
        self.continue_btn.clicked.connect(self.show_window3)
        self.back_btn.clicked.connect(self.back_to_main)
        
        self.main_window = None
        self.window3 = None
        self.db_path = None
        
    def center_on_screen(self):
        screen = QApplication.primaryScreen().geometry()
        screen_width = screen.width()
        screen_height = screen.height()
        window_width = self.width()
        window_height = self.height()
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        self.move(x, y)
        
    def upload_folder_and_extract(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Image Folder")
        if folder:
            db_name, ok1 = QInputDialog.getText(self, "Database Name", "Enter database name (e.g., cv_database):")
            if ok1 and db_name:
                db_folder = QFileDialog.getExistingDirectory(self, "Select Database Save Location")
                if db_folder:
                    self.db_path = os.path.join(db_folder, f"{db_name}.db")
                    self.status_label.setText(f"Status: Extracting CVs to {self.db_path}")
                    
                    self.progress_bar.setValue(0)
                    self.timer = QTimer()
                    self.timer.timeout.connect(self.update_progress)
                    self.timer.start(100)
                    
                    # Lấy tên thư mục ảnh và thư mục cha
                    folder_name = os.path.basename(folder)  # e.g., "resumes_img"
                    parent_dir = os.path.dirname(folder)    # e.g., "...\1_DEMO_only"
                    
                    # Tạo đường dẫn cho text_output_dir và summary_output_dir
                    text_output_base = os.path.join(parent_dir, f"{folder_name}2textoutput")
                    summary_output_base = os.path.join(parent_dir, f"{folder_name}2textoutput")
                    
                    text_output_dir = os.path.join(text_output_base, "text_extracted")
                    summary_output_dir = os.path.join(summary_output_base, "summaries")
                    
                    # Tạo các thư mục nếu chưa tồn tại
                    os.makedirs(text_output_dir, exist_ok=True)
                    os.makedirs(summary_output_dir, exist_ok=True)
                    
                    pipeline = CVPipeline(
                        image_input_dir=folder,
                        text_output_dir=text_output_dir,
                        summary_output_dir=summary_output_dir,
                        db_path=self.db_path,
                        model_path=os.path.join(os.getcwd(), "db_19_2.6895_3.3356.h5")
                    )
                    pipeline.run("", 0)
                    
    def update_progress(self):
        current_value = self.progress_bar.value()
        if current_value < 100:
            self.progress_bar.setValue(current_value + 1)
        else:
            self.timer.stop()
            self.status_label.setText(f"Extraction completed. Saved to {self.db_path}")
            QMessageBox.information(self, "Success", f"Data extracted successfully to {self.db_path}")
        
    def show_window3(self):
        if self.window3 is None:
            self.window3 = Window3(self.db_path)
        else:
            self.window3.set_db_path(self.db_path)
        self.window3.show()
        self.hide()
        
    def back_to_main(self):
        self.main_window = MainWindow()
        self.main_window.show()
        self.hide()

class Window3(QWidget):
    def __init__(self, initial_db_path=None):
        super().__init__()
        self.setWindowTitle("Search Candidates")
        self.setFixedSize(*WINDOW3_SIZE)  # Sử dụng biến kích thước
        
        # Cố định ở giữa màn hình
        self.center_on_screen()
        
        main_layout = QVBoxLayout(self)
        
        self.db_label = QLabel("Selected Database: None")
        self.choose_db_btn = QPushButton("Choose Database")
        
        self.text_edit1 = QTextEdit()
        self.text_edit1.setPlaceholderText("Job description (select database first)")
        self.text_edit1.setEnabled(False)
        self.text_edit2 = QTextEdit()
        self.text_edit2.setPlaceholderText("Number of candidates to visualize")
        
        self.show_btn = QPushButton("Show")
        self.back_btn = QPushButton("Back")
        
        main_layout.addWidget(self.choose_db_btn)
        main_layout.addWidget(self.db_label)
        main_layout.addWidget(self.text_edit1)
        main_layout.addWidget(self.text_edit2)
        main_layout.addWidget(self.show_btn)
        main_layout.addStretch()
        main_layout.addWidget(self.back_btn, alignment=Qt.AlignmentFlag.AlignLeft)
        
        self.choose_db_btn.clicked.connect(self.choose_database)
        self.show_btn.clicked.connect(self.show_result)
        self.back_btn.clicked.connect(self.back_to_main)
        
        self.main_window = None
        self.result_window = None
        self.db_path = initial_db_path
        if self.db_path:
            self.db_label.setText(f"Selected Database: {os.path.basename(self.db_path)}")
            self.text_edit1.setEnabled(True)
        
    def center_on_screen(self):
        screen = QApplication.primaryScreen().geometry()
        screen_width = screen.width()
        screen_height = screen.height()
        window_width = self.width()
        window_height = self.height()
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        self.move(x, y)
        
    def choose_database(self):
        file, _ = QFileDialog.getOpenFileName(self, "Choose Database File", "", "Database Files (*.db)")
        if file:
            self.db_path = file
            self.db_label.setText(f"Selected Database: {os.path.basename(self.db_path)}")
            self.text_edit1.setEnabled(True)
        
    def set_db_path(self, db_path):
        self.db_path = db_path
        self.db_label.setText(f"Selected Database: {os.path.basename(self.db_path)}")
        self.text_edit1.setEnabled(True)
        
    def show_result(self):
        if not self.db_path:
            QMessageBox.warning(self, "Error", "Please select a database first!")
            return
        
        job_desc = self.text_edit1.toPlainText()
        try:
            top_candidates = int(self.text_edit2.toPlainText())
        except ValueError:
            QMessageBox.warning(self, "Error", "Please enter a valid number of candidates!")
            return
        
        if not job_desc:
            QMessageBox.warning(self, "Error", "Please enter a job description!")
            return
        
        pipeline = CVPipeline(
            image_input_dir="",  
            text_output_dir=os.path.join(os.getcwd(), "data_extracted_output", "text_extracted"),
            summary_output_dir=os.path.join(os.getcwd(), "data_extracted_output", "summaries"),
            db_path=self.db_path,
            model_path=os.path.join(os.getcwd(), "db_19_2.6895_3.3356.h5")
        )
        pipeline.visualize_candidates(job_description=job_desc, top_candidate=top_candidates)
        
    def back_to_main(self):
        self.main_window = MainWindow()
        self.main_window.show()
        self.hide()

class ResultWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Result")
        self.setFixedSize(*RESULT_WINDOW_SIZE)  # Sử dụng biến kích thước
        
        # Cố định ở giữa màn hình
        self.center_on_screen()
        
        layout = QVBoxLayout(self)
        label = QLabel("Results will be displayed here")
        layout.addWidget(label)
    
    def center_on_screen(self):
        screen = QApplication.primaryScreen().geometry()
        screen_width = screen.width()
        screen_height = screen.height()
        window_width = self.width()
        window_height = self.height()
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        self.move(x, y)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())