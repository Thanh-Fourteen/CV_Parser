import sys
from PyQt6.QtWidgets import (QApplication, QMainWindow, QPushButton, QTextEdit, 
                            QFileDialog, QVBoxLayout, QWidget, QHBoxLayout)

class Window1(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Window 1")
        self.setFixedSize(300, 400)
        
        # Tạo layout chính
        layout = QVBoxLayout()
        
        # Tạo buttons
        self.embedding_btn = QPushButton("Embedding")
        self.predict_btn = QPushButton("Predict")
        
        # Thêm buttons vào layout
        layout.addWidget(self.embedding_btn)
        layout.addWidget(self.predict_btn)
        layout.addStretch()  # Đẩy buttons lên trên
        
        # Tạo widget trung tâm
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)
        
        # Kết nối signals
        self.embedding_btn.clicked.connect(self.show_window2)
        self.predict_btn.clicked.connect(self.show_window3)
        
    def show_window2(self):
        self.window2 = Window2()
        self.window2.show()
        self.hide()
        
    def show_window3(self):
        self.window3 = Window3()
        self.window3.show()
        self.hide()

class Window2(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Window 2")
        self.setFixedSize(300, 400)
        
        # Tạo layout chính
        layout = QVBoxLayout()
        bottom_layout = QHBoxLayout()
        
        # Tạo buttons
        self.upload_btn = QPushButton("Upload Folder")
        self.next_btn = QPushButton("Next")
        
        # Thêm buttons vào layout
        layout.addWidget(self.upload_btn)
        layout.addStretch()  # Đẩy Upload lên trên
        bottom_layout.addStretch()  # Đẩy Next sang phải
        bottom_layout.addWidget(self.next_btn)
        layout.addLayout(bottom_layout)
        
        # Tạo widget trung tâm
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)
        
        # Kết nối signals
        self.upload_btn.clicked.connect(self.upload_folder)
        self.next_btn.clicked.connect(self.show_window3)
        
    def upload_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder:
            print(f"Selected folder: {folder}")
            
    def show_window3(self):
        self.window3 = Window3()
        self.window3.show()
        self.hide()

class Window3(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Window 3")
        self.setFixedSize(300, 400)
        
        # Tạo layout chính
        layout = QVBoxLayout()
        
        # Tạo widgets
        self.choose_btn = QPushButton("Choose Embedding")
        self.text_edit = QTextEdit()
        self.text_edit.setFixedHeight(50)
        self.show_btn = QPushButton("Show")
        self.back_btn = QPushButton("Back")
        
        # Thêm widgets vào layout
        layout.addWidget(self.choose_btn)
        layout.addWidget(self.text_edit)
        layout.addWidget(self.show_btn)
        layout.addStretch()  # Đẩy Back xuống dưới
        layout.addWidget(self.back_btn)
        
        # Tạo widget trung tâm
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)
        
        # Kết nối signals
        self.choose_btn.clicked.connect(self.choose_embedding)
        self.show_btn.clicked.connect(self.show_result)
        self.back_btn.clicked.connect(self.go_back)
        
    def choose_embedding(self):
        file, _ = QFileDialog.getOpenFileName(self, "Choose Embedding File")
        if file:
            print(f"Selected embedding file: {file}")
            
    def show_result(self):
        number = self.text_edit.toPlainText()
        if number:
            # Tạo cửa sổ mới để hiển thị kết quả
            self.result_window = QMainWindow()
            self.result_window.setWindowTitle("Result")
            self.result_window.setFixedSize(200, 200)
            self.result_window.show()
            print(f"Number entered: {number}")
            
    def go_back(self):
        self.window1 = Window1()
        self.window1.show()
        self.hide()

def main():
    app = QApplication(sys.argv)
    window = Window1()
    window.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()