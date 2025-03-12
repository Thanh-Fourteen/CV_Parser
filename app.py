import sys
from PyQt6.QtWidgets import (QApplication, QMainWindow, QPushButton, QWidget, 
                            QVBoxLayout, QHBoxLayout, QTextEdit, QFileDialog)
from PyQt6.QtCore import Qt

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Window 1")
        self.setFixedSize(200, 200)
        
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Create buttons
        self.embedding_btn = QPushButton("Embedding")
        self.predict_btn = QPushButton("Predict")
        
        # Add buttons to layout
        layout.addWidget(self.embedding_btn)
        layout.addWidget(self.predict_btn)
        
        # Connect buttons to functions
        self.embedding_btn.clicked.connect(self.show_window2)
        self.predict_btn.clicked.connect(self.show_window3)
        
        self.window2 = None
        self.window3 = None
        
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
        self.setWindowTitle("Window 2")
        self.setFixedSize(300, 200)
        
        # Create main layout
        main_layout = QVBoxLayout(self)
        
        # Create upload button
        self.upload_btn = QPushButton("Upload Folder")
        main_layout.addWidget(self.upload_btn)
        
        # Create bottom layout for Next and Back buttons
        bottom_layout = QHBoxLayout()
        self.back_btn = QPushButton("Back")
        self.next_btn = QPushButton("Next")
        
        bottom_layout.addWidget(self.back_btn)
        bottom_layout.addStretch()
        bottom_layout.addWidget(self.next_btn)
        
        main_layout.addStretch()
        main_layout.addLayout(bottom_layout)
        
        # Connect buttons
        self.upload_btn.clicked.connect(self.upload_folder)
        self.next_btn.clicked.connect(self.show_window3)
        self.back_btn.clicked.connect(self.back_to_main)
        
        self.main_window = None
        self.window3 = None
        
    def upload_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder:
            print(f"Selected folder: {folder}")
            
    def show_window3(self):
        if self.window3 is None:
            self.window3 = Window3()
        self.window3.show()
        self.hide()
        
    def back_to_main(self):
        self.main_window = MainWindow()
        self.main_window.show()
        self.hide()

class Window3(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Window 3")
        self.setFixedSize(400, 300)
        
        # Create main layout
        main_layout = QVBoxLayout(self)
        
        # Create buttons and text edits
        self.choose_btn = QPushButton("Choose Embedding")
        self.text_edit1 = QTextEdit()
        self.text_edit1.setPlaceholderText("Enter text here")
        self.text_edit2 = QTextEdit()
        self.text_edit2.setPlaceholderText("Enter number here")
        self.show_btn = QPushButton("Show")
        self.back_btn = QPushButton("Back")
        
        # Add widgets to layout
        main_layout.addWidget(self.choose_btn)
        main_layout.addWidget(self.text_edit1)
        main_layout.addWidget(self.text_edit2)
        main_layout.addWidget(self.show_btn)
        main_layout.addStretch()
        main_layout.addWidget(self.back_btn, alignment=Qt.AlignmentFlag.AlignLeft)
        
        # Connect buttons
        self.choose_btn.clicked.connect(self.choose_embedding)
        self.show_btn.clicked.connect(self.show_result)
        self.back_btn.clicked.connect(self.back_to_main)
        
        self.main_window = None
        self.result_window = None
        
    def choose_embedding(self):
        file, _ = QFileDialog.getOpenFileName(self, "Choose Embedding File")
        if file:
            print(f"Selected embedding file: {file}")
            
    def show_result(self):
        if self.result_window is None:
            self.result_window = ResultWindow()
        self.result_window.show()
        
    def back_to_main(self):
        self.main_window = MainWindow()
        self.main_window.show()
        self.hide()

class ResultWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Result")
        self.setFixedSize(200, 200)
        # Add your result display logic here
        
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())