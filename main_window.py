import sys
import os
import cv2
import numpy as np
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QPushButton, QFileDialog, 
                             QSlider, QFrame, QProgressBar, QMessageBox)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt6.QtGui import QPixmap, QImage, QIcon, QFont

# Add root to path to import src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.inference import FireRiskSystem

class InferenceThread(QThread):
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, image_path):
        super().__init__()
        self.image_path = image_path
        self.system = None

    def run(self):
        try:
            if self.system is None:
                self.system = FireRiskSystem(device='cpu') # Force CPU for compatibility in GUI often
            
            result = self.system.run_inference(self.image_path)
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SmartFire - Advanced Risk Assessment")
        self.resize(1200, 800)
        
        # Style
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1e1e2e;
            }
            QLabel {
                color: #cdd6f4;
                font-family: 'Segoe UI', sans-serif;
            }
            QPushButton {
                background-color: #89b4fa;
                color: #1e1e2e;
                border-radius: 8px;
                padding: 10px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #b4befe;
            }
            QFrame {
                background-color: #313244;
                border-radius: 12px;
            }
            QProgressBar {
                border: 2px solid #45475a;
                border-radius: 5px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #a6e3a1;
            }
        """)

        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        self.layout = QHBoxLayout(self.main_widget)
        self.layout.setContentsMargins(20, 20, 20, 20)

        # -- Left Panel (Controls) --
        self.left_panel = QFrame()
        self.left_panel.setFrameShape(QFrame.Shape.StyledPanel)
        self.left_layout = QVBoxLayout(self.left_panel)
        
        # Logo/Title
        title = QLabel("SmartFire AI")
        title.setFont(QFont("Arial", 24, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.left_layout.addWidget(title)
        
        # Buttons
        self.btn_upload = QPushButton("📂 Upload Image")
        self.btn_upload.clicked.connect(self.upload_image)
        self.left_layout.addWidget(self.btn_upload)
        
        self.btn_run = QPushButton("🚀 Run Analysis")
        self.btn_run.clicked.connect(self.run_analysis)
        self.btn_run.setEnabled(False)
        self.left_layout.addWidget(self.btn_run)
        
        # Info Box
        self.info_label = QLabel("No image loaded.")
        self.info_label.setWordWrap(True)
        self.info_label.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.left_layout.addWidget(self.info_label)
        
        self.left_layout.addStretch()
        
        # Progress
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        self.left_layout.addWidget(self.progress)

        self.layout.addWidget(self.left_panel, 1)

        # -- Right Panel (Display) --
        self.right_panel = QFrame()
        self.right_layout = QVBoxLayout(self.right_panel)
        
        # Image Display Area
        self.image_label = QLabel("Visual Output")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("border: 2px dashed #45475a; color: #585b70;")
        self.right_layout.addWidget(self.image_label, 3)
        
        # Bottom Results (Horizontal)
        self.res_layout = QHBoxLayout()
        
        self.mask_label = QLabel("Mask")
        self.mask_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.mask_label.setStyleSheet("background-color: #181825; border-radius: 8px;")
        
        self.cam_label = QLabel("Grad-CAM")
        self.cam_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.cam_label.setStyleSheet("background-color: #181825; border-radius: 8px;")
        
        self.res_layout.addWidget(self.mask_label)
        self.res_layout.addWidget(self.cam_label)
        
        self.right_layout.addLayout(self.res_layout, 1)
        
        self.layout.addWidget(self.right_panel, 3)

        # State
        self.current_image_path = None
        self.inference_thread = None

    def upload_image(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.jpg *.png *.bmp)")
        if fname:
            self.current_image_path = fname
            self.show_image(fname, self.image_label)
            self.btn_run.setEnabled(True)
            self.info_label.setText(f"Loaded: {os.path.basename(fname)}\nReady to analyze.")
            
    def show_image(self, path_or_arr, label_widget):
        if isinstance(path_or_arr, str):
            pixmap = QPixmap(path_or_arr)
        else:
            # Numpy array to pixmap
            # Convert BGR to RGB if needed, usually openCV is BGR
            if len(path_or_arr.shape) == 3:
                h, w, ch = path_or_arr.shape
                bytes_per_line = ch * w
                # assume BGR from opencv
                rgb = cv2.cvtColor(path_or_arr, cv2.COLOR_BGR2RGB)
                qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                pixmap = QPixmap.fromImage(qimg)
            else:
                pass # grayscale unsupported for now
                
        # Scale to fit
        scaled = pixmap.scaled(label_widget.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        label_widget.setPixmap(scaled)

    def run_analysis(self):
        self.btn_run.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.setRange(0, 0) # Infinite loading
        self.info_label.setText("Analyzing... Please wait.")
        
        self.inference_thread = InferenceThread(self.current_image_path)
        self.inference_thread.finished.connect(self.on_analysis_done)
        self.inference_thread.error.connect(self.on_error)
        self.inference_thread.start()

    def on_analysis_done(self, result):
        self.progress.setVisible(False)
        self.progress.setRange(0, 100)
        self.btn_run.setEnabled(True)
        
        # Display Results
        self.show_image(result['processed_image'], self.image_label)
        
        # Show GradCAM
        self.show_image(result['grad_cam'], self.cam_label)
        
        # Show Color Mask (need to colorize mask, but run_inference returns mask index map)
        # Actually inference returns 'mask' as logic map. Let's make a pretty mask viz.
        mask = result['mask']
        mask_viz = np.zeros((*mask.shape, 3), dtype=np.uint8)
        mask_viz[mask == 1] = [0, 0, 255] # Red
        mask_viz[mask == 2] = [200, 200, 200] # smoke
        self.show_image(mask_viz, self.mask_label)
        
        # Update Text
        sev = result['severity']
        txt = (f"<h2>Risk Level: <span style='color: {'red' if sev['label'] in ['High', 'Critical'] else 'orange'}'>{sev['label']}</span></h2>"
               f"Score: {sev['score']}/100<br>"
               f"Fire Ratio: {sev['details']['fire_ratio']:.4f}<br>"
               f"Latency: {result['latency']:.2f}s")
        self.info_label.setText(txt)

    def on_error(self, err):
        self.progress.setVisible(False)
        self.btn_run.setEnabled(True)
        QMessageBox.critical(self, "Error", f"Analysis failed:\n{err}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
