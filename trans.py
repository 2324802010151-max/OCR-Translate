import sys
import os
import re
import traceback
import tempfile
import torch
import keyboard  # pip install keyboard
from PIL import ImageGrab

# --- 1. CẤU HÌNH FIX LỖI DLL TORCH (Windows) ---
path_to_torch_dlls = r"C:\Users\admin\AppData\Local\Programs\Python\Python311\Lib\site-packages\torch\lib"
if os.path.exists(path_to_torch_dlls):
    os.add_dll_directory(path_to_torch_dlls)

from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QLabel, 
                             QVBoxLayout, QWidget, QFrame)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QRect, QPoint
from PyQt5.QtGui import QPainter, QColor, QPen

# --- IMPORT MODELS ---
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModel

# --- QUẢN LÝ MODEL (SINGLETON PATTERN) ---
class ModelManager:
    _instance = None
    
    def __init__(self):
        self.ocr_model = None
        self.ocr_tokenizer = None
        self.trans_model = None
        self.trans_tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def load_ocr(self):
        if self.ocr_model is None:
            model_name = 'ucaslcl/GOT-OCR2_0'
            print(">>> Loading OCR Model...")
            self.ocr_tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            self.ocr_model = AutoModel.from_pretrained(
                model_name, 
                trust_remote_code=True, 
                low_cpu_mem_usage=True, 
                device_map=self.device, 
                use_safetensors=True, 
                pad_token_id=self.ocr_tokenizer.eos_token_id
            ).eval()
            if self.device == "cuda":
                self.ocr_model = self.ocr_model.cuda() # Đảm bảo nằm trên GPU

    def load_trans(self):
        if self.trans_model is None:
            model_name = "facebook/nllb-200-distilled-600M"
            print(">>> Loading Trans Model...")
            self.trans_tokenizer = AutoTokenizer.from_pretrained(model_name)
            # Dùng float16 để nhẹ và nhanh hơn
            self.trans_model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name, 
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            ).to(self.device).eval()

# --- WORKER XỬ LÝ NẶNG ---
class WorkerThread(QThread):
    status_update = pyqtSignal(str)
    result_ready = pyqtSignal(str)

    def __init__(self, image_path=None, mode="scan"):
        super().__init__()
        self.image_path = image_path
        self.mode = mode
        self.manager = ModelManager.get_instance()

    def clean_text(self, text):
        """Làm sạch text thông minh hơn"""
        text = text.replace("\n", " ")
        text = re.sub(r"\s+", " ", text).strip()
        # Nối từ bị đứt (VD: "exam- ple" -> "example")
        text = re.sub(r"([a-z])-\s+([a-z])", r"\1\2", text)
        # Fix lỗi khoảng trắng trước dấu câu
        text = re.sub(r"\s+([.,!?;:])", r"\1", text)
        return text

    def run(self):
        try:
            # 1. Nạp Models (Nếu chưa nạp)
            if self.mode == "preload":
                self.status_update.emit("🚀 Đang nạp OCR...")
                self.manager.load_ocr()
                self.status_update.emit("🚀 Đang nạp Trans...")
                self.manager.load_trans()
                self.status_update.emit(f"✅ Sẵn sàng! (GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'OFF'}). Alt+X để chụp.")
                return

            # 2. Xử lý OCR
            if not self.image_path: return
            self.status_update.emit("🔍 Đang đọc chữ (OCR)...")
            
            # Đảm bảo model đã load
            self.manager.load_ocr()
            
            res = self.manager.ocr_model.chat(self.manager.ocr_tokenizer, self.image_path, ocr_type='ocr')
            raw_text = str(res)
            clean_text_str = self.clean_text(raw_text)

            if not clean_text_str:
                self.result_ready.emit("⚠️ Không tìm thấy chữ trong ảnh!")
                return

            # 3. Dịch Thuật (Batch Processing)
            self.status_update.emit("🌐 Đang dịch...")
            self.manager.load_trans()
            
            # Tách câu
            sentences = re.split(r'(?<=[.!?])\s+', clean_text_str)
            sentences = [s for s in sentences if len(s.strip()) > 1]
            
            final_vn = ""
            if sentences:
                tokenizer = self.manager.trans_tokenizer
                model = self.manager.trans_model
                device = self.manager.device
                tgt_lang = "vie_Latn"

                # Batch hóa dữ liệu đầu vào (Gửi 1 lần nhiều câu)
                inputs = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True).to(device)
                
                with torch.no_grad():
                    translated_tokens = model.generate(
                        **inputs, 
                        forced_bos_token_id=tokenizer.lang_code_to_id[tgt_lang], 
                        max_length=512
                    )
                
                # Decode kết quả
                trans_texts = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
                final_vn = " ".join(trans_texts)

            # 4. Trả kết quả
            final_output = f"🇬🇧 <b>GỐC:</b><br>{clean_text_str}<br><br>🇻🇳 <b>DỊCH:</b><br>{final_vn}"
            self.result_ready.emit(final_output)

        except Exception as e:
            traceback.print_exc()
            self.result_ready.emit(f"❌ Lỗi: {str(e)}")
        finally:
            # Xóa file ảnh tạm
            if self.image_path and os.path.exists(self.image_path):
                try:
                    os.remove(self.image_path)
                except: pass

# --- SNIPPING TOOL ---
class SnippingWidget(QWidget):
    snippet_taken = pyqtSignal(str) 

    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setCursor(Qt.CrossCursor)
        self.start_point = None
        self.end_point = None
        self.is_sniping = False

    def start_selection(self):
        # Reset mỗi khi bắt đầu phiên làm việc mới cho chắc chắn
        self.start_point = None
        self.end_point = None
        self.setGeometry(QApplication.primaryScreen().geometry())
        self.show()
        self.activateWindow()

    def paintEvent(self, event):
        if not self.isVisible(): return
        
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Vẽ lớp phủ mờ toàn màn hình
        painter.setBrush(QColor(0, 0, 0, 100))
        painter.setPen(Qt.NoPen)
        painter.drawRect(self.rect())

        # CHỈ vẽ khung đỏ nếu cả 2 điểm đều đã tồn tại
        if self.start_point and self.end_point:
            rect = QRect(self.start_point, self.end_point).normalized()
            painter.setCompositionMode(QPainter.CompositionMode_Clear)
            painter.drawRect(rect)
            painter.setCompositionMode(QPainter.CompositionMode_SourceOver)
            painter.setPen(QPen(QColor(0, 255, 204), 2))
            painter.drawRect(rect)

    def mousePressEvent(self, event):
        self.start_point = event.pos()
        self.end_point = event.pos()
        self.is_sniping = True
        self.update()

    def mouseMoveEvent(self, event):
        if self.is_sniping:
            self.end_point = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if not self.is_sniping: return
        self.is_sniping = False
        
        rect = QRect(self.start_point, event.pos()).normalized()
        
        # Lấy tọa độ xong thì ẩn ngay và xóa điểm
        self.hide()
        
        # XỬ LÝ CHỤP ẢNH
        if rect.width() > 10 and rect.height() > 10:
            # Thực hiện chụp ảnh ở đây (giống code cũ của bạn)
            path = self.capture_screen(rect)
            self.snippet_taken.emit(path)
        
        # QUAN TRỌNG: Reset về None ngay lập tức sau khi hoàn tất
        self.start_point = None
        self.end_point = None
        self.update() # Vẽ lại một lần cuối để xóa khung

    def capture_screen(self, rect):
        # Hàm phụ để xử lý lưu ảnh
        x, y, w, h = rect.x(), rect.y(), rect.width(), rect.height()
        fd, path = tempfile.mkstemp(suffix=".jpg")
        os.close(fd)
        img = ImageGrab.grab(bbox=(x, y, x+w, y+h))
        img.save(path, quality=95)
        return path

# --- GIAO DIỆN CHÍNH ---
class ResultWindow(QMainWindow):
    request_snip_signal = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.resize(500, 400)
        self.old_pos = None

        # Widget chính
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)
        self.layout.setContentsMargins(0, 0, 0, 0)
        
        # Frame chứa nội dung
        self.frame = QFrame()
        self.frame.setStyleSheet("""
            QFrame {
                background-color: rgba(28, 28, 30, 0.95);
                border: 1px solid #333;
                border-radius: 12px;
                color: #E0E0E0;
            }
        """)
        self.layout.addWidget(self.frame)
        self.frame_layout = QVBoxLayout(self.frame)

        # Header
        header_layout = QVBoxLayout()
        self.lbl_title = QLabel("AI TRANSLATOR (Alt + X)")
        self.lbl_title.setStyleSheet("font-weight: bold; color: #00FFCC; font-size: 14px; border: none;")
        header_layout.addWidget(self.lbl_title)
        self.frame_layout.addLayout(header_layout)

        # Nội dung Text
        self.lbl_result = QLabel("Đang khởi động AI...")
        self.lbl_result.setWordWrap(True)
        self.lbl_result.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.lbl_result.setStyleSheet("border: none; padding: 5px; font-size: 13px; color: white;")
        self.lbl_result.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.frame_layout.addWidget(self.lbl_result)
        self.frame_layout.addStretch()

        # Nút Đóng
        self.btn_minimize = QPushButton("Thu nhỏ")
        self.btn_minimize.setCursor(Qt.PointingHandCursor)
        self.btn_minimize.clicked.connect(self.showMinimized)
        self.btn_minimize.setStyleSheet("""
            QPushButton {
                background: #3A3A3C; color: white; border-radius: 6px; padding: 8px;
                border: 1px solid #48484A;
            }
            QPushButton:hover { background: #48484A; }
        """)
        self.frame_layout.addWidget(self.btn_minimize)

        # Logic Snipping & Worker
        self.snipper = SnippingWidget()
        self.snipper.snippet_taken.connect(self.process_image)

        self.preload_worker = WorkerThread(mode="preload")
        self.preload_worker.status_update.connect(self.update_text)
        self.preload_worker.start()

        self.request_snip_signal.connect(self.start_snipping)
        try:
            keyboard.add_hotkey('alt+x', self.request_snip_signal.emit)
        except ImportError:
            print("Cần chạy quyền Admin để dùng hotkey")

    def start_snipping(self):
        self.hide()
        self.snipper.start_selection()

    def process_image(self, img_path):
        self.showNormal()
        self.activateWindow()
        self.update_text("⏳ Đang xử lý...")
        
        self.worker = WorkerThread(image_path=img_path, mode="scan")
        self.worker.status_update.connect(self.update_text)
        self.worker.result_ready.connect(self.update_html_text) # Dùng HTML để format đẹp hơn
        self.worker.start()

    def update_text(self, text):
        self.lbl_result.setText(text)

    def update_html_text(self, text):
        self.lbl_result.setText(text)

    # --- DRAGGABLE WINDOW ---
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.old_pos = event.globalPos()

    def mouseMoveEvent(self, event):
        if self.old_pos: 
            delta = event.globalPos() - self.old_pos
            self.move(self.pos() + delta)
            self.old_pos = event.globalPos()

    def mouseReleaseEvent(self, event):
        self.old_pos = None

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ResultWindow()
    window.show()
    sys.exit(app.exec_())
