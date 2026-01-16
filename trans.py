import sys
import os
import re
import time
import traceback
import torch
import keyboard  # pip install keyboard
from PIL import ImageGrab

# --- 1. CẤU HÌNH FIX LỖI DLL TORCH ---
path_to_torch_dlls = r"C:\Users\admin\AppData\Local\Programs\Python\Python311\Lib\site-packages\torch\lib"
if os.path.exists(path_to_torch_dlls):
    os.add_dll_directory(path_to_torch_dlls)

from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QLabel, 
                             QVBoxLayout, QWidget, QFrame, QHBoxLayout)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QRect, QPoint
from PyQt5.QtGui import QCursor, QPainter, QColor, QPen, QBrush

# --- IMPORT MODELS ---
# pip install sentencepiece protobuf transformers
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModel

# --- BIẾN TOÀN CỤC ---
GLOBAL_OCR_MODEL = None
GLOBAL_OCR_TOKENIZER = None
GLOBAL_TRANS_MODEL = None
GLOBAL_TRANS_TOKENIZER = None

class GOTOCRWorker(QThread):
    result_ready = pyqtSignal(str)
    model_loaded_signal = pyqtSignal()

    def __init__(self, image_path=None, mode="scan"):
        super().__init__()
        self.image_path = image_path
        self.mode = mode

    def clean_text(self, text):
        """Hàm làm sạch rác OCR để dịch chuẩn hơn"""
        # 1. Nối dấu nháy: "school ' s" -> "school's"
        text = re.sub(r"\s+(['’])\s*([a-zA-Z])", r"'\2", text)
        # 2. Nối dấu câu: "Hello ." -> "Hello."
        text = re.sub(r"\s+([.,!?;:])", r"\1", text)
        # 3. Mẹo nối từ bị đứt (VD: "handsomes t" -> "handsomest")
        text = re.sub(r"([a-zA-Z]{3,})\s+([a-zA-Z])\b", r"\1\2", text)
        # 4. Xóa xuống dòng thừa
        text = text.replace("\n", " ")
        # 5. Xóa khoảng trắng kép
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def run(self):
        global GLOBAL_OCR_MODEL, GLOBAL_OCR_TOKENIZER, GLOBAL_TRANS_MODEL, GLOBAL_TRANS_TOKENIZER
        try:
            # --- 1. Nạp Model OCR ---
            if GLOBAL_OCR_MODEL is None:
                self.result_ready.emit("🚀 Đang nạp Model OCR (GOT-2.0)...")
                model_ocr_name = 'ucaslcl/GOT-OCR2_0'
                GLOBAL_OCR_TOKENIZER = AutoTokenizer.from_pretrained(model_ocr_name, trust_remote_code=True)
                GLOBAL_OCR_MODEL = AutoModel.from_pretrained(
                    model_ocr_name, trust_remote_code=True, low_cpu_mem_usage=True, 
                    device_map='cuda', use_safetensors=True, 
                    pad_token_id=GLOBAL_OCR_TOKENIZER.eos_token_id
                )
                GLOBAL_OCR_MODEL = GLOBAL_OCR_MODEL.eval().cuda()

            # --- 2. Nạp Model Dịch ---
            if GLOBAL_TRANS_MODEL is None:
                self.result_ready.emit("🚀 Đang nạp Model NLLB-200...")
                model_trans_name = "facebook/nllb-200-distilled-600M"
                GLOBAL_TRANS_TOKENIZER = AutoTokenizer.from_pretrained(model_trans_name)
                GLOBAL_TRANS_MODEL = AutoModelForSeq2SeqLM.from_pretrained(model_trans_name)
                device = "cuda" if torch.cuda.is_available() else "cpu"
                GLOBAL_TRANS_MODEL = GLOBAL_TRANS_MODEL.to(device)
                
                self.result_ready.emit(f"✅ Sẵn sàng! (GPU: {torch.cuda.is_available()}). Bấm Alt+X để chụp.")
                self.model_loaded_signal.emit()

            if self.mode == "preload": return

            # --- 3. Thực hiện OCR ---
            if not self.image_path: return
            self.result_ready.emit("⏳ Đang đọc chữ từ ảnh...")
            
            abs_image_path = os.path.abspath(self.image_path)
            res = GLOBAL_OCR_MODEL.chat(GLOBAL_OCR_TOKENIZER, abs_image_path, ocr_type='ocr')
            raw_text = str(res)
            
            # --- 4. Làm sạch & Dịch (Chia câu) ---
            self.result_ready.emit("⏳ Đang dịch...")
            clean_text_str = self.clean_text(raw_text)
            
            # Tách câu để dịch không bị sót
            sentences = re.split(r'([.!?]+)', clean_text_str)
            translated_parts = []
            
            # Ghép lại thành các câu hoàn chỉnh (Text + Dấu câu)
            full_sentences = []
            current_sent = ""
            for part in sentences:
                if re.match(r'[.!?]+', part):
                    current_sent += part
                    full_sentences.append(current_sent)
                    current_sent = ""
                else:
                    current_sent += part
            if current_sent: full_sentences.append(current_sent)

            # Dịch từng câu
            device = GLOBAL_TRANS_MODEL.device
            tgt_lang = "vie_Latn"
            
            for sent in full_sentences:
                if len(sent.strip()) < 2: continue
                
                inputs = GLOBAL_TRANS_TOKENIZER(sent, return_tensors="pt").to(device)
                translated_tokens = GLOBAL_TRANS_MODEL.generate(
                    **inputs, 
                    forced_bos_token_id=GLOBAL_TRANS_TOKENIZER.lang_code_to_id[tgt_lang], 
                    max_length=512
                )
                trans_text = GLOBAL_TRANS_TOKENIZER.batch_decode(translated_tokens, skip_special_tokens=True)[0]
                translated_parts.append(trans_text)

            final_vn = " ".join(translated_parts)
            final_output = f"🇬🇧 GỐC:\n{clean_text_str}\n\n🇻🇳 DỊCH:\n{final_vn}"
            self.result_ready.emit(final_output)

        except Exception as e:
            traceback.print_exc()
            self.result_ready.emit(f"Lỗi: {str(e)}")

# --- PHẦN SNIPPING TOOL (ĐÃ FIX LỖI VÙNG CHỌN CŨ) ---
class SnippingWidget(QWidget):
    snippet_taken = pyqtSignal(object) 

    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setCursor(Qt.CrossCursor)
        self.start_point = None
        self.end_point = None
        self.is_sniping = False

    def start_selection(self):
        # FIX: Reset tọa độ để không hiện lại khung đỏ cũ
        self.start_point = None
        self.end_point = None
        self.setGeometry(QApplication.primaryScreen().geometry())
        self.show()
        self.activateWindow()

    def paintEvent(self, event):
        if not self.isVisible(): return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setBrush(QColor(0, 0, 0, 100))
        painter.setPen(Qt.NoPen)
        painter.drawRect(self.rect())

        if self.start_point and self.end_point:
            rect = QRect(self.start_point, self.end_point).normalized()
            painter.setCompositionMode(QPainter.CompositionMode_Clear)
            painter.drawRect(rect)
            painter.setCompositionMode(QPainter.CompositionMode_SourceOver)
            painter.setPen(QPen(Qt.red, 2))
            painter.setBrush(Qt.NoBrush)
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
        self.hide() 
        if rect.width() > 10 and rect.height() > 10:
            x, y = rect.x(), rect.y()
            w, h = rect.width(), rect.height()
            try:
                img = ImageGrab.grab(bbox=(x, y, x+w, y+h))
                img.save("capture.jpg", quality=100)
                self.snippet_taken.emit("capture.jpg")
            except Exception as e:
                print(e)

# --- GIAO DIỆN CHÍNH (ĐÃ FIX LỖI NÚT BẤM) ---
class ResultWindow(QMainWindow):
    request_snip_signal = pyqtSignal()

    def __init__(self):
        super().__init__()
        # Cấu hình cửa sổ không viền, luôn nổi
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.resize(500, 400)
        self.old_pos = None

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)
        
        self.frame = QFrame()
        self.frame.setStyleSheet("""
            QFrame {
                background-color: rgba(20, 20, 20, 0.95);
                border: 2px solid #00ffcc;
                border-radius: 10px;
                color: white;
            }
        """)
        self.layout.addWidget(self.frame)
        self.frame_layout = QVBoxLayout(self.frame)

        # Tiêu đề
        self.lbl_title = QLabel("NLLB-200 TRANSLATOR (Alt + X)")
        self.lbl_title.setStyleSheet("font-weight: bold; color: #00ffcc; font-size: 14px; border: none;")
        self.frame_layout.addWidget(self.lbl_title)

        # Kết quả
        self.lbl_result = QLabel("Đang khởi động Model...")
        self.lbl_result.setWordWrap(True)
        self.lbl_result.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.lbl_result.setStyleSheet("border: none; padding: 5px; font-size: 13px;")
        self.lbl_result.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.frame_layout.addWidget(self.lbl_result)
        self.frame_layout.addStretch()

        # Nút Thu nhỏ (FIX LỖI)
        self.btn_close = QPushButton("Thu nhỏ (-)")
        self.btn_close.clicked.connect(self.handle_minimize) # Dùng hàm riêng
        self.btn_close.setStyleSheet("background: #444; color: white; border-radius: 5px; padding: 6px;")
        self.frame_layout.addWidget(self.btn_close)

        # Worker & Signals
        self.snipper = SnippingWidget()
        self.snipper.snippet_taken.connect(self.process_image)

        self.preload_worker = GOTOCRWorker(mode="preload")
        self.preload_worker.result_ready.connect(self.update_status)
        self.preload_worker.start()

        self.request_snip_signal.connect(self.start_snipping)
        keyboard.add_hotkey('alt+x', self.emit_snip_signal)

    def handle_minimize(self):
        # Ép cửa sổ thu nhỏ
        self.setWindowState(Qt.WindowMinimized)

    def emit_snip_signal(self):
        self.request_snip_signal.emit()

    def start_snipping(self):
        self.hide()
        self.snipper.start_selection()

    def process_image(self, img_path):
        self.showNormal() # Hiện lại cửa sổ
        self.activateWindow()
        self.update_status("⏳ Đang xử lý ảnh...")
        self.worker = GOTOCRWorker(image_path=img_path, mode="scan")
        self.worker.result_ready.connect(self.update_status)
        self.worker.start()

    def update_status(self, text):
        self.lbl_result.setText(text)

    # --- FIX LỖI KÉO CỬA SỔ ---
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            # Nếu bấm vào nút thì KHÔNG tính là kéo cửa sổ
            if isinstance(self.childAt(event.pos()), QPushButton):
                return
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
