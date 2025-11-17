# gui/mainwindow.py
# coding: utf-8
import sys, os, datetime
from pathlib import Path
from collections import deque
from PySide6.QtWidgets import (
    QApplication, QWidget, QLabel, QLineEdit, QPushButton, QHBoxLayout, QVBoxLayout,
    QFileDialog, QCheckBox, QSpinBox, QTextEdit, QListWidget, QListWidgetItem,
    QProgressBar, QFrame, QRadioButton, QButtonGroup
)
from PySide6.QtCore import Qt, QRectF, QTimer
from PySide6.QtGui import QIcon, QPainter, QColor, QFontMetrics
from gui.worker import TranscodeWorker

INPUT_EXTS = (
    '.mp4', '.mov', '.mkv', '.avi', '.wmv', '.flv', '.ts', '.m2ts', '.mts',
    '.m4v', '.webm', '.3gp', '.f4v', '.ogv', '.vob', '.mpg', '.mpeg'
)

LOG_FILE = "transcode_log.csv"

# --- 单文件进度条 ---
class TextProgressBar(QProgressBar):
    """简化版进度条 + 文件名 + 状态文字 + 呼吸动画 + 局部反色"""
    def __init__(self, filename: str):
        super().__init__()
        self.filename = filename
        self.status_text = "等待中"
        self.setTextVisible(False)
        self.setMinimumHeight(26)
        self._success = None
        self._pulse = 0.0
        self._pulse_direction = 1
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._update_pulse)
        self._timer.start(50)

    def _update_pulse(self):
        step = 0.03
        self._pulse += step * self._pulse_direction
        if self._pulse >= 1.0:
            self._pulse = 1.0
            self._pulse_direction = -1
        elif self._pulse <= 0.0:
            self._pulse = 0.0
            self._pulse_direction = 1
        self.update()

    def set_status(self, text: str):
        self.status_text = text
        self.update()

    def set_finished(self, success: bool):
        self._success = success
        self.update()

    def paintEvent(self, event):
        rect = self.rect()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setBrush(QColor(240, 240, 240))
        painter.setPen(QColor(180, 180, 180))
        painter.drawRoundedRect(rect.adjusted(0.5, 0.5, -0.5, -0.5), 6, 6)

        progress_ratio = self.value() / 100.0
        filled_width = rect.width() * progress_ratio
        if filled_width > 0:
            base_color = QColor(0, 122, 255)
            pulse_offset = int(self._pulse * 40)
            fill_color = QColor(
                min(base_color.red() + pulse_offset, 255),
                min(base_color.green() + pulse_offset, 255),
                min(base_color.blue() + pulse_offset, 255)
            )
            painter.setBrush(fill_color)
            painter.setPen(Qt.NoPen)
            painter.drawRoundedRect(QRectF(rect.left(), rect.top(), filled_width, rect.height()).adjusted(0, 0, -0.5, 0), 6, 6)

        fm = QFontMetrics(self.font())
        status_text = f"{self.status_text} ({self.value()}%)"
        status_width = fm.horizontalAdvance(status_text) + 6
        right_rect = QRectF(rect.right() - status_width - 6, rect.top(), status_width, rect.height())
        name, ext = os.path.splitext(self.filename)
        max_name_width = rect.width() - status_width - 12
        elided_name = fm.elidedText(name, Qt.ElideRight, max_name_width - fm.horizontalAdvance(ext)) + ext
        left_rect = QRectF(6, rect.top(), max_name_width, rect.height())

        overlap_left = min(left_rect.right(), filled_width) - left_rect.left()
        if overlap_left > 0:
            painter.setClipRect(QRectF(left_rect.left(), left_rect.top(), overlap_left, left_rect.height()))
            painter.setPen(Qt.white)
            painter.drawText(left_rect, Qt.AlignVCenter | Qt.AlignLeft, elided_name)
        if overlap_left < left_rect.width():
            painter.setClipRect(QRectF(left_rect.left() + overlap_left, left_rect.top(),
                                       left_rect.width() - overlap_left, left_rect.height()))
            painter.setPen(QColor(30, 30, 30))
            painter.drawText(left_rect, Qt.AlignVCenter | Qt.AlignLeft, elided_name)

        overlap_right = max(filled_width - right_rect.left(), 0)
        overlap_right = min(overlap_right, right_rect.width())
        if overlap_right > 0:
            painter.setClipRect(QRectF(right_rect.left(), right_rect.top(), overlap_right, right_rect.height()))
            painter.setPen(Qt.white)
            painter.drawText(right_rect, Qt.AlignVCenter | Qt.AlignRight, status_text)
        if overlap_right < right_rect.width():
            painter.setClipRect(QRectF(right_rect.left() + overlap_right, right_rect.top(),
                                       right_rect.width() - overlap_right, right_rect.height()))
            painter.setPen(QColor(30, 30, 30))
            painter.drawText(right_rect, Qt.AlignVCenter | Qt.AlignRight, status_text)
        painter.end()

# --- 文件项 ---
class FileItemWidget(QFrame):
    def __init__(self, filename: str):
        super().__init__()
        self.filename = filename
        self.setFrameShape(QFrame.StyledPanel)
        self.setAutoFillBackground(True)
        self.setStyleSheet("""
            QFrame {
                border: 1px solid #ccc;
                border-radius: 6px;
                background-color: #fafafa;
            }
        """)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 4, 6, 4)
        self.pb = TextProgressBar(filename)
        layout.addWidget(self.pb)

    def update_progress(self, percent: int, status: str):
        self.pb.setValue(percent)
        self.pb.set_status(status)

    def set_finished(self, success: bool):
        if success:
            self.setStyleSheet("""
                QFrame {
                    border: 1px solid #4CAF50;
                    border-radius: 6px;
                    background-color: #f2f2f2;
                }
            """)
            self.pb.set_finished(True)
        else:
            self.setStyleSheet("""
                QFrame {
                    border: 1px solid #d88;
                    border-radius: 6px;
                    background-color: #ffecec;
                }
            """)
            self.pb.set_finished(False)

# --- 主窗口 ---
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Apple HEVC 批量转码 v1.6.5")
        self.setWindowIcon(QIcon("icon.ico"))
        self.resize(900, 650)
        self.workers = []
        self.file_widgets = {}
        self.results = []
        self.queue = deque()           # 待处理队列
        self.active_workers = []       # 当前运行的 Worker

        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        year = datetime.datetime.now().year
        layout.addWidget(QLabel(f"© {year} uingei. All rights reserved."), alignment=Qt.AlignCenter)

        # 输入输出
        self.input_edit = QLineEdit()
        self.output_edit = QLineEdit()
        b_in = QPushButton("选择输入")
        b_out = QPushButton("选择输出")
        b_in.clicked.connect(lambda: self.select_dir(self.input_edit))
        b_out.clicked.connect(lambda: self.select_dir(self.output_edit))
        h1 = QHBoxLayout()
        h1.addWidget(QLabel("输入目录"))
        h1.addWidget(self.input_edit)
        h1.addWidget(b_in)
        h2 = QHBoxLayout()
        h2.addWidget(QLabel("输出目录"))
        h2.addWidget(self.output_edit)
        h2.addWidget(b_out)
        layout.addLayout(h1)
        layout.addLayout(h2)

        # 控制选项
        self.debug_cb = QCheckBox("Debug")
        self.skip_validator_cb = QCheckBox("跳过 Validator")

        self.force_cpu_rb = QRadioButton("强制 CPU")
        self.force_gpu_rb = QRadioButton("强制 GPU")
        self.force_cpu_rb.setChecked(True)  # 默认选 CPU

        self.force_group = QButtonGroup(self)
        self.force_group.addButton(self.force_cpu_rb)
        self.force_group.addButton(self.force_gpu_rb)

        self.workers_spin = QSpinBox()
        self.workers_spin.setRange(1, os.cpu_count() or 4)
        self.workers_spin.setValue(2)

        option_layout = QHBoxLayout()
        left_opts = QVBoxLayout()
        left_opts.addWidget(self.debug_cb)
        left_opts.addWidget(self.skip_validator_cb)
        right_opts = QVBoxLayout()
        right_opts.addWidget(self.force_cpu_rb)
        right_opts.addWidget(self.force_gpu_rb)
        option_layout.addLayout(left_opts)
        option_layout.addLayout(right_opts)
        layout.addLayout(option_layout)

        layout.addWidget(QLabel("最大并发线程"))
        layout.addWidget(self.workers_spin)

        # 文件列表 & 总体进度 & 日志
        self.file_list = QListWidget()
        layout.addWidget(self.file_list)
        layout.addWidget(QLabel("总体进度"))
        self.overall_pb = QProgressBar()
        self.overall_pb.setValue(0)
        layout.addWidget(self.overall_pb)
        layout.addWidget(QLabel("日志"))
        self.log_edit = QTextEdit()
        self.log_edit.setReadOnly(True)
        layout.addWidget(self.log_edit)

        # 控制按钮
        btn_layout = QHBoxLayout()
        b_start = QPushButton("开始批量转码")
        b_stop = QPushButton("停止")
        b_start.clicked.connect(self.start_batch)
        b_stop.clicked.connect(self.stop_all)
        btn_layout.addWidget(b_start)
        btn_layout.addWidget(b_stop)
        layout.addLayout(btn_layout)
        self.setLayout(layout)

    def select_dir(self, edit: QLineEdit):
        d = QFileDialog.getExistingDirectory(self, "选择目录")
        if d:
            edit.setText(d)

    def append_log(self, text: str):
        self.log_edit.append(text)
        self.log_edit.ensureCursorVisible()

    # --- 批量处理 ---
    def start_batch(self):
        input_dir = Path(self.input_edit.text())
        output_dir = Path(self.output_edit.text())
        if not input_dir.exists():
            self.append_log("[ERROR] 输入目录不存在")
            return
        output_dir.mkdir(parents=True, exist_ok=True)

        self.file_list.clear()
        self.file_widgets.clear()
        self.results.clear()
        self.queue.clear()
        self.active_workers.clear()

        files = [f for f in input_dir.rglob("*") if f.suffix.lower() in INPUT_EXTS]
        if not files:
            self.append_log("[WARN] 未找到支持的视频文件")
            return

        for f in files:
            item = QListWidgetItem(self.file_list)
            widget = FileItemWidget(f.name)
            self.file_list.addItem(item)
            self.file_list.setItemWidget(item, widget)
            item.setSizeHint(widget.sizeHint())
            self.file_widgets[f.name] = widget
            self.queue.append(f)

        debug = self.debug_cb.isChecked()
        skip_validator = self.skip_validator_cb.isChecked()
        force_cpu = self.force_cpu_rb.isChecked()
        force_gpu = self.force_gpu_rb.isChecked()
        max_workers = self.workers_spin.value()
        self.append_log(f"[INFO] 开始处理 {len(files)} 个文件, 并发 {max_workers}")

        for _ in range(min(max_workers, len(self.queue))):
            self._start_next_worker(debug, skip_validator, force_cpu, force_gpu, output_dir)

    def _start_next_worker(self, debug, skip_validator, force_cpu, force_gpu, output_dir):
        if not self.queue:
            return
        f = self.queue.popleft()
        worker = TranscodeWorker(f, output_dir, debug, skip_validator, force_cpu, force_gpu)
        worker.progress.connect(self.on_progress)
        worker.finished.connect(self.on_finished)
        worker.log.connect(self.append_log)
        worker.start()
        self.active_workers.append(worker)

    def stop_all(self):
        for w in self.active_workers:
            w.stop()
        self.append_log("[INFO] 已请求停止所有任务")

    def on_progress(self, filename: str, frame: int, total: int):
        widget = self.file_widgets.get(filename)
        if widget and widget.pb.value() < 100:
            percent = int(frame / total * 100 if total else 0)
            widget.update_progress(percent, "处理中")

        if self.file_widgets:
            avg = sum(w.pb.value() for w in self.file_widgets.values()) / len(self.file_widgets)
            self.overall_pb.setValue(int(avg))

    def on_finished(self, log_entry: dict):
        filename = log_entry["file"]
        widget = self.file_widgets.get(filename)
        if widget:
            success = log_entry["status"].lower() == "success"
            widget.update_progress(100, log_entry["status"])
            widget.set_finished(success)

        self.results.append(log_entry)
        self.append_log(f"[INFO] 完成 {filename} — {log_entry['status']}")
        self.save_csv()

        finished_worker = next((w for w in self.active_workers if w.file_path.name == filename), None)
        if finished_worker:
            self.active_workers.remove(finished_worker)

        debug = self.debug_cb.isChecked()
        skip_validator = self.skip_validator_cb.isChecked()
        force_cpu = self.force_cpu_rb.isChecked()
        force_gpu = self.force_gpu_rb.isChecked()
        output_dir = Path(self.output_edit.text())
        max_workers = self.workers_spin.value()
        if len(self.active_workers) < max_workers:
            self._start_next_worker(debug, skip_validator, force_cpu, force_gpu, output_dir)

        if self.file_widgets:
            avg = sum(w.pb.value() for w in self.file_widgets.values()) / len(self.file_widgets)
            self.overall_pb.setValue(int(avg))

    def save_csv(self):
        import csv
        try:
            with open(LOG_FILE, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=["file", "status", "quality", "retries", "method", "hdr"])
                w.writeheader()
                w.writerows(self.results)
        except Exception as e:
            self.append_log(f"[ERROR] 写 CSV 失败: {e}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())
