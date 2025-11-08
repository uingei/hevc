# gui/mainwindow.py
# coding: utf-8
import sys, os, datetime
from pathlib import Path
from PySide6.QtWidgets import (
    QApplication, QWidget, QLabel, QLineEdit, QPushButton, QHBoxLayout, QVBoxLayout,
    QFileDialog, QCheckBox, QSpinBox, QTextEdit, QListWidget, QListWidgetItem, QProgressBar
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon
from gui.worker import TranscodeWorker

INPUT_EXTS = (
    '.mp4', '.mov', '.mkv', '.avi', '.wmv', '.flv', '.ts', '.m2ts', '.mts',
    '.m4v', '.webm', '.3gp', '.f4v', '.ogv', '.vob', '.mpg', '.mpeg'
)

LOG_FILE = "transcode_log.csv"

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Apple HEVC 批量转码 v1.6.3")
        self.setWindowIcon(QIcon("icon.ico"))
        self.resize(800,600)
        self.workers = []
        self.file_frames = {}
        self.results = []

        layout = QVBoxLayout()
        year = datetime.datetime.now().year
        copyright_label = QLabel(
            f"© {year} uingei. All rights reserved."
        )
        copyright_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(copyright_label, alignment=Qt.AlignCenter)

        # 输入输出目录
        h1 = QHBoxLayout()
        self.input_edit = QLineEdit(); self.output_edit = QLineEdit()
        b_in = QPushButton("选择输入"); b_out = QPushButton("选择输出")
        b_in.clicked.connect(lambda: self.select_dir(self.input_edit))
        b_out.clicked.connect(lambda: self.select_dir(self.output_edit))
        h1.addWidget(QLabel("输入目录")); h1.addWidget(self.input_edit); h1.addWidget(b_in)
        h2 = QHBoxLayout()
        h2.addWidget(QLabel("输出目录")); h2.addWidget(self.output_edit); h2.addWidget(b_out)
        layout.addLayout(h1); layout.addLayout(h2)

        # 选项
        self.debug_cb = QCheckBox("Debug")
        self.skip_validator_cb = QCheckBox("跳过 Validator")
        self.force_cpu_cb = QCheckBox("强制 CPU")
        self.force_gpu_cb = QCheckBox("强制 GPU")
        self.workers_spin = QSpinBox(); self.workers_spin.setRange(1, os.cpu_count() or 4); self.workers_spin.setValue(2)
        layout.addWidget(self.debug_cb); layout.addWidget(self.skip_validator_cb)
        layout.addWidget(self.force_cpu_cb); layout.addWidget(self.force_gpu_cb)
        layout.addWidget(QLabel("最大并发线程")); layout.addWidget(self.workers_spin)

        # 文件列表、进度条、日志
        self.file_list = QListWidget()
        self.overall_pb = QProgressBar(); self.overall_pb.setValue(0)
        self.log_edit = QTextEdit(); self.log_edit.setReadOnly(True)
        layout.addWidget(self.file_list)
        layout.addWidget(QLabel("总体进度")); layout.addWidget(self.overall_pb)
        layout.addWidget(QLabel("日志")); layout.addWidget(self.log_edit)

        # 按钮
        b_start = QPushButton("开始批量转码"); b_stop = QPushButton("停止")
        b_start.clicked.connect(self.start_batch)
        b_stop.clicked.connect(self.stop_all)
        layout.addWidget(b_start)
        layout.addWidget(b_stop)

        self.setLayout(layout)

    def select_dir(self, edit: QLineEdit):
        d = QFileDialog.getExistingDirectory(self, "选择目录")
        if d: edit.setText(d)

    def append_log(self, text: str):
        self.log_edit.append(text)
        self.log_edit.ensureCursorVisible()

    def start_batch(self):
        input_dir = Path(self.input_edit.text())
        output_dir = Path(self.output_edit.text())
        if not input_dir.exists():
            self.append_log('[ERROR] 输入目录不存在')
            return
        if not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)

        self.file_list.clear()
        self.results.clear()
        self.file_frames.clear()
        files = [f for f in input_dir.rglob("*") if f.suffix.lower() in INPUT_EXTS]
        for f in files:
            total_frames = 1  # 先占位
            self.file_frames[f.name] = {'done':0,'total':total_frames}
            it = QListWidgetItem(f.name)
            it.setData(Qt.UserRole, {'path':f, 'status':'PENDING','progress':0})
            self.file_list.addItem(it)

        debug = self.debug_cb.isChecked()
        skip_validator = self.skip_validator_cb.isChecked()
        force_cpu = self.force_cpu_cb.isChecked()
        force_gpu = self.force_gpu_cb.isChecked()
        max_workers = self.workers_spin.value()
        self.append_log(f'[INFO] 开始处理 {len(files)} 个文件, 并发 {max_workers}')

        # 启动 worker
        self.workers = []
        for f in files:
            worker = TranscodeWorker(f, output_dir, debug, skip_validator, force_cpu, force_gpu)
            worker.progress.connect(self.on_progress)
            worker.finished.connect(self.on_finished)
            worker.log.connect(self.append_log)
            worker.start()
            self.workers.append(worker)

    def stop_all(self):
        for w in self.workers:
            w.stop()
        self.append_log("[INFO] 已请求停止所有任务")

    def on_progress(self, filename: str, frame: int, total: int):
        for i in range(self.file_list.count()):
            it = self.file_list.item(i)
            d = it.data(Qt.UserRole)
            if d and d.get('path') and d['path'].name == filename:
                d['status'] = 'PROCESSING'
                d['progress'] = int(frame / total * 100 if total else 0)
                it.setText(f"{filename} — {d['status']} — {d['progress']}%")
                it.setData(Qt.UserRole,d)
                break

    def on_finished(self, log_entry: dict):
        self.results.append(log_entry)
        for i in range(self.file_list.count()):
            it = self.file_list.item(i)
            d = it.data(Qt.UserRole)
            if d and d.get('path') and d['path'].name == log_entry['file']:
                d['status'] = log_entry['status']
                d['progress'] = 100
                it.setText(f"{log_entry['file']} — {log_entry['status']} — 100%")
                it.setData(Qt.UserRole,d)
                break
        self.append_log(f"[INFO] 完成 {log_entry['file']} — {log_entry['status']}")
        self.save_csv()

    def save_csv(self):
        import csv
        try:
            with open(LOG_FILE,'w',newline='',encoding='utf-8') as f:
                w = csv.DictWriter(f, fieldnames=["file","status","crf","retries","method","hdr"])
                w.writeheader()
                w.writerows(self.results)
        except Exception as e:
            self.append_log(f"[ERROR] 写 CSV 失败: {e}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())
