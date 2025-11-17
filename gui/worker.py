# gui/worker.py
# coding: utf-8
from PySide6.QtCore import QThread, Signal
from pathlib import Path
import threading
from core.transcoder import convert_video

class WorkerSignals:
    """信号占位，使用 QThread 自带的 signal 替代"""
    progress = Signal(str, int, int)
    finished = Signal(dict)
    log = Signal(str)

class TranscodeWorker(QThread):
    progress = Signal(str, int, int)
    finished = Signal(dict)
    log = Signal(str)

    def __init__(self, file_path: Path, out_dir: Path, debug=False,
                 skip_validator=False, force_cpu=False, force_gpu=False):
        super().__init__()
        self.file_path = file_path
        self.out_dir = out_dir
        self.debug = debug
        self.skip_validator = skip_validator
        self.force_cpu = force_cpu
        self.force_gpu = force_gpu
        self.stop_event = threading.Event()

    def run(self):
        try:
            result = convert_video(
                self.file_path,
                self.out_dir,
                progress_callback=self.progress.emit,
                debug=self.debug,
                skip_validator=self.skip_validator,
                force_cpu=self.force_cpu,
                force_gpu=self.force_gpu,
                stop_event=self.stop_event
            )
            self.finished.emit(result)
        except Exception as e:
            self.log.emit(f"[ERROR] {self.file_path.name}: {e}")
            self.finished.emit({
                "file": self.file_path.name,
                "status": "FAILED",
                "quality": None,
                "retries": 0,
                "method": "UNKNOWN",
                "hdr": False
            })

    def stop(self):
        """外部调用以请求停止任务"""
        self.stop_event.set()
