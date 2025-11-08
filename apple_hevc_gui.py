#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Apple HEVC 批量转码 Windows GUI v1.6.3
完全 GUI 版，功能与命令行一致
依赖: PySide6, ffmpeg, ffprobe, 可选 NVIDIA NVENC
"""

import sys, os, json, csv, subprocess, threading, logging, datetime
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor
from PySide6.QtWidgets import (
    QApplication, QWidget, QLabel, QLineEdit, QPushButton, QHBoxLayout, QVBoxLayout,
    QFileDialog, QCheckBox, QSpinBox, QTextEdit, QListWidget, QListWidgetItem, QProgressBar
)
from PySide6.QtCore import Qt, Signal, QObject
from PySide6.QtGui import QPixmap, QIcon

# -------------------- 配置 --------------------
INPUT_EXTS = (
    '.mp4', '.mov', '.mkv', '.avi', '.wmv', '.flv', '.ts', '.m2ts', '.mts',
    '.m4v', '.webm', '.3gp', '.f4v', '.ogv', '.vob', '.mpg', '.mpeg'
)
DEFAULT_CRF = 18
MAX_WORKERS_SDR = max(1, os.cpu_count() or 1)
MAX_WORKERS_HDR = 2
LOG_FILE = "transcode_log.csv"

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# -------------------- 数据结构 --------------------
@dataclass
class VideoInfo:
    width: int
    height: int
    fps: float
    color_primaries: str
    color_transfer: str
    color_space: str
    master_display: str
    max_cll: str
    duration: float

@dataclass
class FFmpegParams:
    vcodec: str
    pix_fmt: str
    profile: str
    level: str
    color_flags: List[str]
    vparams: List[str]

# -------------------- 工具函数 --------------------
def parse_fps(rate_str: str) -> float:
    try:
        if not rate_str or '/' not in rate_str:
            return 30.0
        num, den = map(int, rate_str.split('/'))
        return num / den if den != 0 else 30.0
    except Exception:
        return 30.0

def probe_video(file_path: Path) -> VideoInfo:
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_streams', '-show_format', str(file_path)],
            capture_output=True, text=True, check=True, encoding='utf-8', errors='replace'
        )
        info = json.loads(result.stdout)
        v = next(s for s in info.get('streams', []) if s.get('codec_type') == 'video')
        width = v.get('width', 1920)
        height = v.get('height', 1080)
        fps = parse_fps(v.get('avg_frame_rate') or v.get('r_frame_rate', '30/1'))
        tags = info.get('format', {}).get('tags', {})
        duration = float(info.get('format', {}).get('duration', 0) or 0)
        return VideoInfo(
            width, height, fps,
            v.get('color_primaries', 'bt709'),
            v.get('color_transfer', 'bt709'),
            v.get('color_space', 'bt709'),
            tags.get('master-display', ''),
            tags.get('max-cll', ''),
            duration
        )
    except Exception as e:
        logger.error(f"探测视频信息失败 {file_path}: {e}")
        return VideoInfo(1920, 1080, 30, 'bt709', 'bt709', 'bt709', '', '', 0)

def probe_audio_channels(file_path: Path) -> int:
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'quiet', '-select_streams', 'a:0',
             '-show_entries', 'stream=channels', '-of', 'csv=p=0', str(file_path)],
            capture_output=True, text=True, check=True, encoding='utf-8', errors='replace'
        )
        return int(result.stdout.strip() or 2)
    except:
        return 2

def is_hdr(info: VideoInfo) -> bool:
    return info.color_transfer.lower() == 'smpte2084' or info.color_primaries.lower() == 'bt2020'

def calculate_apple_hevc_level(info: VideoInfo) -> str:
    mb_w = (info.width + 15) // 16
    mb_h = (info.height + 15) // 16
    total_mbs = mb_w * mb_h
    mbps = total_mbs * info.fps
    levels = [
        {'level': '4.0', 'max_mbs': 8192, 'max_mbps': 245760},
        {'level': '4.1', 'max_mbs': 8192, 'max_mbps': 552960},
        {'level': '5.0', 'max_mbs': 22080, 'max_mbps': 983040},
        {'level': '5.1', 'max_mbs': 36864, 'max_mbps': 2073600},
        {'level': '5.2', 'max_mbs': 36864, 'max_mbps': 4177920}
    ]
    for l in levels:
        if total_mbs <= l['max_mbs'] and mbps <= l['max_mbps']:
            return l['level']
    return '5.2'

def has_nvenc() -> bool:
    try:
        result = subprocess.run(['ffmpeg', '-hide_banner', '-encoders'],
                                capture_output=True, text=True, check=True, encoding='utf-8', errors='replace')
        return 'hevc_nvenc' in result.stdout
    except:
        return False

def detect_gpu_type() -> str:
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                                capture_output=True, text=True, check=True, encoding='utf-8', errors='replace')
        return result.stdout.strip().lower()
    except:
        return "unknown"

def build_hdr_metadata(master_display: str, max_cll: str) -> List[str]:
    if not master_display:
        master_display = 'G(13250,34500)B(7500,3000)R(34000,16000)WP(15635,16450)L(10000000,50)'
    if not max_cll:
        max_cll = '1000,400'
    return ['-color_primaries', 'bt2020', '-color_trc', 'smpte2084', '-colorspace', 'bt2020nc',
            '-master-display', master_display, '-max-cll', max_cll]

def auto_nvenc_cq(info: VideoInfo, gpu_name: str) -> int:
    if max(info.width, info.height) >= 3840:
        base_cq = 18
    elif max(info.width, info.height) >= 2560:
        base_cq = 19
    else:
        base_cq = 20
    if "rtx" in gpu_name:
        base_cq -= 1
    return base_cq

def select_nvenc_preset(info: VideoInfo, gpu_name: str) -> str:
    res = max(info.width, info.height)
    if 'rtx' in gpu_name:
        if res >= 3840: return 'p7'
        elif res >= 2560: return 'p7'
        else: return 'p6'
    else:
        if res >= 3840: return 'p6'
        elif res >= 2560: return 'p6'
        else: return 'p5'

def decide_encoder(info: VideoInfo, force_cpu: bool, force_gpu: bool) -> bool:
    hdr = is_hdr(info)
    if hdr or force_cpu:
        return False
    if force_gpu:
        return has_nvenc()
    return has_nvenc()

def build_ffmpeg_params(info: VideoInfo, use_nvenc: bool, crf: int, gpu_name: str) -> FFmpegParams:
    hdr = is_hdr(info)
    is_vertical = info.height > info.width
    level = calculate_apple_hevc_level(info)
    profile = 'main10' if hdr else 'main'
    pix_fmt = 'p010le' if hdr else 'yuv420p'
    keyint = max(1, int(round(info.fps * (1.2 if is_vertical else 1.5))))
    color_flags = ['-color_primaries','bt709','-color_trc','bt709','-colorspace','bt709']
    if hdr:
        color_flags = build_hdr_metadata(info.master_display, info.max_cll)
        if not use_nvenc or not has_nvenc():
            use_nvenc = False

    if use_nvenc:
        vcodec = 'hevc_nvenc'
        cq_base = auto_nvenc_cq(info, gpu_name)
        if is_vertical: cq_base = max(15, cq_base-1)
        preset = select_nvenc_preset(info, gpu_name)
        if is_vertical and preset.startswith("p") and preset[1:].isdigit():
            preset = f"p{min(int(preset[1:])+1,7)}"
        vparams = [
            '-rc', 'vbr', '-tune', 'hq', '-multipass', 'fullres', '-cq', str(cq_base), '-b:v', '0',
            '-bf', '3', '-b_ref_mode', 'middle', '-rc-lookahead', '20', '-spatial-aq', '1',
            '-aq-strength', '8', '-temporal-aq', '1', '-preset', preset, '-strict_gop', '1',
            '-no-scenecut', '1', '-g', str(keyint), '-bsf:v', 'hevc_metadata=aud=insert',
            '-movflags', 'use_metadata_tags+faststart'
        ]
    else:
        vcodec = 'libx265'
        x265_params = [
            f'crf={crf}', 'log-level=error', 'repeat-headers=1', 'aud=1', 'hrd=1',
            'strong-intra-smoothing=0', 'psy-rd=2', 'psy-rdoq=1.5',
            'profile=main10' if hdr else 'profile=main'
        ]
        if hdr:
            x265_params += ['hdr10=1','hdr-opt=1','colorprim=bt2020','transfer=smpte2084','colormatrix=bt2020nc']
        if info.master_display:
            x265_params.append(f'master-display={info.master_display}')
        if info.max_cll:
            x265_params.append(f'max-cll={info.max_cll}')
        vparams = ['-preset','slow','-x265-params',':'.join(x265_params)]
    return FFmpegParams(vcodec,pix_fmt,profile,level,color_flags,vparams)

COMMON_FFMPEG_FLAGS = [
    '-metadata:s:v:0','handler_name=VideoHandler',
    '-metadata:s:a:0','handler_name=SoundHandler',
    '-c:a','aac','-b:a','192k','-ar','48000'
]

NVENC_RETRIES = [
    {'-bf':'3','-b_ref_mode':'middle'},
    {'-bf':'0','-b_ref_mode':'disabled'},
    {'-bf':'0','-b_ref_mode':'disabled','-temporal-aq':'0'},
    {'-bf':'0','-b_ref_mode':'disabled','-temporal-aq':'0','-spatial-aq':'0'}
]

def adjust_nvenc_params(params: List[str], attempt: int) -> List[str]:
    new_params = params.copy()
    if 1 <= attempt <= len(NVENC_RETRIES):
        retry_mods = NVENC_RETRIES[attempt-1]
        for key,val in retry_mods.items():
            if key in new_params:
                idx = new_params.index(key)+1
                new_params[idx] = val
    return new_params

def build_ffmpeg_command(file_path, out_path, ff_params, vparams, audio_channels):
    vparams_to_use = vparams if vparams is not None else ff_params.vparams
    return [
        'ffmpeg', '-hide_banner', '-y', '-i', str(file_path),
        '-map_metadata', '0',
        '-c:v', ff_params.vcodec,
        '-pix_fmt', ff_params.pix_fmt,
        '-profile:v', ff_params.profile,
        '-level:v', ff_params.level,
        '-tag:v', 'hvc1',
        *ff_params.color_flags,
        *vparams_to_use,
        '-ac', str(audio_channels),
        *COMMON_FFMPEG_FLAGS,
        str(out_path)
    ]

# -------------------- 转码逻辑 --------------------
def convert_video(file_path: Path, out_dir: Path, progress_callback=None,
                  debug=False, skip_validator=False, force_cpu=False, force_gpu=False):
    info = probe_video(file_path)
    gpu_name = detect_gpu_type()
    out_path = out_dir / (file_path.stem + '.mp4')
    crf_attempt = DEFAULT_CRF
    use_nvenc = decide_encoder(info, force_cpu, force_gpu)
    audio_channels = probe_audio_channels(file_path)
    hdr = is_hdr(info)
    log_entry = {"file": file_path.name, "status": "FAILED", "crf": None,
                 "retries": 0, "method": "NVENC" if use_nvenc else "CPU", "hdr": hdr}

    ff_params = build_ffmpeg_params(info, use_nvenc, crf_attempt, gpu_name)
    total_frames = max(1, int(info.duration * info.fps))

    def run_ffmpeg(cmd):
        with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace') as proc:
            frame = 0
            for line in proc.stdout:
                if "frame=" in line:
                    try:
                        frame = int(line.strip().split("frame=")[-1].split()[0])
                    except:
                        frame = 0
                    if progress_callback:
                        progress_callback(file_path.name, frame, total_frames)
            return proc.wait()

    if use_nvenc:
        for attempt in range(1, len(NVENC_RETRIES) + 2):
            retry_vparams = adjust_nvenc_params(ff_params.vparams, attempt if attempt <= len(NVENC_RETRIES) else 1)
            cmd = build_ffmpeg_command(file_path, out_path, ff_params, vparams=retry_vparams, audio_channels=audio_channels)
            if debug: logger.debug(f"NVENC FFmpeg 命令 (尝试 {attempt}): {' '.join(cmd)}")
            if run_ffmpeg(cmd) == 0:
                log_entry["status"] = "SUCCESS"
                log_entry["crf"] = auto_nvenc_cq(info, gpu_name)
                log_entry["retries"] = attempt if attempt <= len(NVENC_RETRIES) else len(NVENC_RETRIES)
                break
            if attempt == len(NVENC_RETRIES) + 1:
                logger.info("NVENC 最终失败，回退 CPU")
                use_nvenc = False
                ff_params = build_ffmpeg_params(info, False, crf_attempt, gpu_name)

    if not use_nvenc:
        cpu_params = build_ffmpeg_params(info, False, crf_attempt, gpu_name)
        cmd_cpu = build_ffmpeg_command(file_path, out_path, cpu_params, None, audio_channels)
        if debug: logger.debug(f"CPU FFmpeg 命令: {' '.join(cmd_cpu)}")
        if run_ffmpeg(cmd_cpu) == 0:
            log_entry["status"] = "SUCCESS"
            log_entry["crf"] = crf_attempt
            log_entry["retries"] = 0
            log_entry["method"] = "CPU"

    if log_entry["status"] == "SUCCESS" and not skip_validator:
        run_apple_validator(out_path)

    if progress_callback:
        progress_callback(file_path.name, total_frames, total_frames)
    return log_entry

def run_apple_validator(file_path: Path):
    possible_paths=[Path('/Applications/Apple Video Tools/AppleHEVCValidator'),
                    Path('/usr/local/bin/AppleHEVCValidator'),
                    Path('C:/Program Files/Apple/AppleHEVCValidator.exe')]
    validator = next((p for p in possible_paths if p.exists()),None)
    if not validator: return
    try: subprocess.run([str(validator),str(file_path)],check=True, encoding='utf-8', errors='replace')
    except subprocess.CalledProcessError: logger.warning(f"Apple Validator 未通过: {file_path.name}")

# -------------------- GUI --------------------
class WorkerSignals(QObject):
    progress = Signal(str, int, int)
    finished = Signal(dict)
    log = Signal(str)

class ConvertWorker(threading.Thread):
    def __init__(self, file_path: Path, out_dir: Path, signals: WorkerSignals,
                 debug=False, skip_validator=False, force_cpu=False, force_gpu=False):
        super().__init__()
        self.file_path = file_path
        self.out_dir = out_dir
        self.signals = signals
        self.debug = debug
        self.skip_validator = skip_validator
        self.force_cpu = force_cpu
        self.force_gpu = force_gpu

    def run(self):
        try:
            res = convert_video(
                self.file_path,
                self.out_dir,
                progress_callback=self.signals.progress.emit,
                debug=self.debug,
                skip_validator=self.skip_validator,
                force_cpu=self.force_cpu,
                force_gpu=self.force_gpu
            )
            self.signals.finished.emit(res)
        except Exception as e:
            self.signals.log.emit(f"[ERROR] {self.file_path.name}: {e}")
            self.signals.finished.emit({
                "file": self.file_path.name,
                "status": "FAILED",
                "crf": None,
                "retries": 0,
                "method": "UNKNOWN",
                "hdr": False
            })

class BatchManager(QObject):
    progress = Signal(str, int, int)
    finished_one = Signal(dict)
    log = Signal(str)

    def __init__(self):
        super().__init__()
        self.executor = None
        self._stopping = False
        self.futures = []

    def start(self, input_dir: Path, output_dir: Path, max_workers=2, debug=False,
              skip_validator=False, force_cpu=False, force_gpu=False):
        files = [f for f in input_dir.rglob("*") if f.suffix.lower() in INPUT_EXTS]
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self._stopping = False
        self.futures.clear()
        for f in files:
            signals = WorkerSignals()
            signals.progress.connect(self.progress.emit)
            signals.finished.connect(self.finished_one.emit)
            signals.log.connect(self.log.emit)
            worker = ConvertWorker(f, output_dir, signals, debug, skip_validator, force_cpu, force_gpu)
            self.futures.append(self.executor.submit(worker.run))

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Apple HEVC 批量转码 v1.6.3")
        self.setWindowIcon(QIcon("icon.ico"))
        self.resize(800,600)
        self.file_frames = {}
        self.results = []
        self.manager = BatchManager()
        self.manager.progress.connect(self.on_progress)
        self.manager.finished_one.connect(self.on_finished)
        self.manager.log.connect(self.append_log)

        layout = QVBoxLayout()
        # ----------------- 版权 -----------------
        year = datetime.datetime.now().year
        copyright_label = QLabel()
        copyright_label.setText(f"© {year} uingei. All rights reserved.")
        copyright_label.setAlignment(Qt.AlignCenter)

        layout.addWidget(copyright_label, alignment=Qt.AlignCenter)
        h1 = QHBoxLayout()
        self.input_edit = QLineEdit(); self.output_edit = QLineEdit()
        b_in = QPushButton("选择输入"); b_out = QPushButton("选择输出")
        b_in.clicked.connect(lambda: self.select_dir(self.input_edit))
        b_out.clicked.connect(lambda: self.select_dir(self.output_edit))
        h1.addWidget(QLabel("输入目录")); h1.addWidget(self.input_edit); h1.addWidget(b_in)
        h2 = QHBoxLayout()
        h2.addWidget(QLabel("输出目录")); h2.addWidget(self.output_edit); h2.addWidget(b_out)
        layout.addLayout(h1); layout.addLayout(h2)

        self.debug_cb = QCheckBox("Debug"); self.skip_validator_cb = QCheckBox("跳过 Validator")
        self.force_cpu_cb = QCheckBox("强制 CPU"); self.force_gpu_cb = QCheckBox("强制 GPU")
        self.workers_spin = QSpinBox(); self.workers_spin.setRange(1, os.cpu_count() or 4); self.workers_spin.setValue(2)
        layout.addWidget(self.debug_cb); layout.addWidget(self.skip_validator_cb)
        layout.addWidget(self.force_cpu_cb); layout.addWidget(self.force_gpu_cb)
        layout.addWidget(QLabel("最大并发线程")); layout.addWidget(self.workers_spin)

        self.file_list = QListWidget()
        self.overall_pb = QProgressBar(); self.overall_pb.setValue(0)
        self.log_edit = QTextEdit(); self.log_edit.setReadOnly(True)
        layout.addWidget(self.file_list); layout.addWidget(QLabel("总体进度")); layout.addWidget(self.overall_pb)
        layout.addWidget(QLabel("日志")); layout.addWidget(self.log_edit)

        b_start = QPushButton("开始批量转码"); b_start.clicked.connect(self.start_batch)
        layout.addWidget(b_start)
        self.setLayout(layout)

    def select_dir(self, edit: QLineEdit):
        d = QFileDialog.getExistingDirectory(self, "选择目录")
        if d: edit.setText(d)

    def append_log(self, text: str):
        self.log_edit.append(text)
        self.log_edit.ensureCursorVisible()
        logger.info(text)

    def start_batch(self):
        input_dir = Path(self.input_edit.text())
        output_dir = Path(self.output_edit.text())
        if not input_dir.exists(): self.append_log('[ERROR] 输入目录不存在'); return
        if not output_dir.exists(): output_dir.mkdir(parents=True, exist_ok=True)

        self.file_list.clear(); self.results.clear(); self.file_frames.clear()
        files = [f for f in input_dir.rglob("*") if f.suffix.lower() in INPUT_EXTS]
        for f in files:
            info = probe_video(f)
            total_frames = max(1, int(info.fps * info.duration))
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

        self.manager.start(input_dir, output_dir, max_workers, debug, skip_validator, force_cpu, force_gpu)

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
        if filename in self.file_frames:
            self.file_frames[filename]['done'] = frame
            self.file_frames[filename]['total'] = total
        total_done = sum(f['done'] for f in self.file_frames.values())
        total_frames = sum(f['total'] for f in self.file_frames.values())
        overall_pct = int(total_done / total_frames * 100) if total_frames else 0
        self.overall_pb.setValue(overall_pct)

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
        try:
            with open(LOG_FILE,'w',newline='',encoding='utf-8') as f:
                w = csv.DictWriter(f, fieldnames=["file","status","crf","retries","method","hdr"])
                w.writeheader()
                w.writerows(self.results)
        except Exception as e:
            self.append_log(f"[ERROR] 写 CSV 失败: {e}")

# -------------------- 主程序 --------------------
if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())
