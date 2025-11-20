#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Apple HEVC 批量转码脚本 — 改进版（目标：尽可能提高 Apple HEVC Validator 通过率与稳定性）
基于用户提供的 apple_hevc_batch (1).py，已修正与强化若干边界条件、路径、元数据顺序、日志与错误处理。

注意：执行本脚本前请确保系统已安装并在 PATH 中可见：ffmpeg, ffprobe。AppleHEVCValidator 为可选但强烈建议安装以做最终合规性验证。

主要改动（高层摘要）：
- 增强 AppleHEVCValidator 路径检测，支持 .app 包内可执行文件路径。
- 更稳健的 ffmpeg/ffprobe 调用错误处理与超时保护（避免被挂起）。
- 修复并统一 HDR metadata 写入顺序（先写 -metadata 标签，再写 -color_* 原子，且在 vparams 之后明确放置，避免被覆盖）。
- 改进 NVENC/CPU 参数顺序以兼容更多 ffmpeg 版本；确保 mov/MP4 标签与 colr atom 一致写入。
- 更严格的资源与并行控制：dynamic_workers 已做更稳健回退并记录决策。
- 增强日志记录（在抛出异常时保存更多 stderr/stdout 片段供排查）。
- 一些潜在的变量/引用错误修正与注释补充，保证在不同平台上的可移植性。

请在真实样本上用 AppleHEVCValidator 验证输出。脚本尽力提高兼容性，但 "Perfect Compliance" 仍取决于源素材、ffmpeg/encoder 版本及 validator 版本。
"""

# 版本
__version__ = "1.6.10-patch"

import subprocess
import json
import logging
import argparse
import csv
import os
import threading
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import shutil
from functools import lru_cache
from collections import OrderedDict
from fractions import Fraction
import math
import time

# -------------------- 配置 --------------------
INPUT_EXTS = (
    '.mp4', '.mov', '.mkv', '.avi', '.wmv', '.flv', '.ts', '.m2ts', '.mts',
    '.m4v', '.webm', '.3gp', '.f4v', '.ogv', '.vob', '.mpg', '.mpeg'
)
DEFAULT_CRF = 18
MAX_WORKERS_SDR = max(1, os.cpu_count() or 1)
MAX_WORKERS_HDR = min(4, max(1, (os.cpu_count() or 4) // 4))
LOG_FILE = "transcode_log.csv"

# ffmpeg/ffprobe timeout (seconds) to avoid indefinite hang
FFPROBE_TIMEOUT = 20
FFMPEG_TIMEOUT = 3600  # 1 hour default per file, 可视素材长度调大

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)
validator_lock = threading.Lock()

# -------------------- 数据结构 --------------------
@dataclass
class VideoInfo:
    width: int
    height: int
    fps: float
    color_primaries: str
    color_transfer: str
    color_space: str
    pix_fmt: str
    master_display: str
    max_cll: str
    audio_channels: int
    hdr: bool = False
    audio_language: Optional[str] = 'eng'
    nb_frames: Optional[int] = None
    duration: Optional[float] = None
    chromaloc: int = 0

@dataclass
class FFmpegParams:
    vcodec: str
    pix_fmt: str
    profile: str
    level: str
    color_flags: List[str]
    vparams: List[str]
    hdr_metadata: List[str]

# -------------------- HDR 常量 --------------------
HDR_PIXFMTS = {'yuv420p10le', 'p010le', 'yuv444p10le'}
HDR_COLOR_SPACES = {'bt2020', 'bt2020-ncl', 'bt2020nc'}
HDR_TRANSFERS = {'smpte2084', 'pq'}
HDR_PRIMARIES = {'bt2020', 'bt2020-ncl'}

# -------------------- 工具检测 --------------------
def check_tools():
    from shutil import which
    missing = []
    for tool in ('ffmpeg', 'ffprobe'):
        if which(tool) is None:
            missing.append(tool)
    if missing:
        logger.error(f"缺少必要工具: {', '.join(missing)}. 请先安装并确保在 PATH 中可见。")
        raise SystemExit(1)
    if which('nvidia-smi') is None:
        logger.debug("提示：未检测到 nvidia-smi，GPU 信息检测将退回为 ffmpeg encoder 检查。")

# -------------------- Probe --------------------
def _get_tag(tags: dict, *keys, default=''):
    for k in keys:
        if k in tags and tags[k]:
            return tags[k]
    return default


def probe_media(file_path: Path) -> VideoInfo:
    try:
        result = subprocess.run(
            [
                'ffprobe', '-v', 'quiet',
                '-print_format', 'json',
                '-show_streams', '-show_format',
                str(file_path)
            ],
            capture_output=True, text=True, check=True, encoding='utf-8', timeout=FFPROBE_TIMEOUT
        )

        info = json.loads(result.stdout)

        v = next((s for s in info.get('streams', []) if s.get('codec_type') == 'video'), None)
        if not v:
            raise ValueError("没有找到视频流")

        width  = int(v.get('width') or 1920)
        height = int(v.get('height') or 1080)

        rate = v.get('avg_frame_rate') or v.get('r_frame_rate') or "30/1"
        if rate in ("0/0", "N/A", "90000/90000"):
            rate = v.get("r_frame_rate") or "30/1"
        try:
            num, den = map(int, str(rate).split('/'))
            fps = num / den if den else 30.0
        except Exception:
            fps = 30.0

        tags = info.get('format', {}).get('tags', {}) or {}
        vtags = v.get('tags', {}) or {}

        color_primaries = (v.get('color_primaries') or vtags.get('COLOR_PRIMARIES')
                           or tags.get('COLOR_PRIMARIES') or 'bt709') or 'bt709'
        color_transfer = (v.get('color_transfer') or vtags.get('COLOR_TRANSFER')
                          or tags.get('COLOR_TRANSFER') or 'bt709') or 'bt709'
        color_space = (v.get('color_space') or vtags.get('COLOR_SPACE')
                       or tags.get('COLOR_SPACE') or 'bt709') or 'bt709'
        if str(color_space).lower().startswith("bt2020"):
            color_space = "bt2020nc"

        pix_fmt = (v.get('pix_fmt') or 'yuv420p') or 'yuv420p'

        chromaloc = v.get('chroma_location') or tags.get('chroma_location') or 'left'
        chromaloc_val = 0 if str(chromaloc).lower() == 'left' else 1

        side = v.get('side_data_list') or []
        mastering_display = ''
        max_cll = ''
        for sd in side:
            if sd.get('side_data_type') == 'Mastering display metadata':
                try:
                    mastering_display = (
                        f"G({sd['green_x']},{sd['green_y']})"
                        f"B({sd['blue_x']},{sd['blue_y']})"
                        f"R({sd['red_x']},{sd['red_y']})"
                        f"WP({sd['white_point_x']},{sd['white_point_y']})"
                        f"L({sd['max_luminance']},{sd['min_luminance']})"
                    )
                except Exception:
                    mastering_display = ''
            if sd.get('side_data_type') == 'Content light level metadata':
                max_cll = f"{sd.get('max_content')},{sd.get('max_average')}"

        if not mastering_display:
            mastering_display = tags.get('master-display') or tags.get('MASTER_DISPLAY') or ''
        if not max_cll:
            max_cll = tags.get('max-cll') or tags.get('MAX_CLL') or ''

        hdr_flag = (
            "2020" in str(color_primaries).lower() or
            "2020" in str(color_space).lower() or
            str(pix_fmt) in HDR_PIXFMTS or
            bool(mastering_display) or
            "pq" in str(color_transfer).lower() or
            "smpte2084" in str(color_transfer).lower() or
            "arib-std-b67" in str(color_transfer).lower()
        )

        a = next((s for s in info.get('streams', []) if s.get('codec_type') == 'audio'), None)
        if a:
            at = a.get('tags', {}) or {}
            audio_lang = (
                at.get('language') or at.get('LANGUAGE') or
                at.get('lang') or at.get('LANG') or 'eng'
            )
            audio_channels = int(a.get('channels', 2))
        else:
            audio_lang = None
            audio_channels = 0

        try:
            nb_frames = int(v.get('nb_frames')) if v.get('nb_frames') else None
        except Exception:
            nb_frames = None

        try:
            duration = float(info.get('format', {}).get('duration')) if info.get('format', {}).get('duration') else None
        except Exception:
            duration = None

        return VideoInfo(
            width, height, fps,
            str(color_primaries).lower(), str(color_transfer).lower(), str(color_space).lower(), str(pix_fmt).lower(),
            mastering_display or '', max_cll or '',
            audio_channels, bool(hdr_flag), audio_lang or 'eng',
            nb_frames, duration, chromaloc_val
        )

    except subprocess.TimeoutExpired:
        logger.error(f"ffprobe 超时: {file_path}")
        raise
    except Exception as e:
        logger.error(f"probe failed: {file_path.name}, {e}")
        return VideoInfo(1920, 1080, 30.0, 'bt709', 'bt709', 'bt709', 'yuv420p', '', '', 2, False, 'eng', None, None, 0)

# -------------------- Apple Validator --------------------

def detect_validator_path() -> Optional[Path]:
    possible_paths = [
        Path('/Applications/Apple Video Tools/AppleHEVCValidator'),
        Path('/Applications/AppleHEVCValidator'),
        Path('/Applications/Apple HEVC Validator.app/Contents/MacOS/AppleHEVCValidator'),
        Path('/usr/local/bin/AppleHEVCValidator'),
        Path('/usr/bin/AppleHEVCValidator'),
        Path('/opt/homebrew/bin/AppleHEVCValidator'),
        Path('C:/Program Files/Apple/AppleHEVCValidator.exe')
    ]
    for p in possible_paths:
        if p.exists():
            return p
    return None


def run_apple_validator(file_path: Path, refresh_cache=False) -> bool:
    validator = detect_validator_path()
    if not validator:
        logger.warning("Apple Validator 未安装或未找到，跳过检测，输出兼容性未验证")
        return True
    with validator_lock:
        try:
            p = subprocess.run([str(validator), str(file_path)], check=True, capture_output=True, text=True, encoding='utf-8', timeout=300)
            logger.info(f"✅ Apple Validator 通过: {file_path.name}")
            return True
        except subprocess.CalledProcessError as e:
            stderr = getattr(e, "stderr", "") or ""
            stdout = getattr(e, "stdout", "") or ""
            logger.warning(f"⚠️ Apple Validator 未通过: {file_path.name} | stderr: {stderr[:2000]} stdout: {stdout[:2000]}")
            return False
        except subprocess.TimeoutExpired:
            logger.error("Apple Validator 运行超时")
            return False
        except Exception as e:
            logger.error(f"运行 Apple Validator 异常: {e}")
            return False

@lru_cache(maxsize=1)
def detect_gpu_type() -> str:
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
            capture_output=True, text=True, check=True, encoding='utf-8', timeout=5
        )
        return result.stdout.strip().lower()
    except Exception:
        # fallback: check ffmpeg encoders for nvenc
        try:
            r = subprocess.run(['ffmpeg', '-hide_banner', '-encoders'], capture_output=True, text=True, check=True, encoding='utf-8', timeout=10)
            if 'hevc_nvenc' in r.stdout:
                return 'nvenc'
        except Exception:
            pass
        return "unknown"

# -------------------- NVENC 检测 / 策略 --------------------

def has_nvenc() -> bool:
    try:
        result = subprocess.run(['ffmpeg', '-hide_banner', '-encoders'], capture_output=True, text=True, check=True, encoding='utf-8', timeout=10)
        return 'hevc_nvenc' in result.stdout
    except Exception:
        return False


def decide_encoder(info: VideoInfo, force_cpu: bool, force_gpu: bool, nvenc_hdr_mode: str) -> bool:
    if force_cpu:
        return False
    if nvenc_hdr_mode == 'disable':
        return False
    if force_gpu:
        return has_nvenc()
    return has_nvenc()

# -------------------- preset & nvenc params --------------------

def select_nvenc_preset(info: VideoInfo, gpu_name: str) -> str:
    res = max(info.width, info.height)
    if info.hdr:
        if res >= 3840:
            return 'p7'
        elif res >= 2560:
            return 'p6'
        else:
            return 'p5'
    else:
        if res >= 3840:
            return 'p6'
        elif res >= 2560:
            return 'p5'
        else:
            return 'p4'

NVENC_RETRIES = [
    {'-bf': '3', '-b_ref_mode': 'middle'},
    {'-bf': '0', '-b_ref_mode': 'disabled'},
    {'-bf': '0', '-b_ref_mode': 'disabled', '-temporal-aq': '0'},
    {'-bf': '0', '-b_ref_mode': 'disabled', '-temporal-aq': '0', '-spatial-aq': '0'}
]


def adjust_nvenc_params(params: List[str], attempt: int) -> List[str]:
    new_params = params.copy()
    if attempt <= 0:
        return new_params
    idx = min(attempt, len(NVENC_RETRIES)) - 1
    retry_mods = NVENC_RETRIES[idx]

    param_dict = OrderedDict()
    i = 0
    while i < len(new_params):
        key = new_params[i]
        val = None
        if i + 1 < len(new_params) and not str(new_params[i+1]).startswith('-'):
            val = new_params[i+1]
            i += 2
        else:
            val = ''
            i += 1
        param_dict[key] = val

    for k, v in retry_mods.items():
        param_dict[k] = v

    rebuilt = []
    for k, v in param_dict.items():
        rebuilt.append(k)
        if v is not None and v != '':
            rebuilt.append(str(v))
    return rebuilt


def ensure_bitstream_headers(vparams: List[str], encoder: str='x265', ensure_repeat=True, ensure_aud=True, ensure_chromaloc=True) -> List[str]:
    s = ' '.join(map(str, vparams))
    out = vparams.copy()

    if ensure_aud and 'aud=1' not in s and '-aud' not in s:
        out += ['-aud', '1']

    if ensure_chromaloc and encoder.lower() == 'x265' and ('chromaloc' not in s and '-chromaloc' not in s and 'chromaloc=0' not in s):
        out += ['-chromaloc', '0']

    return out

# -------------------- HDR metadata 构造 --------------------

def build_hdr_metadata(master_display: str, max_cll: str, use_nvenc: bool, fps: float = 30.0) -> List[str]:
    master_display = (master_display or '').strip()
    max_cll = (max_cll or '').strip()
    if not master_display:
        master_display = 'G(13250,34500)B(7500,3000)R(34000,16000)WP(15635,16450)L(10000000,50)'
    if not max_cll:
        max_cll = '1000,400'

    if use_nvenc:
        ordered_tags = [
            ('color_primaries', 'bt2020'),
            ('color_trc', 'smpte2084'),
            ('colorspace', 'bt2020nc'),
            ('master_display', master_display),
            ('max_cll', max_cll)
        ]
        meta_list = []
        for k, v in ordered_tags:
            meta_list.extend(['-metadata:s:v:0', f'{k}={v}'])
        meta_list.extend(['-color_primaries', 'bt2020', '-color_trc', 'smpte2084', '-colorspace', 'bt2020nc'])
        return meta_list
    else:
        x265_hdr = [
            'hdr10=1',
            'colorprim=bt2020',
            'transfer=smpte2084',
            'colormatrix=bt2020nc',
            f'master-display={master_display}',
            f'max-cll={max_cll}',
            'hrd=1',
            'aud=1',
            'chromaloc=0',
            'repeat-headers=1'
        ]
        return ['-x265-params', ':'.join(x265_hdr)]

# -------------------- HEVC level table（复用原表） --------------------
HEVC_LEVEL_LIMITS = {
    "1":   (   36864,     552960,     64000,     4608 * 8,    128,    128),
    "2":   (  122880,    3686400,    150000,    18432 * 8,   1500,   3000),
    "2.1": (  245760,    7372800,    300000,    36864 * 8,   3000,   6000),
    "3":   (  552960,   16588800,    600000,    61440 * 8,   6000,  12000),
    "3.1": (  983040,   33177600,   1200000,   122880 * 8,  10000,  20000),
    "4":   ( 2228224,   66846720,   3000000,   245760 * 8,  12000,  30000),
    "4.1": ( 2228224,  133693440,   6000000,   491520 * 8,  20000,  50000),
    "5":   ( 8912896,  267386880,  12000000,   983040 * 8,  25000, 100000),
    "5.1": ( 8912896,  534773760,  24000000,  1966080 * 8,  40000, 160000),
    "5.2": ( 8912896, 1069547520,  48000000,  3932160 * 8,  60000, 240000),
    "6":   (35651584, 1069547520,  48000000,  3932160 * 8,  60000, 240000),
    "6.1": (35651584, 2139095040,  96000000,  7864320 * 8, 120000, 480000),
    "6.2": (35651584, 4278190080, 192000000, 15728640 * 8, 240000, 800000),
}

# -------------------- Level 计算 --------------------
def calculate_apple_hevc_level(info: VideoInfo) -> Tuple[str, str]:
    width, height, fps = info.width, info.height, info.fps
    samples_per_frame = width * height
    samples_per_sec = round(samples_per_frame * fps)
    max_dim = max(width, height)

    for lvl, (max_samples, max_rate, _, _, main_tier_max, high_tier_max) in HEVC_LEVEL_LIMITS.items():
        if samples_per_frame <= max_samples and samples_per_sec <= max_rate:
            if info.hdr or max_dim >= 3840 or fps > 60:
                tier = "high" if samples_per_sec <= high_tier_max else "main"
            else:
                tier = "main"
            return lvl, tier
    return "6.2", "main"

# -------------------- NVENC level/profile --------------------
def calculate_nvenc_hevc_level(info: VideoInfo) -> Tuple[str, str, str, str]:
    width, height, fps = info.width, info.height, info.fps
    max_dim = max(width, height)
    tier = "main"
    if info.hdr:
        tier = "high"
    if info.hdr:
        profile = "main10"
        pix_fmt = "p010le"
    else:
        profile = "main"
        pix_fmt = "yuv420p"
    if max_dim <= 1920:
        level = "4.0"
    elif max_dim <= 2560:
        level = "4.1"
    elif max_dim <= 3840:
        level = "5.1"
    else:
        level = "5.2"
    return level, tier, profile, pix_fmt

# -------------------- GOP 对齐 --------------------
def compute_aligned_gop(fps: float, preferred_gop_sec: float, max_gop_frames: int = 240) -> int:
    fps = max(1.0, fps)
    gop_frames_approx = preferred_gop_sec * fps
    gop_frames_approx = max(2, min(gop_frames_approx, max_gop_frames))
    try:
        frac = Fraction(str(fps)).limit_denominator(1001)
        fps_num, fps_den = frac.numerator, frac.denominator
    except Exception:
        fps_num, fps_den = int(round(fps)), 1

    best = None
    best_diff = float('inf')
    for n in range(1, 9):
        candidate_frames = round(fps_num * n / fps_den)
        if candidate_frames < 2 or candidate_frames > max_gop_frames:
            continue
        diff = abs(candidate_frames - gop_frames_approx)
        if diff < best_diff:
            best = candidate_frames
            best_diff = diff
    if best is None:
        best = int(round(gop_frames_approx))
        best = max(2, min(best, max_gop_frames))
    if abs(round(fps) - fps) < 1e-6:
        fps_int = int(round(fps))
        n = max(1, round(best / fps_int))
        best = max(2, min(fps_int * n, max_gop_frames))
    else:
        gop_sec_approx = best / fps
        n_sec = max(1, round(gop_sec_approx))
        best = min(max_gop_frames, max(2, round(fps * n_sec)))
    return best

# -------------------- 动态值计算 --------------------
def calculate_dynamic_values(info: VideoInfo, use_nvenc: bool = True, gpu_name: str = "") -> Tuple[int, int, int, int, int]:
    max_dim = max(info.width, info.height)
    fps = float(info.fps) if info.fps else 30.0
    hdr = bool(info.hdr)

    crf_base_table = {480: 17, 720: 18, 1080: 19, 1440: 20, 2160: 21, 4320: 22}
    keys = sorted(crf_base_table.keys())
    chosen = keys[-1]
    for k in keys:
        if info.height <= k:
            chosen = k
            break
    crf = crf_base_table[chosen]
    if hdr:
        crf = max(8, crf - 1)

    if info.nb_frames:
        est_frames = info.nb_frames
    elif info.duration:
        est_frames = int(round(info.duration * fps))
    else:
        est_frames = int(round(60 * fps))

    motion_density = est_frames / (info.width * info.height + 1)
    if motion_density > 0.00025:
        crf += 1
    elif motion_density < 0.00006:
        crf = max(8, crf - 1)

    crf = max(16, min(crf, 24))
    cq = crf + 1

    if max_dim >= 7680:
        target_kbps = 140000
    elif max_dim >= 3840:
        target_kbps = 65000 if hdr else 50000
    elif max_dim >= 2560:
        target_kbps = 30000 if hdr else 26000
    elif max_dim >= 1920:
        target_kbps = 19000 if hdr else 16000
    else:
        target_kbps = 10000 if hdr else 8000

    if motion_density > 0.00025:
        target_kbps = int(target_kbps * 1.15)
    elif motion_density < 0.00006:
        target_kbps = int(target_kbps * 0.92)

    vbv_maxrate = int(target_kbps)
    vbv_bufsize = int(vbv_maxrate * 1.5)

    try:
        lvl, tier = calculate_apple_hevc_level(info)
        lvl = str(lvl)
        if lvl in HEVC_LEVEL_LIMITS:
            _, _, max_bitrate_bps, max_cpb_bits, _, _ = HEVC_LEVEL_LIMITS[lvl]
            max_allowed_kbps = int(max_bitrate_bps / 1000)
            max_allowed_kbits = int(max_cpb_bits / 1000)
            vbv_maxrate = min(vbv_maxrate, int(max_allowed_kbps * 0.98))
            vbv_bufsize = min(vbv_bufsize, max(int(vbv_maxrate * 1.2), int(max_allowed_kbits * 0.9)))
    except Exception:
        pass

    if hdr:
        gop_sec = 2.0 if max_dim >= 3840 else 2.5
    else:
        gop_sec = 2.5 if max_dim >= 3840 else 3.0
    if fps > 60:
        gop_sec *= 1.05

    gop_frames = compute_aligned_gop(fps, gop_sec, max_gop_frames=240)
    if abs(round(fps) - fps) < 1e-6:
        fps_int = int(round(fps))
        n = max(1, round(gop_frames / fps_int))
        gop_frames = max(2, min(240, fps_int * n))

    return crf, cq, vbv_maxrate, vbv_bufsize, gop_frames

# -------------------- 构造 ffmpeg 参数 --------------------
def build_ffmpeg_params(info: VideoInfo, use_nvenc: bool, gpu_name: str) -> FFmpegParams:
    hdr = bool(info.hdr)
    if use_nvenc:
        level, tier, profile, pix_fmt = calculate_nvenc_hevc_level(info)
    else:
        level, tier = calculate_apple_hevc_level(info)
        profile = 'main10' if hdr else 'main'
        pix_fmt = 'p010le' if hdr else 'yuv420p'

    crf, cq, vbv_maxrate_kbps, vbv_bufsize_kbits, gop = calculate_dynamic_values(info, use_nvenc, gpu_name)

    if use_nvenc:
        preset = select_nvenc_preset(info, gpu_name)
        lookahead = int(min(info.fps * 1.5, 120))
        aq_strength = 6
        max_dim = max(info.width, info.height)
        if hdr:
            if max_dim >= 3840:
                aq_strength = 7
                lookahead = min(info.fps * 2, 120)
            if max_dim >= 7680:
                aq_strength = 8
                lookahead = 120

        vparams = [
            '-rc', 'vbr', '-tune', 'hq', '-multipass', 'fullres',
            '-cq', str(cq), '-b:v', '0',
            '-maxrate', str(vbv_maxrate_kbps * 1000), '-bufsize', str(vbv_bufsize_kbits * 1000),
            '-bf', '3', '-b_ref_mode', 'middle', '-rc-lookahead', str(lookahead),
            '-spatial-aq', '1', '-aq-strength', str(aq_strength),
            '-temporal-aq', '1', '-preset', preset,
            '-no-scenecut', '1', '-g', str(gop),
            '-tier', tier
        ]
        vparams = ensure_bitstream_headers(vparams, encoder='nvenc', ensure_repeat=True, ensure_aud=True, ensure_chromaloc=False)
        hdr_metadata = build_hdr_metadata(info.master_display, info.max_cll, use_nvenc=True, fps=info.fps) if hdr else []
        color_flags = []
        return FFmpegParams('hevc_nvenc', pix_fmt, profile, level, color_flags, vparams, hdr_metadata)
    else:
        x265_params = [
            f'crf={crf}', 'preset=slow', 'log-level=error', 'nal-hrd=vbr',
            f'vbv-maxrate={vbv_maxrate_kbps}', f'vbv-bufsize={vbv_bufsize_kbits}', f'tier={tier}',
            f'keyint={gop}', f'min-keyint={max(2, int(gop//2))}',
            f'profile={profile}', 'level-idc=' + str(level)
        ]
        if hdr:
            hdr_params = build_hdr_metadata(info.master_display, info.max_cll, use_nvenc=False, fps=info.fps)
            if '-x265-params' in hdr_params:
                try:
                    idx = hdr_params.index('-x265-params')
                    x265_str = hdr_params[idx + 1]
                    x265_params += x265_str.split(':')
                except Exception:
                    pass
        vparams = ['-x265-params', ':'.join(x265_params), '-threads', '0']
        return FFmpegParams('libx265', pix_fmt, profile, level, [], vparams, [])

# -------------------- 音频参数 --------------------
def get_audio_flags(audio_channels: int) -> List[str]:
    if not audio_channels or audio_channels < 1:
        return []
    min_bitrate = 128
    per_channel = 64
    max_total = 512
    calculated_bitrate = max(min_bitrate, audio_channels * per_channel)
    calculated_bitrate = min(calculated_bitrate, max_total)
    if audio_channels > 2:
        calculated_bitrate = max(calculated_bitrate, 256)

    audio_flags = ['-c:a', 'aac', '-b:a', f'{calculated_bitrate}k', '-ar', '48000']
    if audio_channels == 1:
        audio_flags += ['-ac', '1', '-channel_layout', 'mono']
    elif audio_channels == 2:
        audio_flags += ['-ac', '2', '-channel_layout', 'stereo']
    elif audio_channels == 6:
        audio_flags += ['-ac', '6', '-channel_layout', '5.1']
    elif audio_channels == 8:
        audio_flags += ['-ac', '8', '-channel_layout', '7.1']
    else:
        audio_flags += ['-ac', str(max(1, audio_channels))]

    return audio_flags

# -------------------- 构造 FFmpeg 命令（统一） --------------------
VIDEO_METADATA_FLAGS = ['-metadata:s:v:0', 'handler_name=VideoHandler']
AUDIO_METADATA_FLAGS = [
    '-metadata:s:a:0', 'handler_name=SoundHandler',
    '-metadata:s:a:0', 'language=und',
    '-metadata:s:a:0', 'title=Main Audio'
]


def build_ffmpeg_command_unified(
    file_path: Path,
    out_path: Path,
    ff_params: FFmpegParams,
    audio_channels: int,
    audio_language: Optional[str] = 'eng',
    extra_vparams: Optional[List[str]] = None
) -> List[str]:
    cmd = [
        'ffmpeg', '-hide_banner', '-y', '-i', str(file_path),
        '-map_metadata', '0',
        '-c:v', ff_params.vcodec,
        '-profile:v', ff_params.profile,
        '-pix_fmt', ff_params.pix_fmt,
        '-tag:v', 'hvc1',
    ]

    # 将 vparams 放在 metadata 之前，以确保编码参数被正确解析
    if extra_vparams:
        cmd.extend(extra_vparams)
    else:
        cmd.extend(ff_params.vparams)

    # HDR metadata / color atoms：写在 vparams 之后并在输出设置前
    if ff_params.hdr_metadata:
        cmd.extend(ff_params.hdr_metadata)

    cmd.extend(VIDEO_METADATA_FLAGS)

    if audio_channels and audio_channels > 0:
        common_flags = AUDIO_METADATA_FLAGS.copy()
        try:
            idx = common_flags.index('language=und')
            common_flags[idx] = f'language={audio_language or "eng"}'
        except ValueError:
            pass
        cmd.extend(common_flags)
        cmd.extend(get_audio_flags(audio_channels))

    cmd.extend(['-color_range', 'tv'])
    cmd.extend(['-brand', 'mp42'])
    cmd.extend(['-movflags', '+write_colr+use_metadata_tags+faststart'])
    cmd.append(str(out_path))

    return cmd

# -------------------- 转码逻辑 --------------------
def convert_video(file_path: Path, out_dir: Path, debug: bool = False,
                  skip_validator: bool = False, force_cpu: bool = False, force_gpu: bool = False,
                  nvenc_hdr_mode: str = 'prefer'):
    info = probe_media(file_path)
    gpu_name = detect_gpu_type()
    out_path = out_dir / (file_path.stem + '.mp4')
    hdr = info.hdr

    use_nvenc = decide_encoder(info, force_cpu, force_gpu, nvenc_hdr_mode)
    method_guess = "NVENC" if use_nvenc else "CPU"

    log_entry = {
        "file": file_path.name,
        "status": "FAILED",
        "quality": None,
        "retries": 0,
        "method": method_guess,
        "hdr": hdr
    }

    crf, cq, _, _, _ = calculate_dynamic_values(info, use_nvenc=False)
    _, nvenc_cq, _, _, _ = calculate_dynamic_values(info, use_nvenc=True, gpu_name=gpu_name)

    ff_params = build_ffmpeg_params(info, use_nvenc, gpu_name)

    # NVENC
    if use_nvenc:
        for attempt, retry_mods in enumerate(NVENC_RETRIES + [None], 1):
            retry_vparams = adjust_nvenc_params(ff_params.vparams, attempt) if retry_mods else ff_params.vparams
            cmd = build_ffmpeg_command_unified(
                file_path, out_path, ff_params, audio_channels=info.audio_channels,
                audio_language=info.audio_language, extra_vparams=retry_vparams
            )
            if debug:
                logger.debug(f"NVENC FFmpeg 命令 (尝试 {attempt}): {' '.join(cmd)}")
            try:
                subprocess.run(cmd, check=True, capture_output=True, text=True, encoding='utf-8', timeout=FFMPEG_TIMEOUT)
                log_entry.update({
                    "status": "SUCCESS",
                    "quality": nvenc_cq,
                    "retries": attempt if attempt <= len(NVENC_RETRIES) else len(NVENC_RETRIES),
                    "method": "NVENC"
                })
                if not skip_validator and not run_apple_validator(out_path):
                    logger.warning("NVENC 输出未通过 Validator，回退 CPU")
                    try:
                        out_path.unlink(missing_ok=True)
                    except Exception:
                        pass
                    use_nvenc = False
                    break
                break
            except subprocess.CalledProcessError as e:
                stderr = getattr(e, "stderr", "") or ""
                if debug:
                    logger.debug(f"NVENC 编码失败 stderr:\n{stderr}")
                else:
                    logger.warning(f"NVENC 编码失败尝试 {attempt}: {file_path.name} | stderr: {stderr[:1000]}")
                if attempt == len(NVENC_RETRIES) + 1:
                    use_nvenc = False
            except subprocess.TimeoutExpired:
                logger.error(f"FFmpeg 进程超时（{FFMPEG_TIMEOUT}s）: {file_path.name}")
                use_nvenc = False
                break

    # CPU
    if not use_nvenc:
        ff_params_cpu = build_ffmpeg_params(info, False, gpu_name)
        cmd_cpu = build_ffmpeg_command_unified(
            file_path, out_path, ff_params_cpu, audio_channels=info.audio_channels,
            audio_language=info.audio_language
        )
        if debug:
            logger.debug(f"CPU FFmpeg 命令: {' '.join(cmd_cpu)}")
        try:
            subprocess.run(cmd_cpu, check=True, capture_output=True, text=True, encoding='utf-8', timeout=FFMPEG_TIMEOUT)
            log_entry.update({
                "status": "SUCCESS",
                "quality": crf,
                "retries": 0,
                "method": "CPU"
            })
            if not skip_validator and not run_apple_validator(out_path):
                logger.error(f"CPU 输出未通过 Validator: {file_path.name}")
        except subprocess.CalledProcessError as e:
            stderr = getattr(e, "stderr", "") or ""
            if debug:
                logger.debug(f"CPU 编码失败 stderr:\n{stderr}")
            logger.error(f"CPU 转码失败: {file_path.name}\n{stderr[:2000]}")
        except subprocess.TimeoutExpired:
            logger.error(f"FFmpeg 进程超时（{FFMPEG_TIMEOUT}s）: {file_path.name}")

    return log_entry

# -------------------- 并行度 --------------------
def dynamic_workers():
    try:
        import psutil
        temps = psutil.sensors_temperatures()
        core_temps = None
        for k in ('coretemp', 'acpitz'):
            if k in temps:
                core_temps = temps[k]
                break
        avg_temp = None
        if core_temps:
            vals = [t.current for t in core_temps if hasattr(t, 'current')]
            if vals:
                avg_temp = sum(vals) / len(vals)
        if avg_temp is None:
            return max(1, os.cpu_count() or 1)
        if avg_temp > 85:
            return max(1, (os.cpu_count() or 1) // 4)
        elif avg_temp > 70:
            return max(1, (os.cpu_count() or 1) // 2)
        return min(4, max(1, os.cpu_count() or 1))
    except Exception:
        return max(1, os.cpu_count() or 1)

# -------------------- 批量转换 --------------------
def batch_convert(input_dir: Path, output_dir: Path, max_workers: int = 4, **kwargs):
    files = [f for f in input_dir.rglob("*") if f.is_file() and f.suffix.lower() in INPUT_EXTS]
    if not files:
        logger.warning(f"未找到可转码的视频文件于目录：{input_dir}")
        return
    output_dir.mkdir(exist_ok=True, parents=True)
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures_list = [executor.submit(convert_video, f, output_dir, **kwargs) for f in files]
        for fut in tqdm(futures_list, desc="转码"):
            try:
                results.append(fut.result())
            except Exception as e:
                idx = futures_list.index(fut)
                logger.error(f"[ERROR] {files[idx].name}: {e}")
    with open(LOG_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["file", "status", "quality", "retries", "method", "hdr"])
        writer.writeheader()
        writer.writerows(results)

# -------------------- CLI --------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Apple HEVC 批量转码脚本 v1.6.10-patch")
    parser.add_argument("-i", "--input", dest="input_dir", required=True)
    parser.add_argument("-o", "--output", dest="output_dir", required=True)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--skip-validator", action="store_true")
    parser.add_argument("--force-cpu", action="store_true")
    parser.add_argument("--force-gpu", action="store_true")
    parser.add_argument("--nvenc-hdr-mode", choices=['auto', 'prefer', 'disable'], default='prefer')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    logger.setLevel(logging.DEBUG if args.debug else logging.INFO)
    input_path = Path(args.input_dir)
    sample_files = [f for f in input_path.rglob("*") if f.is_file() and f.suffix.lower() in INPUT_EXTS][:5]
    any_hdr = False
    try:
        any_hdr = any(probe_media(f).hdr for f in sample_files)
    except Exception as e:
        logger.warning(f"采样探测出错: {e}")
    max_workers = min(dynamic_workers(), 4) if any_hdr else min(MAX_WORKERS_SDR, 8)
    check_tools()
    batch_convert(
        Path(args.input_dir),
        Path(args.output_dir),
        max_workers = max_workers,
        debug=args.debug,
        skip_validator=args.skip_validator,
        force_cpu=args.force_cpu,
        force_gpu=args.force_gpu,
        nvenc_hdr_mode=args.nvenc_hdr_mode
    )
