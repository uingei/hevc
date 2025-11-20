#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
apple_hevc_batch (revised) — aim: maximize Apple HEVC Validator compatibility

Notes:
- Requires: ffmpeg, ffprobe in PATH.
- AppleHEVCValidator is optional but strongly recommended to validate outputs.
- This rewrite focuses on: deterministic ffmpeg argument ordering, HDR metadata + colr atom consistency,
  robust nvenc fallback, better logging for validator failures.
"""
__version__ = "1.6.10-perfect-commit"

import argparse
import csv
import json
import logging
import os
import shutil
import subprocess
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import List, Optional, Tuple
from functools import lru_cache
from tqdm import tqdm

# -------------------- config --------------------
INPUT_EXTS = {
    '.mp4', '.mov', '.mkv', '.avi', '.wmv', '.flv', '.ts', '.m2ts', '.mts',
    '.m4v', '.webm', '.3gp', '.f4v', '.ogv', '.vob', '.mpg', '.mpeg'
}
LOG_FILE = "transcode_log.csv"
FFPROBE_TIMEOUT = 20
FFMPEG_TIMEOUT = 3600  # per file default (adjust if necessary)
MAX_WORKERS_SDR = max(1, os.cpu_count() or 1)
MAX_WORKERS_HDR = min(4, max(1, (os.cpu_count() or 4) // 4))

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)
validator_lock = threading.Lock()

# -------------------- dataclasses --------------------
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

# -------------------- HDR / helpers --------------------
HDR_PIXFMTS = {'yuv420p10le', 'p010le', 'yuv444p10le'}
HDR_COLOR_SPACES = {'bt2020', 'bt2020-ncl', 'bt2020nc'}
HDR_TRANSFERS = {'smpte2084', 'pq'}
HDR_PRIMARIES = {'bt2020', 'bt2020-ncl'}

def _safe_exec(cmd: List[str], timeout: int, debug: bool = False) -> subprocess.CompletedProcess:
    """
    Helper to run subprocess.run safely with captured output and timeout.
    """
    if debug:
        logger.debug("Running: " + " ".join(cmd))
    return subprocess.run(cmd, check=True, capture_output=True, text=True, encoding='utf-8', timeout=timeout)

# -------------------- tool checks --------------------
def check_tools():
    missing = []
    for tool in ('ffmpeg', 'ffprobe'):
        if shutil.which(tool) is None:
            missing.append(tool)
    if missing:
        logger.error("Missing required tools: %s. Install and ensure they are in PATH.", ", ".join(missing))
        raise SystemExit(1)

# -------------------- probe --------------------
def probe_media(file_path: Path) -> VideoInfo:
    try:
        p = subprocess.run([
            'ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_streams', '-show_format', str(file_path)
        ], capture_output=True, text=True, check=True, encoding='utf-8', timeout=FFPROBE_TIMEOUT)
        info = json.loads(p.stdout or "{}")
        streams = info.get('streams', []) or []
        fmt_tags = (info.get('format', {}) or {}).get('tags', {}) or {}

        v = next((s for s in streams if s.get('codec_type') == 'video'), None)
        if not v:
            raise ValueError("No video stream found")

        width = int(v.get('width') or 1920)
        height = int(v.get('height') or 1080)

        rate = v.get('avg_frame_rate') or v.get('r_frame_rate') or "30/1"
        if rate in ("0/0", "N/A"):
            rate = v.get("r_frame_rate") or "30/1"
        try:
            num, den = map(int, str(rate).split('/'))
            fps = num / den if den else 30.0
        except Exception:
            fps = 30.0

        vtags = v.get('tags', {}) or {}
        def first_nonempty(*keys, default=''):
            for k in keys:
                v = vtags.get(k) or fmt_tags.get(k) or v.get(k)
                if v:
                    return v
            return default

        color_primaries = str(first_nonempty('color_primaries', 'COLOR_PRIMARIES') or 'bt709').lower()
        color_transfer = str(first_nonempty('color_transfer', 'COLOR_TRANSFER', 'transfer') or 'bt709').lower()
        color_space = str(first_nonempty('color_space', 'COLOR_SPACE') or 'bt709').lower()
        pix_fmt = str(v.get('pix_fmt') or 'yuv420p').lower()

        # chroma location: normalize to 0 (left) or 1 (center)
        chroma_loc = v.get('chroma_location') or vtags.get('chroma_location') or fmt_tags.get('chroma_location') or 'left'
        chromaloc_val = 0 if str(chroma_loc).lower() in ('left','ml') else 1

        # side_data
        mastering_display = ''
        max_cll = ''
        for sd in v.get('side_data_list', []) or []:
            if sd.get('side_data_type') == 'Mastering display metadata':
                # build master-display string if available
                try:
                    mastering_display = (
                        f"G({sd['green_x']},{sd['green_y']})B({sd['blue_x']},{sd['blue_y']})"
                        f"R({sd['red_x']},{sd['red_y']})WP({sd['white_point_x']},{sd['white_point_y']})"
                        f"L({sd['max_luminance']},{sd['min_luminance']})"
                    )
                except Exception:
                    mastering_display = ''
            if sd.get('side_data_type') == 'Content light level metadata':
                try:
                    max_cll = f"{sd.get('max_content')},{sd.get('max_average')}"
                except Exception:
                    max_cll = ''

        if not mastering_display:
            mastering_display = fmt_tags.get('master-display') or fmt_tags.get('MASTER_DISPLAY') or ''
        if not max_cll:
            max_cll = fmt_tags.get('max-cll') or fmt_tags.get('MAX_CLL') or ''

        hdr_flag = any([
            any(k in color_primaries for k in ('2020','bt2020')),
            any(k in color_space for k in ('2020','bt2020')),
            pix_fmt in HDR_PIXFMTS,
            bool(mastering_display),
            any(t in color_transfer for t in ('smpte2084','pq', 'arib-std-b67'))
        ])

        a = next((s for s in streams if s.get('codec_type') == 'audio'), None)
        if a:
            atags = a.get('tags', {}) or {}
            audio_lang = (atags.get('language') or atags.get('LANGUAGE') or atags.get('lang') or atags.get('LANG') or 'eng')
            audio_channels = int(a.get('channels') or 2)
        else:
            audio_lang = 'und'
            audio_channels = 0

        nb_frames = None
        try:
            nb_frames = int(v.get('nb_frames')) if v.get('nb_frames') else None
        except Exception:
            nb_frames = None

        duration = None
        try:
            duration = float((info.get('format') or {}).get('duration')) if (info.get('format') or {}).get('duration') else None
        except Exception:
            duration = None

        return VideoInfo(width, height, fps, color_primaries, color_transfer, color_space, pix_fmt,
                         mastering_display or '', max_cll or '', audio_channels, hdr_flag, audio_lang,
                         nb_frames, duration, chromaloc_val)
    except subprocess.TimeoutExpired:
        logger.error("ffprobe timeout for %s", file_path)
        raise
    except Exception as e:
        logger.exception("probe_media failed for %s: %s", file_path, e)
        # fallback safe defaults
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
        Path('C:/Program Files/Apple/AppleHEVCValidator.exe'),
        Path.cwd() / 'AppleHEVCValidator'
    ]
    # also search PATH
    for p in possible_paths:
        if p.exists():
            return p
    exe = shutil.which('AppleHEVCValidator')
    if exe:
        return Path(exe)
    return None

def run_apple_validator(file_path: Path, timeout: int = 300) -> bool:
    validator = detect_validator_path()
    if not validator:
        logger.warning("AppleHEVCValidator not found - skipping validator step.")
        return True
    with validator_lock:
        try:
            res = subprocess.run([str(validator), str(file_path)], check=True, capture_output=True, text=True, encoding='utf-8', timeout=timeout)
            logger.info("Apple Validator passed: %s", file_path.name)
            return True
        except subprocess.CalledProcessError as e:
            stderr = getattr(e, "stderr", "") or ""
            stdout = getattr(e, "stdout", "") or ""
            logger.warning("Apple Validator failed for %s. stdout/snippet: %s | stderr/snippet: %s", file_path.name, stdout[:1000], stderr[:1000])
            return False
        except subprocess.TimeoutExpired:
            logger.error("Apple Validator timeout.")
            return False
        except Exception as e:
            logger.exception("Apple Validator execution error: %s", e)
            return False

# -------------------- GPU / NVENC detection --------------------
@lru_cache(maxsize=1)
def detect_gpu_type() -> str:
    # prefer nvidia-smi info
    nvsmi = shutil.which('nvidia-smi')
    if nvsmi:
        try:
            out = subprocess.run([nvsmi, '--query-gpu=name', '--format=csv,noheader'], capture_output=True, text=True, check=True, encoding='utf-8', timeout=5)
            return out.stdout.strip().lower()
        except Exception:
            pass
    # fallback to ffmpeg encoders list
    try:
        out = subprocess.run(['ffmpeg', '-hide_banner', '-encoders'], capture_output=True, text=True, check=True, encoding='utf-8', timeout=10)
        if 'hevc_nvenc' in out.stdout:
            return 'nvenc'
    except Exception:
        pass
    return "unknown"

def has_nvenc() -> bool:
    try:
        out = subprocess.run(['ffmpeg', '-hide_banner', '-encoders'], capture_output=True, text=True, check=True, encoding='utf-8', timeout=10)
        return 'hevc_nvenc' in out.stdout
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

# -------------------- level calculations --------------------
HEVC_LEVEL_LIMITS = {
    "1":   (   36864,     552960,     64000,     4608 * 8,    128,    128),
    "2":   (  122880,    3686400,    150000,    18432 * 8,   1500,   3000),
    "2.1": (  245760,    7372800,    300000,    36864 * 8,   3000,   6000),
    "3":   (  552960,   16588800,    600000,    61440 * 8,   6000,  12000),
    "3.1": (  983040,   33177600,   1200000,   122880 * 8,  10000,  20000),
    "4":   ( 2228224,   66846720,   3000000,   245760 * 8,  12000,  30000),
    "4.1": ( 2228224,  133693440,   6000000,  491520 * 8,  20000,  50000),
    "5":   ( 8912896,  267386880,  12000000,   983040 * 8,  25000, 100000),
    "5.1": ( 8912896,  534773760,  24000000,  1966080 * 8,  40000, 160000),
    "5.2": ( 8912896, 1069547520,  48000000,  3932160 * 8,  60000, 240000),
    "6":   (35651584, 1069547520,  48000000,  3932160 * 8,  60000, 240000),
    "6.1": (35651584, 2139095040,  96000000,  7864320 * 8, 120000, 480000),
    "6.2": (35651584, 4278190080, 192000000, 15728640 * 8, 240000, 800000),
}

def calculate_apple_hevc_level(info: VideoInfo) -> Tuple[str, str]:
    width, height, fps = info.width, info.height, info.fps
    samples_per_frame = width * height
    samples_per_sec = round(samples_per_frame * fps)
    max_dim = max(width, height)
    for lvl, (max_samples, max_rate, _, _, main_tier_max, high_tier_max) in HEVC_LEVEL_LIMITS.items():
        if samples_per_frame <= max_samples and samples_per_sec <= max_rate:
            # choose tier: prefer high for HDR or very large resolution / high fps
            if info.hdr or max_dim >= 3840 or fps > 60:
                # choose high if within high tier limits, else main
                tier = "high" if samples_per_sec <= high_tier_max else "main"
            else:
                tier = "main"
            return lvl, tier
    return "6.2", "main"

def calculate_nvenc_hevc_level(info: VideoInfo) -> Tuple[str, str, str, str]:
    max_dim = max(info.width, info.height)
    tier = "high" if info.hdr else "main"
    profile = "main10" if info.hdr else "main"
    pix_fmt = "p010le" if info.hdr else "yuv420p"
    if max_dim <= 1920:
        level = "4.0"
    elif max_dim <= 2560:
        level = "4.1"
    elif max_dim <= 3840:
        level = "5.1"
    else:
        level = "5.2"
    return level, tier, profile, pix_fmt

# -------------------- GOP alignment --------------------
def compute_aligned_gop(fps: float, preferred_gop_sec: float, max_gop_frames: int = 240) -> int:
    fps = max(1.0, float(fps))
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

# -------------------- dynamic values --------------------
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

# -------------------- ensure headers --------------------
def ensure_bitstream_headers(vparams: List[str], encoder: str='x265', ensure_repeat=True, ensure_aud=True, ensure_chromaloc=True) -> List[str]:
    """
    Ensure AUD / chromaloc / repeat-headers exist in the encoder params.
    For x265: these are x265 params (no leading dashes, e.g. 'chromaloc=0').
    For nvenc: ffmpeg flags like '-aud 1' are acceptable.
    """
    s = ' '.join(map(str, vparams))
    out = list(vparams)
    if encoder.lower().startswith('nvenc'):
        if ensure_aud and '-aud' not in s and 'aud=1' not in s:
            out += ['-aud', '1']
        # nvenc chromaloc handled via ffmpeg args '-chroma_sample_location' if desired
    else:
        # x265 case: inject into -x265-params string rather than as separate -chromaloc flags
        # caller must ensure -x265-params exists and is a single argument.
        return out
    return out

# -------------------- build hdr metadata --------------------
def build_hdr_metadata(master_display: str, max_cll: str, use_nvenc: bool, info: VideoInfo) -> List[str]:
    master_display = (master_display or '').strip()
    max_cll = (max_cll or '').strip()
    if not master_display:
        master_display = 'G(13250,34500)B(7500,3000)R(34000,16000)WP(15635,16450)L(10000000,50)'
    if not max_cll:
        max_cll = '1000,400'

    if use_nvenc:
        # For NVENC: write both metadata tags and color flags to ensure colr atom and tags match.
        meta_list = [
            '-metadata:s:v:0', f'color_primaries=bt2020',
            '-metadata:s:v:0', f'color_trc=smpte2084',
            '-metadata:s:v:0', f'colorspace=bt2020nc',
            '-metadata:s:v:0', f'master_display={master_display}',
            '-metadata:s:v:0', f'max_cll={max_cll}'
        ]
        # Also provide ffmpeg color flags which will generate colr atom
        meta_list += ['-color_primaries', 'bt2020', '-color_trc', 'smpte2084', '-colorspace', 'bt2020nc']
        # ensure content light level atoms via metadata tags (some ffmpeg builds will write max_cll into CLL atom)
        return meta_list
    else:
        # For x265: form x265 param string that includes mastering info
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
        # return as a -x265-params argument handled by build_ffmpeg_params
        return ['-x265-params', ':'.join(x265_hdr)]

# -------------------- choose params --------------------
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
    # convert param list to dict sequence preserving order, then apply retry modifications
    param_dict = OrderedDict()
    i = 0
    new_params = params.copy()
    while i < len(new_params):
        k = new_params[i]
        v = ''
        if i + 1 < len(new_params) and not str(new_params[i+1]).startswith('-'):
            v = str(new_params[i+1])
            i += 2
        else:
            i += 1
        param_dict[k] = v
    if attempt <= 0:
        # no changes
        rebuilt = []
        for k, v in param_dict.items():
            rebuilt.append(k)
            if v != '':
                rebuilt.append(v)
        return rebuilt
    idx = min(max(1, attempt), len(NVENC_RETRIES)) - 1
    retry_mods = NVENC_RETRIES[idx]
    for k, v in retry_mods.items():
        param_dict[k] = v
    rebuilt = []
    for k, v in param_dict.items():
        rebuilt.append(k)
        if v != '':
            rebuilt.append(v)
    return rebuilt

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

        # Note: NVENC parameters are provided as ffmpeg encoder options (pairs)
        vparams = [
            '-rc', 'vbr',
            '-tune', 'hq',
            '-multipass', 'fullres',
            '-cq', str(cq),
            '-b:v', '0',
            '-maxrate', str(vbv_maxrate_kbps * 1000),
            '-bufsize', str(vbv_bufsize_kbits * 1000),
            '-bf', '3',
            '-b_ref_mode', 'middle',
            '-rc-lookahead', str(lookahead),
            '-spatial-aq', '1',
            '-aq-strength', str(aq_strength),
            '-temporal-aq', '1',
            '-preset', preset,
            '-no-scenecut', '1',
            '-g', str(gop),
            '-tier', tier
        ]
        vparams = ensure_bitstream_headers(vparams, encoder='nvenc', ensure_repeat=True, ensure_aud=True, ensure_chromaloc=False)
        hdr_metadata = build_hdr_metadata(info.master_display, info.max_cll, use_nvenc=True, info=info) if hdr else []
        color_flags = []  # populated later in command builder
        return FFmpegParams('hevc_nvenc', pix_fmt, profile, level, color_flags, vparams, hdr_metadata)
    else:
        # x265 params assembled into one -x265-params argument
        x265_params = [
            f'crf={crf}',
            'preset=slow',
            'log-level=error',
            'nal-hrd=vbr',
            f'vbv-maxrate={vbv_maxrate_kbps}',
            f'vbv-bufsize={vbv_bufsize_kbits}',
            f'tier={tier}',
            f'keyint={gop}',
            f'min-keyint={max(2, int(gop//2))}',
            f'profile={profile}',
            f'level-idc={level}'
        ]
        if hdr:
            hdr_params = build_hdr_metadata(info.master_display, info.max_cll, use_nvenc=False, info=info)
            # hdr_params is ['-x265-params', '...'] — merge the rhs into x265_params
            if len(hdr_params) == 2 and hdr_params[0] == '-x265-params':
                x265_params += hdr_params[1].split(':')
        vparams = ['-x265-params', ':'.join(x265_params), '-threads', '0']
        return FFmpegParams('libx265', pix_fmt, profile, level, [], vparams, [])

# -------------------- audio flags --------------------
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

# -------------------- build ffmpeg command --------------------
VIDEO_METADATA_FLAGS = ['-metadata:s:v:0', 'handler_name=VideoHandler']
AUDIO_METADATA_TEMPLATE = [
    '-metadata:s:a:0', 'handler_name=SoundHandler',
    '-metadata:s:a:0', 'language={lang}',
    '-metadata:s:a:0', 'title=Main Audio'
]

def build_ffmpeg_command_unified(
    file_path: Path,
    out_path: Path,
    ff_params: FFmpegParams,
    audio_channels: int,
    audio_language: Optional[str] = 'eng',
    extra_vparams: Optional[List[str]] = None,
    debug: bool = False
) -> List[str]:
    """
    Build ffmpeg command with deterministic ordering to maximize Apple's validator acceptance:
    ordering:
      ffmpeg -hide_banner -y -i input -map 0 -c:v <encoder> -profile:v ... -pix_fmt ... -tag:v hvc1
        [encoder params / x265-params or nvenc vparams]
        [hdr metadata as -metadata:s:v:0 ... and also -color_primaries ... flags]
        [video metadata flags]
        [audio metadata & audio codec flags]
        [-color_range tv -brand mp42 -movflags +write_colr+use_metadata_tags+faststart]
        output
    """
    cmd = ['ffmpeg', '-hide_banner', '-y', '-i', str(file_path)]

    # map everything from source; this preserves attachments/subtitles/chapters unless user wants to strip
    cmd += ['-map', '0']

    # video encoder/profiles
    cmd += ['-c:v', ff_params.vcodec]
    # profile setting is valid for many encoders; place early
    cmd += ['-profile:v', ff_params.profile]
    # pixel format for output
    cmd += ['-pix_fmt', ff_params.pix_fmt]
    # tag as hvc1 (Apple often expects hvc1 box)
    cmd += ['-tag:v', 'hvc1']

    # add encoder-specific parameters (x265 or nvenc)
    if extra_vparams:
        cmd += list(extra_vparams)
    else:
        cmd += list(ff_params.vparams)

    # hdr metadata & color atoms: ensure we write both metadata tags and color flags so ffmpeg writes colr atom
    if ff_params.hdr_metadata:
        cmd += list(ff_params.hdr_metadata)

    # write explicit color metadata flags if present (helpful for colr atom)
    # sometimes ff_params.color_flags may be empty: for nvenc hdr we used hdr_metadata to include color flags
    if ff_params.color_flags:
        cmd += ff_params.color_flags

    # set chroma sample location explicitly if needed to avoid ambiguity
    cmd += ['-chroma_sample_location', 'left']  # explicit; Apple often expects chroma loc defined

    # standard video metadata
    cmd += VIDEO_METADATA_FLAGS

    # audio
    if audio_channels and audio_channels > 0:
        audio_meta = [s for s in AUDIO_METADATA_TEMPLATE]
        # substitute language placeholder
        for i, t in enumerate(audio_meta):
            if '{lang}' in t:
                audio_meta[i] = t.format(lang=(audio_language or 'und'))
        cmd += audio_meta
        cmd += get_audio_flags(audio_channels)
    else:
        # if no audio, remove audio streams (explicit) to avoid creating empty audio tracks
        cmd += ['-an']

    # set color range and container flags
    cmd += ['-color_range', 'tv']
    cmd += ['-brand', 'mp42']
    # movflags: write colr atom and use metadata tags (helps Apple Validator); faststart to move moov to front
    cmd += ['-movflags', '+write_colr+use_metadata_tags+faststart']

    # ensure deterministic atom ordering by forcing metadata mapping; (ffmpeg orders as given)
    cmd += [str(out_path)]

    if debug:
        logger.debug("FFmpeg cmd: %s", " ".join(cmd))
    return cmd

# -------------------- conversion --------------------
NVENC_RETRIES_COUNT = len(NVENC_RETRIES)

def convert_video(file_path: Path, out_dir: Path, debug: bool = False,
                  skip_validator: bool = False, force_cpu: bool = False, force_gpu: bool = False,
                  nvenc_hdr_mode: str = 'prefer') -> dict:
    info = probe_media(file_path)
    gpu_name = detect_gpu_type()
    out_path = out_dir / (file_path.stem + '.mp4')
    hdr = info.hdr

    use_nvenc = decide_encoder(info, force_cpu, force_gpu, nvenc_hdr_mode)
    method_guess = "NVENC" if use_nvenc else "CPU"

    log_entry = {"file": file_path.name, "status": "FAILED", "quality": None, "retries": 0, "method": method_guess, "hdr": hdr}

    # precompute quality estimates
    crf, cq, _, _, _ = calculate_dynamic_values(info, use_nvenc=False)
    _, nvenc_cq, _, _, _ = calculate_dynamic_values(info, use_nvenc=True, gpu_name=gpu_name)

    ff_params = build_ffmpeg_params(info, use_nvenc, gpu_name)

    # NVENC path with retry adjustments
    if use_nvenc:
        last_exc = None
        for attempt in range(1, NVENC_RETRIES_COUNT + 2):  # include final attempt without extra mods
            retry_vparams = adjust_nvenc_params(ff_params.vparams, attempt-1) if attempt > 1 else ff_params.vparams
            cmd = build_ffmpeg_command_unified(file_path, out_path, ff_params, audio_channels=info.audio_channels,
                                               audio_language=info.audio_language, extra_vparams=retry_vparams, debug=debug)
            try:
                subprocess.run(cmd, check=True, capture_output=True, text=True, encoding='utf-8', timeout=FFMPEG_TIMEOUT)
                # success; run validator if requested
                log_entry.update({"status": "SUCCESS", "quality": nvenc_cq, "retries": attempt-1, "method": "NVENC"})
                if not skip_validator:
                    valid = run_apple_validator(out_path)
                    if not valid:
                        logger.warning("Validator failed for NVENC output; will fallback to CPU encode for %s", file_path.name)
                        # delete the output to avoid confusion
                        try:
                            out_path.unlink(missing_ok=True)
                        except Exception:
                            pass
                        use_nvenc = False
                        break
                # success and validated (or skipped)
                break
            except subprocess.CalledProcessError as e:
                last_exc = e
                stderr = getattr(e, "stderr", "") or ""
                stdout = getattr(e, "stdout", "") or ""
                if debug:
                    logger.debug("NVENC encoding error: stdout=%s\nstderr=%s", stdout[:2000], stderr[:2000])
                else:
                    logger.warning("NVENC attempt %d failed for %s: %s", attempt, file_path.name, stderr[:1000])
                # if exhausted attempts, fallback
                if attempt >= NVENC_RETRIES_COUNT + 1:
                    logger.info("Exhausted NVENC retries; falling back to CPU for %s", file_path.name)
                    use_nvenc = False
            except subprocess.TimeoutExpired:
                logger.error("FFmpeg timeout for %s (NVENC).", file_path.name)
                use_nvenc = False
                break
            except Exception as e:
                logger.exception("Unexpected NVENC error for %s: %s", file_path.name, e)
                use_nvenc = False
                break

    # CPU path
    if not use_nvenc:
        ff_params_cpu = build_ffmpeg_params(info, False, gpu_name)
        cmd_cpu = build_ffmpeg_command_unified(file_path, out_path, ff_params_cpu, audio_channels=info.audio_channels,
                                               audio_language=info.audio_language, debug=debug)
        try:
            subprocess.run(cmd_cpu, check=True, capture_output=True, text=True, encoding='utf-8', timeout=FFMPEG_TIMEOUT)
            log_entry.update({"status": "SUCCESS", "quality": crf, "retries": 0, "method": "CPU"})
            if not skip_validator:
                if not run_apple_validator(out_path):
                    logger.error("CPU output did not pass Apple Validator: %s", file_path.name)
        except subprocess.CalledProcessError as e:
            stderr = getattr(e, "stderr", "") or ""
            logger.error("CPU transcode failed for %s: %s", file_path.name, stderr[:2000])
        except subprocess.TimeoutExpired:
            logger.error("FFmpeg timeout for %s (CPU).", file_path.name)
        except Exception as e:
            logger.exception("Unexpected CPU transcode error for %s: %s", file_path.name, e)

    return log_entry

# -------------------- concurrency helpers --------------------
def dynamic_workers():
    try:
        import psutil
        temps = psutil.sensors_temperatures()
        core_temps = None
        for k in ('coretemp', 'acpitz'):
            if k in temps:
                core_temps = temps[k]
                break
        if not core_temps:
            return max(1, os.cpu_count() or 1)
        vals = [t.current for t in core_temps if hasattr(t, 'current')]
        if not vals:
            return max(1, os.cpu_count() or 1)
        avg_temp = sum(vals) / len(vals)
        if avg_temp > 85:
            return max(1, (os.cpu_count() or 1) // 4)
        elif avg_temp > 70:
            return max(1, (os.cpu_count() or 1) // 2)
        return min(4, max(1, os.cpu_count() or 1))
    except Exception:
        return max(1, os.cpu_count() or 1)

# -------------------- batch convert --------------------
def batch_convert(input_dir: Path, output_dir: Path, max_workers: int = 4, **kwargs):
    files = [f for f in input_dir.rglob("*") if f.is_file() and f.suffix.lower() in INPUT_EXTS]
    if not files:
        logger.warning("No input videos found in %s", input_dir)
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(convert_video, f, output_dir, **kwargs) for f in files]
        for fut in tqdm(futures, desc="Transcoding"):
            try:
                results.append(fut.result())
            except Exception as e:
                logger.exception("Unexpected error in worker: %s", e)
    # write CSV log
    try:
        with open(LOG_FILE, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=["file", "status", "quality", "retries", "method", "hdr"])
            writer.writeheader()
            writer.writerows(results)
    except Exception:
        logger.exception("Failed to write log file")

# -------------------- CLI --------------------
def parse_args():
    p = argparse.ArgumentParser(description="Apple HEVC batch transcode (aim: Apple HEVC Validator compatibility)")
    p.add_argument("-i", "--input", required=True, dest="input_dir")
    p.add_argument("-o", "--output", required=True, dest="output_dir")
    p.add_argument("--debug", action="store_true")
    p.add_argument("--skip-validator", action="store_true")
    p.add_argument("--force-cpu", action="store_true")
    p.add_argument("--force-gpu", action="store_true")
    p.add_argument("--nvenc-hdr-mode", choices=['auto', 'prefer', 'disable'], default='prefer')
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    logger.setLevel(logging.DEBUG if args.debug else logging.INFO)

    input_path = Path(args.input).expanduser()
    output_path = Path(args.output).expanduser()
    if not input_path.exists():
        logger.error("Input path does not exist: %s", input_path)
        raise SystemExit(1)

    # sample a few files to detect HDR presence and set reasonable worker counts
    sample_files = [f for f in input_path.rglob("*") if f.is_file() and f.suffix.lower() in INPUT_EXTS][:6]
    any_hdr = False
    try:
        any_hdr = any(probe_media(s).hdr for s in sample_files)
    except Exception as e:
        logger.warning("Sampling probe failed: %s", e)

    max_workers = min(dynamic_workers(), 4) if any_hdr else min(MAX_WORKERS_SDR, 8)
    check_tools()
    batch_convert(input_path, output_path, max_workers=max_workers,
                  debug=args.debug, skip_validator=args.skip_validator,
                  force_cpu=args.force_cpu, force_gpu=args.force_gpu,
                  nvenc_hdr_mode=args.nvenc_hdr_mode)
