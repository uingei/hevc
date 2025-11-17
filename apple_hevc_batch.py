#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
============================================================
🍏 Apple HEVC 批量转码脚本 v1.6.10
============================================================

变更（相对于 v1.6.7 - 重要修改处已用 ```MODIFIED``` 标注）:
- 更科学的 VBV/bitrate 上限逻辑，并用此反推 CRF/CQ，使画质与体积更可控。
- 统一并固定 HDR metadata 顺序以提高 Apple Validator 兼容性（NVENC/CPU 都采用同一序列）。
- GOP 计算严格限制（<=240）并尽量与帧率整除，避免非整数帧间距。
- 引入 motion_density 概念（基于时长/帧数/分辨率）来微调 CRF。
- NVENC 参数微调（AQ 强度、rc-lookahead 根据帧率自适应）。
- 添加动态并行度（基于 psutil 温控采样，若 psutil 不可用回退为 cpu_count）。
- 若有修改处，均用 ```MODIFIED``` 注释块包裹说明。

请在目标机器上用真实样本验证 Apple Validator 输出。
============================================================
"""
__version__ = "1.6.10"

import subprocess
import json
import logging
import argparse
import csv
import os
import threading
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import shutil
from functools import lru_cache
from collections import OrderedDict
from fractions import Fraction
import math

# -------------------- 配置 --------------------
INPUT_EXTS = (
    '.mp4', '.mov', '.mkv', '.avi', '.wmv', '.flv', '.ts', '.m2ts', '.mts',
    '.m4v', '.webm', '.3gp', '.f4v', '.ogv', '.vob', '.mpg', '.mpeg'
)
DEFAULT_CRF = 18
# 动态计算 HDR 并行 worker，避免固定过小/过大
MAX_WORKERS_SDR = max(1, os.cpu_count() or 1)
MAX_WORKERS_HDR = min(4, max(1, (os.cpu_count() or 4) // 4))
LOG_FILE = "transcode_log.csv"

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
    master_display: str
    max_cll: str
    hdr: bool = False
    audio_language: Optional[str] = 'eng'  # 新增字段，用于继承源语言
    nb_frames: Optional[int] = None
    duration: Optional[float] = None

@dataclass
class FFmpegParams:
    vcodec: str
    pix_fmt: str
    profile: str
    level: str
    color_flags: List[str]
    vparams: List[str]
    hdr_metadata: List[str]


# -------------------- HDR 检测辅助常量 --------------------
HDR_PIXFMTS = {'yuv420p10le', 'p010le', 'yuv444p10le'}
HDR_COLOR_SPACES = {'bt2020', 'bt2020-ncl', 'bt2020nc'}
HDR_TRANSFERS = {'smpte2084', 'pq'}
HDR_PRIMARIES = {'bt2020', 'bt2020-ncl'}

def _get_tag(tags: dict, *keys, default=''):
    """在多个可能的 tag 名称中查找并返回第一个非空值（提高对不同容器的兼容性）"""
    for k in keys:
        if k in tags and tags[k]:
            return tags[k]
    return default


def is_hdr(v: dict, tags: dict) -> bool:
    color_primaries = (v.get('color_primaries') or tags.get('COLOR_PRIMARIES', '') or tags.get('color_primaries', '')).lower()
    color_transfer = (v.get('color_transfer') or tags.get('COLOR_TRANSFER', '') or tags.get('color_transfer', '')).lower()
    color_space = (v.get('color_space') or tags.get('COLOR_SPACE') or tags.get('color_space', '')).lower()
    pix_fmt = (v.get('pix_fmt') or '').lower()
    return (
        color_space in HDR_COLOR_SPACES or
        color_transfer in HDR_TRANSFERS or
        color_primaries in HDR_PRIMARIES or
        pix_fmt in HDR_PIXFMTS
    )

# -------------------- Addition: check_tools --------------------
def check_tools():
    """在脚本启动时检查必要外部工具（ffmpeg/ffprobe），nvidia-smi 仅作可选提示。"""
    from shutil import which
    missing = []
    for tool in ('ffmpeg', 'ffprobe'):
        if which(tool) is None:
            missing.append(tool)
    if missing:
        logger.error(f"缺少必要工具: {', '.join(missing)}. 请先安装并确保在 PATH 中可见。")
        raise SystemExit(1)
    # 可选提示：nvidia-smi 用于更准确检测 GPU type
    if which('nvidia-smi') is None:
        logger.debug("提示：未检测到 nvidia-smi，GPU 信息检测将退回为 'unknown' 或 ffmpeg encoder 检查。")

def probe_media(file_path: Path) -> tuple:
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_streams', '-show_format', str(file_path)],
            capture_output=True, text=True, check=True, encoding='utf-8'
        )
        info = json.loads(result.stdout)
        v = next((s for s in info.get('streams', []) if s.get('codec_type') == 'video'), None)
        if not v:
            raise ValueError("没有找到视频流")
        width = int(v.get('width') or 1920)
        height = int(v.get('height') or 1080)
        rate = v.get('avg_frame_rate') or v.get('r_frame_rate') or '30/1'
        if rate.strip() == "0/0" or not rate.strip():
            fps = 30.0
        else:
            try:
                num, den = map(int, rate.split('/'))
                fps = num / den if den else 30.0
            except Exception:
                fps = 30.0

        tags = info.get('format', {}).get('tags', {}) or {}
        color_primaries = (v.get('color_primaries') or tags.get('COLOR_PRIMARIES') or tags.get('color_primaries') or 'bt709').lower()
        color_transfer = (v.get('color_transfer') or tags.get('COLOR_TRANSFER') or tags.get('color_transfer') or 'bt709').lower()
        color_space = (v.get('color_space') or tags.get('COLOR_SPACE') or tags.get('color_space') or 'bt709').lower()
        pix_fmt = (v.get('pix_fmt') or '').lower()

        master_display = _get_tag(tags, 'master-display', 'MASTER_DISPLAY', 'master_display', 'mastering_display', default='')
        max_cll = _get_tag(tags, 'max-cll', 'MAX_CLL', 'max_cll', 'max-cll', default='')

        audio_stream = next((s for s in info.get('streams', []) if s.get('codec_type') == 'audio'), None)
        if audio_stream:
            atags = audio_stream.get('tags', {}) or {}
            audio_lang = atags.get('language') or atags.get('LANGUAGE') or 'eng'
            audio_channels = int(audio_stream.get('channels', audio_stream.get('CHANNELS', 2)))
        else:
            audio_lang = None
            audio_channels = 0

        hdr_flag = is_hdr(v, tags)

        # 尝试读取帧数和时长
        nb_frames = None
        duration = None
        try:
            nb_frames = int(v.get('nb_frames')) if v.get('nb_frames') else None
        except Exception:
            nb_frames = None
        try:
            duration = float(info.get('format', {}).get('duration')) if info.get('format', {}).get('duration') else None
        except Exception:
            duration = None

        video_info = VideoInfo(
            width, height, fps,
            color_primaries, color_transfer, color_space,
            master_display, max_cll, hdr_flag, audio_lang, nb_frames, duration
        )
        return video_info, audio_channels
    except Exception as e:
        logger.error(f"探测媒体信息失败: {file_path.name}, {e}")
        return VideoInfo(1920, 1080, 30.0, 'bt709', 'bt709', 'bt709', '', '', False, 'eng'), 2

# -------------------- Apple Validator --------------------
def detect_validator_path() -> Optional[Path]:
    possible_paths = [
        Path('/Applications/Apple Video Tools/AppleHEVCValidator'),
        Path('/usr/local/bin/AppleHEVCValidator'),
        Path('/usr/bin/AppleHEVCValidator'),
        Path('/opt/homebrew/bin/AppleHEVCValidator'),
        Path('C:/Program Files/Apple/AppleHEVCValidator.exe')
    ]
    return next((p for p in possible_paths if p.exists()), None)

# -------------------- Replacement: run_apple_validator (no lru_cache) --------------------
def run_apple_validator(file_path: Path, refresh_cache=False) -> bool:
    """
    直接运行 Validator（不缓存结果）：文件内容变化或重试时缓存会误导判断，所以不使用 lru_cache。
    返回 True = 通过；False = 未通过或发生异常。
    """
    validator = detect_validator_path()
    if not validator:
        logger.warning("Apple Validator 未安装，跳过检测，输出兼容性未验证")
        return True
    with validator_lock:
        try:
            p = subprocess.run([str(validator), str(file_path)],
                               check=True, capture_output=True, text=True, encoding='utf-8')
            logger.info(f"✅ Apple Validator 通过: {file_path.name}")
            return True
        except subprocess.CalledProcessError as e:
            stderr = getattr(e, "stderr", "") or ""
            stdout = getattr(e, "stdout", "") or ""
            logger.warning(f"⚠️ Apple Validator 未通过: {file_path.name} | stderr: {stderr[:2000]} stdout: {stdout[:2000]}")
            return False
        except Exception as e:
            logger.error(f"运行 Apple Validator 异常: {e}")
            return False

@lru_cache(maxsize=1)
def detect_gpu_type() -> str:
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
            capture_output=True, text=True, check=True, encoding='utf-8'
        )
        return result.stdout.strip().lower()
    except Exception:
        return "unknown"

# -------------------- NVENC 检测 / 策略 --------------------
def has_nvenc() -> bool:
    try:
        result = subprocess.run(['ffmpeg', '-hide_banner', '-encoders'],
                                capture_output=True, text=True, check=True, encoding='utf-8')
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

# -------------------- select_nvenc_preset (HDR/SDR + 分辨率优化) --------------------
def select_nvenc_preset(info: VideoInfo, gpu_name: str) -> str:
    res = max(info.width, info.height)
    # MODIFIED: 根据 HDR + 分辨率优化 preset
    if info.hdr:
        if res >= 3840:  # 4K HDR
            return 'p7'
        elif res >= 2560:  # 2K HDR
            return 'p6'
        else:
            return 'p5'
    else:  # SDR
        if res >= 3840:
            return 'p6'
        elif res >= 2560:
            return 'p5'
        else:
            return 'p4'

# -------------------- NVENC 重试 --------------------
NVENC_RETRIES = [
    {'-bf': '3', '-b_ref_mode': 'middle'},
    {'-bf': '0', '-b_ref_mode': 'disabled'},
    {'-bf': '0', '-b_ref_mode': 'disabled', '-temporal-aq': '0'},
    {'-bf': '0', '-b_ref_mode': 'disabled', '-temporal-aq': '0', '-spatial-aq': '0'}
]

def adjust_nvenc_params(params: List[str], attempt: int) -> List[str]:
    """
    更稳健的 NVENC 参数覆盖：
    - params: 原始参数列表（如 ['-rc','vbr','-cq','18', ...]）
    - attempt: 0 表示不修改，1..N 对应 NVENC_RETRIES 的索引
    """
    new_params = params.copy()
    if attempt <= 0:
        return new_params
    # 限制 attempt 不超过可用 retries 长度
    idx = min(attempt, len(NVENC_RETRIES)) - 1
    retry_mods = NVENC_RETRIES[idx]

    # 解析 params 为 OrderedDict（支持单 flag 情形）
    param_dict = OrderedDict()
    i = 0
    while i < len(new_params):
        key = new_params[i]
        val = None
        if i + 1 < len(new_params) and not new_params[i+1].startswith('-'):
            val = new_params[i+1]
            i += 2
        else:
            # 单 flag（没有随后的值），设为 empty string
            val = ''
            i += 1
        param_dict[key] = val

    # 应用 retry_mods（覆盖或新增）
    for k, v in retry_mods.items():
        param_dict[k] = v

    # 重建列表（恢复为 ['-key','val', ...]，忽略空值时只输出 flag）
    rebuilt = []
    for k, v in param_dict.items():
        rebuilt.append(k)
        if v is not None and v != '':
            rebuilt.append(str(v))
    return rebuilt

def ensure_bitstream_headers(vparams: List[str], encoder: str='x265', ensure_repeat=True, ensure_aud=True, ensure_chromaloc=True) -> List[str]:
    """
    确保 vparams（flat list）包含 repeat-headers / aud / chromaloc 等标志（若未出现则追加）。
    encoder: 'x265' 或 'nvenc'
    """
    s = ' '.join(map(str, vparams))
    out = vparams.copy()

    # repeat-headers 仅针对 x265
    # if ensure_repeat and 'repeat-headers' not in s and '-repeat-headers' not in s:
    #     out += ['-repeat-headers', '1']

    if ensure_aud and 'aud=1' not in s and '-aud' not in s:
        out += ['-aud', '1']

    # chromaloc 仅在 x265 下有效
    if ensure_chromaloc and encoder.lower() == 'x265' and ('chromaloc' not in s and '-chromaloc' not in s and 'chromaloc=0' not in s):
        out += ['-chromaloc', '0']

    return out

# -------------------- Replacement: build_hdr_metadata --------------------
def build_hdr_metadata(master_display: str, max_cll: str, use_nvenc: bool, fps: float = 30.0) -> List[str]:
    """
    为 NVENC 返回 -metadata 列表（按 Apple 推荐顺序并额外写 colr atom）
    为 x265 返回 ['-x265-params', '...']（已包含 repeat-headers / aud / hrd）
    chromaloc 固定为 0（你选择的 1:A），以优先兼容 Apple 播放。
    """
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
        # 额外写入 color atoms，增强兼容性（某些 Apple 解析器更依赖 colr atom）
        meta_list.extend(['-color_primaries', 'bt2020', '-color_trc', 'smpte2084', '-colorspace', 'bt2020nc'])
        return meta_list
    else:
        # x265 参数串，chromaloc=0 （你选择的）
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

# 更精确（保守值）的 HEVC level -> (max_samples, max_rate, max_bitrate_bps, max_cpb_bits, main_tier_max, high_tier_max)
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

# -------------------- Apple HEVC Level --------------------
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

# -------------------- NVENC Level/Pixel/Profile 自动适配 --------------------
def calculate_nvenc_hevc_level(info: VideoInfo) -> Tuple[str, str, str, str]:
    width, height, fps = info.width, info.height, info.fps
    max_dim = max(width, height)
    tier = "main"
    if info.hdr:
        tier = "high"
    # -------------------- MODIFIED v1.6.8: NVENC profile/pix_fmt 更稳健匹配```
    if info.hdr:
        profile = "main10"
        pix_fmt = "p010le"
    else:
        profile = "main"
        pix_fmt = "yuv420p"
    # ```END MODIFIED```
    # -------------------- MODIFIED v1.6.8: level 粗略映射```
    if max_dim <= 1920:
        level = "4.0"
    elif max_dim <= 2560:
        level = "4.1"
    elif max_dim <= 3840:
        level = "5.1"
    else:
        level = "5.2"
    return level, tier, profile, pix_fmt

def compute_aligned_gop(fps: float, preferred_gop_sec: float, max_gop_frames: int = 240):
    """
    返回以帧为单位的 gop_frames，优先对齐到整数秒（处理常见分数 fps，如 24000/1001）
    """
    try:
        frac = Fraction(str(fps)).limit_denominator(1001)
        fps_num, fps_den = frac.numerator, frac.denominator
    except Exception:
        fps_num, fps_den = int(round(fps)), 1

    gop_frames_approx = preferred_gop_sec * fps
    best = None
    best_diff = float('inf')
    # 尝试 1..8 秒整秒的 GOP 候选（可根据需要扩大）
    for n in range(1, 9):
        # candidate_frames = fps * n = (fps_num/fps_den) * n
        candidate_frames = round(fps_num * n / fps_den)
        if candidate_frames < 2:
            continue
        if candidate_frames > max_gop_frames:
            continue
        diff = abs(candidate_frames - gop_frames_approx)
        if diff < best_diff:
            best = int(candidate_frames)
            best_diff = diff
    if best is None:
        # fallback 保守值
        best = min(max(2, int(round(gop_frames_approx))), max_gop_frames)
    best = max(2, min(best, max_gop_frames))
    return best

# -------------------- Replacement: calculate_dynamic_values --------------------
def calculate_dynamic_values(info: VideoInfo, use_nvenc: bool = True, gpu_name: str = "") -> Tuple[int, int, int, int, int]:
    """
    返回 (crf, cq, vbv_maxrate_kbps, vbv_bufsize_kbits, gop_frames)
    - vbv_* 单位为 kbps / kbits（脚本其它处会 *1000 转换为 bps）
    - gop_frames 为帧数（int），尽量对齐到整数 fps 秒边界
    """
    max_dim = max(info.width, info.height)
    fps = float(info.fps) if info.fps else 30.0
    hdr = bool(info.hdr)

    # 基线 CRF（按高度桶）
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

    # 估计帧数 & 动作密度（frames / pixels）
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

    # target kbps 基于分辨率与 HDR
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

    vbv_maxrate = int(target_kbps)               # kbps
    vbv_bufsize = int(vbv_maxrate * 1.5)         # kbits

    # 精确 clamp vbv 到 HEVC level 限制（使用 HEVC_LEVEL_LIMITS）
    try:
        lvl, tier = calculate_apple_hevc_level(info)
        lvl = str(lvl)
        if lvl in HEVC_LEVEL_LIMITS:
            _, _, max_bitrate_bps, max_cpb_bits, _, _ = HEVC_LEVEL_LIMITS[lvl]
            max_allowed_kbps = int(max_bitrate_bps / 1000)
            max_allowed_kbits = int(max_cpb_bits / 1000)
            # margin 保守 98%
            vbv_maxrate = min(vbv_maxrate, int(max_allowed_kbps * 0.98))
            # vbv_bufsize 同时受限于计算出的 max_cpb 以及 vbv_maxrate 的经验比例
            vbv_bufsize = min(vbv_bufsize, max(int(vbv_maxrate * 1.2), int(max_allowed_kbits * 0.9)))
    except Exception:
        # 若 level table 解析失败，保留原先的估算值
        pass

    # GOP（秒级 -> 帧数），优先对齐到整数 fps 秒边界（Apple 播放优化）
    if hdr:
        gop_sec = 2.0 if max_dim >= 3840 else 2.5
    else:
        gop_sec = 2.5 if max_dim >= 3840 else 3.0
    if fps > 60:
        gop_sec *= 1.05

    gop_frames = compute_aligned_gop(fps, gop_sec, max_gop_frames=240)

    # 额外：若 fps 为整数，尽量使 gop 为 fps * n（再次保障）
    if abs(round(fps) - fps) < 1e-6:
        fps_int = int(round(fps))
        n = max(1, round(gop_frames / fps_int))
        gop_frames = max(2, min(240, fps_int * n))

    return crf, cq, vbv_maxrate, vbv_bufsize, gop_frames

# -------------------- Replacement: build_ffmpeg_params --------------------
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
        # 在 vparams 最终确定后，强制补齐 bitstream header flags
        vparams = ensure_bitstream_headers(vparams, encoder='nvenc', ensure_repeat=True, ensure_aud=True, ensure_chromaloc=True)

        hdr_metadata = build_hdr_metadata(info.master_display, info.max_cll, use_nvenc=True, fps=info.fps) if hdr else []
        return FFmpegParams('hevc_nvenc', pix_fmt, profile, level, [], vparams, hdr_metadata)
    else:
        # x265 参数（注意 vbv 单位为 kbps）
        x265_params = [
            f'crf={crf}', 'preset=slow', 'log-level=error', 'nal-hrd=vbr',
            f'vbv-maxrate={vbv_maxrate_kbps}', f'vbv-bufsize={vbv_bufsize_kbits}', f'tier={tier}',
            f'keyint={gop}', f'min-keyint={max(2, int(gop//2))}',
            f'profile={profile}', 'level-idc=' + str(level)
        ]
        if hdr:
            hdr_params = build_hdr_metadata(info.master_display, info.max_cll, use_nvenc=False, fps=info.fps)
            if '-x265-params' in hdr_params:
                idx = hdr_params.index('-x265-params')
                x265_str = hdr_params[idx + 1]
                x265_params += x265_str.split(':')
        # threads=0 让 libx265 自动决定合理线程数（更兼容不同机器）
        vparams = ['-x265-params', ':'.join(x265_params), '-threads', '0']
        return FFmpegParams('libx265', pix_fmt, profile, level, [], vparams, [])

# -------------------- FFmpeg 命令构建 --------------------
VIDEO_METADATA_FLAGS = ['-metadata:s:v:0', 'handler_name=VideoHandler']

AUDIO_METADATA_FLAGS = [
    '-metadata:s:a:0', 'handler_name=SoundHandler',
    '-metadata:s:a:0', 'language=und',
    '-metadata:s:a:0', 'title=Main Audio'
]

# -------------------- Replacement: get_audio_flags --------------------
def get_audio_flags(audio_channels: int) -> List[str]:
    """
    返回音频编码参数，包含明确的 -ac 和 -channel_layout（若已知）
    确保多声道至少 256k 的码率（经验）
    """
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

# -------------------- build_ffmpeg_command_unified --------------------
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
        '-pix_fmt', ff_params.pix_fmt,
        '-profile:v', ff_params.profile,
        '-tag:v', 'hvc1',
    ]

    # --- MODIFIED 开始: NVENC 使用 hdr_metadata, CPU 使用 color_flags ---
    if ff_params.hdr_metadata:
        cmd.extend(ff_params.hdr_metadata)
    #elif ff_params.color_flags:
    #    cmd.extend(ff_params.color_flags)
    # --- MODIFIED 结束 ---

    if extra_vparams:
        cmd.extend(extra_vparams)
    else:
        cmd.extend(ff_params.vparams)

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

# -------------------- 转码主逻辑 --------------------
def convert_video(file_path: Path, out_dir: Path, debug: bool = False,
                  skip_validator: bool = False, force_cpu: bool = False, force_gpu: bool = False,
                  nvenc_hdr_mode: str = 'prefer'):
    info, audio_channels = probe_media(file_path)
    gpu_name = detect_gpu_type()
    out_path = out_dir / (file_path.stem + '.mp4')
    hdr = info.hdr

    # -------------------- 决定编码器 --------------------
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

    # -------------------- 计算 CRF/CQ --------------------
    crf, cq, _, _, _ = calculate_dynamic_values(info, use_nvenc=False)
    _, nvenc_cq, _, _, _ = calculate_dynamic_values(info, use_nvenc=True, gpu_name=gpu_name)

    ff_params = build_ffmpeg_params(info, use_nvenc, gpu_name)

    # -------------------- NVENC 编码 --------------------
    if use_nvenc:
        for attempt, retry_mods in enumerate(NVENC_RETRIES + [None], 1):
            retry_vparams = adjust_nvenc_params(ff_params.vparams, attempt) if retry_mods else ff_params.vparams
            cmd = build_ffmpeg_command_unified(
                file_path, out_path, ff_params, audio_channels,
                audio_language=info.audio_language, extra_vparams=retry_vparams
            )
            if debug:
                logger.debug(f"NVENC FFmpeg 命令 (尝试 {attempt}): {' '.join(cmd)}")
            try:
                subprocess.run(cmd, check=True, capture_output=True, text=True, encoding='utf-8')
                log_entry.update({
                    "status": "SUCCESS",
                    "quality": nvenc_cq,
                    "retries": attempt if attempt <= len(NVENC_RETRIES) else len(NVENC_RETRIES),
                    "method": "NVENC"
                })
                if not skip_validator and not run_apple_validator(out_path):
                    logger.warning("NVENC 输出未通过 Validator，回退 CPU")
                    out_path.unlink(missing_ok=True)
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

    # -------------------- CPU 编码 --------------------
    if not use_nvenc:
        ff_params_cpu = build_ffmpeg_params(info, False, gpu_name)
        cmd_cpu = build_ffmpeg_command_unified(
            file_path, out_path, ff_params_cpu, audio_channels,
            audio_language=info.audio_language
        )
        if debug:
            logger.debug(f"CPU FFmpeg 命令: {' '.join(cmd_cpu)}")
        try:
            subprocess.run(cmd_cpu, check=True, capture_output=True, text=True, encoding='utf-8')
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

    return log_entry

# -------------------- dynamic_workers (基于温度/负载优化 HDR 并行度) --------------------
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
        # MODIFIED v1.6.9: HDR 4K/8K 限制 worker <= 4
        return min(4, max(1, os.cpu_count() or 1))
    except Exception:
        return max(1, os.cpu_count() or 1)

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
    parser = argparse.ArgumentParser(description="Apple HEVC 批量转码脚本 v1.6.10")
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
    any_hdr = any(probe_media(f)[0].hdr for f in sample_files)
    # ```MODIFIED v1.6.8: 使用 dynamic_workers 优化并行度选择（若 psutil 可用则基于温度）```
    max_workers = min(dynamic_workers(), 4) if any_hdr else min(MAX_WORKERS_SDR, 8)
    # 若需要用户指定 max_workers，可在未来添加 CLI 参数覆盖
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
