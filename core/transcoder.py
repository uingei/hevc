# core/transcoder.py
# coding: utf-8
"""
核心转码模块（基于原脚本的 convert_video() 重构）
提供独立、可测试的 convert_video() 接口。
"""

import logging
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Callable, Dict, Any, Tuple
from functools import lru_cache
from collections import OrderedDict
from fractions import Fraction
import math

from core.probe import probe_media, VideoInfo
from core.utils import has_nvenc, detect_gpu_type, build_hdr_metadata
import config

logger = logging.getLogger(__name__)

@dataclass
class FFmpegParams:
    vcodec: str
    pix_fmt: str
    profile: str
    level: str
    color_flags: List[str]
    vparams: List[str]
    hdr_metadata: List[str]

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

def decide_encoder(info: VideoInfo, force_cpu: bool, force_gpu: bool) -> bool:
    if force_cpu:
        return False
    if force_gpu:
        return has_nvenc()
    return has_nvenc()

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
    """
    更稳健的 NVENC 参数覆盖：
    - params: 原始参数列表（如 ['-rc','vbr','-cq','18', ...]）
    - attempt: 0 表示不修改，1..N 对应 NVENC_RETRIES 的索引
    """
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
        if i + 1 < len(new_params) and not new_params[i+1].startswith('-'):
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

def compute_aligned_gop(fps: float, preferred_gop_sec: float, max_gop_frames: int = 240) -> int:
    """
    返回 GOP 帧数，优先对齐到整数秒。
    - fps: 视频帧率（可为非整数，如 23.976, 29.97, 59.94）
    - preferred_gop_sec: 首选 GOP 秒数
    - max_gop_frames: 最大允许 GOP 帧数
    """
    # 安全保护
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

    # 尝试 1..8 秒整秒候选 GOP
    for n in range(1, 9):
        candidate_frames = round(fps_num * n / fps_den)
        if candidate_frames < 2 or candidate_frames > max_gop_frames:
            continue
        diff = abs(candidate_frames - gop_frames_approx)
        if diff < best_diff:
            best = candidate_frames
            best_diff = diff

    # fallback 保守值
    if best is None:
        best = int(round(gop_frames_approx))
        best = max(2, min(best, max_gop_frames))

    # 整数 FPS 再对齐（原逻辑）
    if abs(round(fps) - fps) < 1e-6:
        fps_int = int(round(fps))
        n = max(1, round(best / fps_int))
        best = max(2, min(fps_int * n, max_gop_frames))

    # 分数 FPS 再对齐（增强版，NTSC 29.97/59.94 等）
    else:
        # 尽量对齐到整数秒
        gop_sec_approx = best / fps  # 当前 GOP 秒数
        n_sec = max(1, round(gop_sec_approx))
        best = min(max_gop_frames, max(2, round(fps * n_sec)))

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

VIDEO_METADATA_FLAGS = ['-metadata:s:v:0', 'handler_name=VideoHandler']

AUDIO_METADATA_FLAGS = [
    '-metadata:s:a:0', 'handler_name=SoundHandler',
    '-metadata:s:a:0', 'language=und',
    '-metadata:s:a:0', 'title="Main Audio"'
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

def build_ffmpeg_command(
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

    if ff_params.hdr_metadata:
        cmd.extend(ff_params.hdr_metadata)
    #elif ff_params.color_flags:
    #    cmd.extend(ff_params.color_flags)

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

def run_ffmpeg(cmd: List[str], progress_callback: Optional[Callable[[str, int, int], None]], file_name: str, total_frames: int, stop_event: Optional[threading.Event] = None, debug: bool = False) -> Tuple[int, str]:
    """
    运行 ffmpeg 命令，逐行读取 stdout 以解析进度（frame=）。
    返回: (return_code, stdout_text)
    """
    if debug:
        logger.debug("运行 FFmpeg 命令: %s", " ".join(cmd))
    try:
        output_lines = []
        with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace') as proc:
            frame = 0
            for line in proc.stdout:
                output_lines.append(line)

                if stop_event and stop_event.is_set():
                    try:
                        proc.terminate()
                    except Exception:
                        pass
                    return 1, "".join(output_lines)

                if "frame=" in line:
                    try:
                        part = line.strip().split("frame=")[-1].split()[0]
                        frame = int(part)
                    except Exception:
                        pass

                    if progress_callback:
                        try:
                            progress_callback(file_name, frame, total_frames)
                        except Exception:
                            logger.debug("progress_callback 抛异常", exc_info=True)

            ret = proc.wait()
            return ret, "".join(output_lines)
    except Exception as e:
        logger.error("运行 ffmpeg 失败: %s — %s", cmd[:3], e)
        return 1, str(e)

def convert_video(
    file_path: Path,
    out_dir: Path,
    progress_callback: Optional[Callable[[str, int, int], None]] = None,
    debug: bool = False,
    skip_validator: bool = False,
    force_cpu: bool = False,
    force_gpu: bool = False,
    stop_event: Optional[threading.Event] = None
) -> Dict[str, Any]:
    """
    转码单个文件（从原脚本迁移过来并保持兼容性）。
    返回字典: {"file","status","crf","retries","method","hdr"}
    stop_event: 可选 threading.Event，用于外部请求取消。
    """
    info= probe_media(file_path)
    gpu_name = detect_gpu_type()
    out_path = out_dir / (file_path.stem + '.mp4')
    hdr = info.hdr

    use_nvenc = decide_encoder(info, force_cpu, force_gpu)
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
    total_frames = max(1, int(info.duration * info.fps)) if info.duration and info.fps else 1

    if use_nvenc:
        for attempt, retry_mods in enumerate(NVENC_RETRIES + [None], 1):
            retry_vparams = adjust_nvenc_params(ff_params.vparams, attempt) if retry_mods else ff_params.vparams
            cmd = build_ffmpeg_command(
                file_path, out_path, ff_params, audio_channels=info.audio_channels,
                audio_language=info.audio_language, extra_vparams=retry_vparams
            )
            if debug:
                logger.debug(f"NVENC FFmpeg 命令 (尝试 {attempt}): {' '.join(cmd)}")
            ret, stderr = run_ffmpeg(cmd, progress_callback, file_path.name, total_frames, stop_event=stop_event, debug=debug)
            if ret == 0:
                log_entry["status"] = "SUCCESS"
                log_entry["quality"] = nvenc_cq
                log_entry["retries"] = attempt if attempt <= len(NVENC_RETRIES) else len(NVENC_RETRIES)
                log_entry["method"] = "NVENC"
                break
            else:
                if debug:
                    logger.debug(f"NVENC 编码失败 stderr:\n{stderr}")
                else:
                    logger.warning(f"NVENC 编码失败尝试 {attempt}: {file_path.name} | stderr: {stderr[:1000]}")
                if attempt == len(NVENC_RETRIES) + 1:
                    use_nvenc = False

    if not use_nvenc:
        ff_params_cpu = build_ffmpeg_params(info, False, gpu_name)
        cmd_cpu = build_ffmpeg_command(
            file_path, out_path, ff_params_cpu, audio_channels=info.audio_channels,
            audio_language=info.audio_language
        )
        if debug:
            logger.debug(f"CPU FFmpeg 命令: {' '.join(cmd_cpu)}")
        ret_cpu, stderr_cpu = run_ffmpeg(cmd_cpu, progress_callback, file_path.name, total_frames, stop_event=stop_event, debug=debug)
        if ret_cpu == 0:
            log_entry["status"] = "SUCCESS"
            log_entry["quality"] = crf
            log_entry["retries"] = 0
            log_entry["method"] = "CPU"
        else:
            if debug:
                logger.debug(f"CPU 编码失败 stderr:\n{stderr_cpu}")
            else:
                logger.error(f"CPU 转码失败: {file_path.name}\n{stderr_cpu[:2000]}")

    # 如果成功并且没有跳过验证，运行 Apple Validator（若可用）
    if log_entry["status"] == "SUCCESS" and not skip_validator:
        try:
            run_apple_validator(out_path)
        except Exception:
            # Validator 的失败不应影响主流程
            logger.debug("Apple Validator 执行时抛出异常", exc_info=True)

    # 若外部触发了取消，则标记为 FAILED（但如果已经成功过则保持 SUCCESS）
    if stop_event and stop_event.is_set() and log_entry["status"] != "SUCCESS":
        log_entry["status"] = "CANCELLED"

    # 如果有 progress_callback，保证把进度置为完成（便于 GUI 完成显示）
    if progress_callback:
        try:
            progress_callback(file_path.name, total_frames, total_frames)
        except Exception:
            logger.debug("progress_callback 在结束时抛出异常", exc_info=True)

    return log_entry
