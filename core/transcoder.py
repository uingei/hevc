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

from core.probe import is_hdr, probe_video, probe_audio_channels, VideoInfo
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

@lru_cache(maxsize=128)
def run_apple_validator(file_path: Path, refresh_cache=False) -> bool:
    if refresh_cache:
        run_apple_validator.cache_clear()
    validator = detect_validator_path()
    if not validator:
        logger.debug("Apple Validator 未安装，跳过检测。")
        return True
    with validator_lock:
        try:
            subprocess.run([str(validator), str(file_path)], check=True, capture_output=True, text=True, encoding='utf-8')
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

def calculate_apple_hevc_level(info: VideoInfo) -> Tuple[str, str]:
    width, height, fps = info.width, info.height, info.fps
    samples_per_frame = width * height
    samples_per_sec = round(samples_per_frame * fps)
    max_dim = max(width, height)
    HEVC_LEVELS = [
        {"level": "1", "max_samples": 36864, "max_rate": 552960, "main": 128, "high": 128},
        {"level": "2", "max_samples": 122880, "max_rate": 3686400, "main": 1500, "high": 3000},
        {"level": "2.1", "max_samples": 245760, "max_rate": 7372800, "main": 3000, "high": 6000},
        {"level": "3", "max_samples": 552960, "max_rate": 16588800, "main": 6000, "high": 12000},
        {"level": "3.1", "max_samples": 983040, "max_rate": 33177600, "main": 10000, "high": 20000},
        {"level": "4", "max_samples": 2228224, "max_rate": 66846720, "main": 12000, "high": 30000},
        {"level": "4.1", "max_samples": 2228224, "max_rate": 133693440, "main": 20000, "high": 50000},
        {"level": "5", "max_samples": 8912896, "max_rate": 267386880, "main": 25000, "high": 100000},
        {"level": "5.1", "max_samples": 8912896, "max_rate": 534773760, "main": 40000, "high": 160000},
        {"level": "5.2", "max_samples": 8912896, "max_rate": 1069547520, "main": 60000, "high": 240000},
        {"level": "6", "max_samples": 35651584, "max_rate": 1069547520, "main": 60000, "high": 240000},
        {"level": "6.1", "max_samples": 35651584, "max_rate": 2139095040, "main": 120000, "high": 480000},
        {"level": "6.2", "max_samples": 35651584, "max_rate": 4278190080, "main": 240000, "high": 800000},
    ]
    for lvl in HEVC_LEVELS:
        if samples_per_frame <= lvl["max_samples"] and samples_per_sec <= lvl["max_rate"]:
            if info.hdr or max_dim >= 3840 or fps > 60:
                tier = "high" if samples_per_sec <= lvl["high"] else "main"
            else:
                tier = "main"
            return lvl["level"], tier
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

def calculate_dynamic_values(info: VideoInfo, use_nvenc: bool = True, gpu_name: str = "") -> Tuple[int, int, int, int, int]:
    max_dim = max(info.width, info.height)
    fps = info.fps
    hdr = info.hdr
    level, tier = calculate_apple_hevc_level(info)

    crf_base_table = {
        480: 17,
        720: 18,
        1080: 19,
        1440: 20,
        2160: 21,
        4320: 22
    }

    sorted_keys = sorted(crf_base_table.keys())
    closest_res = sorted_keys[0]
    for k in sorted_keys:
        if info.height <= k:
            closest_res = k
            break
    crf_sdr = crf_base_table[closest_res]
    crf = crf_sdr - 1 if hdr else crf_sdr
    motion_density = ((info.nb_frames or (info.duration or 1)*fps) / (info.width*info.height))
    if motion_density > 0.0002:
        crf += 1
    elif motion_density < 0.00005:
        crf -= 1
    crf = max(16, min(crf, 24))
    cq = crf + 1

    bitrate_ref = {'1080p': 16000, '1440p': 26000, '2160p': 50000}
    target_bitrate = bitrate_ref['2160p'] if max_dim >= 3840 else bitrate_ref['1440p'] if max_dim >= 2560 else bitrate_ref['1080p']

    vbv_maxrate = int(target_bitrate * (1.0 if hdr else 0.95))
    vbv_bufsize = int(vbv_maxrate * 1.4)
    LEVEL_VBV = {"main": {"vbv_maxrate": 0.95, "vbv_bufsize": 1.4}, "high": {"vbv_maxrate": 1.0, "vbv_bufsize": 1.5}}
    vbv_scale = LEVEL_VBV.get(tier, {"vbv_maxrate": 1.0, "vbv_bufsize": 1.5})
    vbv_maxrate = int(vbv_maxrate * vbv_scale["vbv_maxrate"])
    vbv_bufsize = int(vbv_bufsize * vbv_scale["vbv_bufsize"])

    if hdr:
        gop_sec = 2.0 if max_dim >= 3840 else 2.5
    else:
        gop_sec = 2.5 if max_dim >= 3840 else 3.0
    if fps > 60:
        gop_sec *= 1.05
    gop = int(round(gop_sec * fps / 2) * 2)
    gop = max(2, min(240, gop))
    if hdr and fps > 60:
        gop = min(gop, 120)

    return crf, cq, vbv_maxrate, vbv_bufsize, gop

def build_ffmpeg_params(info: VideoInfo, use_nvenc: bool, gpu_name: str) -> FFmpegParams:
    hdr = info.hdr
    if use_nvenc:
        level, tier, profile, pix_fmt = calculate_nvenc_hevc_level(info)
    else:
        level, tier = calculate_apple_hevc_level(info)
        profile = 'main10' if hdr else 'main'
        pix_fmt = 'p010le' if hdr else 'yuv420p'

    crf, cq, vbv_maxrate, vbv_bufsize, gop = calculate_dynamic_values(info, use_nvenc, gpu_name)

    if use_nvenc:
        preset = select_nvenc_preset(info, gpu_name)
        lookahead = int(min(info.fps * 1.5, 120))
        aq_strength = 6
        max_dim = max(info.width, info.height)
        if info.hdr:
            if max_dim >= 3840:
                aq_strength = 7
                lookahead = min(info.fps * 2, 120)
            if max_dim >= 7680:
                aq_strength = 8
                lookahead = 120

        vparams = [
            '-rc', 'vbr', '-tune', 'hq', '-multipass', 'fullres',
            '-cq', str(cq), '-b:v', '0',
            '-maxrate', str(vbv_maxrate * 1000), '-bufsize', str(vbv_bufsize * 1000),
            '-bf', '3', '-b_ref_mode', 'middle', '-rc-lookahead', str(lookahead),
            '-spatial-aq', '1', '-aq-strength', str(aq_strength),
            '-temporal-aq', '1', '-preset', preset,
            '-strict_gop', '1', '-no-scenecut', '1', '-g', str(gop),
            '-tier', tier
        ]
        hdr_metadata = build_hdr_metadata(info.master_display, info.max_cll, use_nvenc=True, fps=info.fps) if hdr else []
        return FFmpegParams('hevc_nvenc', pix_fmt, profile, level, [], vparams, hdr_metadata)
    else:
        x265_params = [
            f'crf={crf}', 'preset=slow', 'log-level=error',
            f'vbv-maxrate={vbv_maxrate}', f'vbv-bufsize={vbv_bufsize}', f'tier={tier}',
            f'keyint={gop}:min-keyint={gop}'
        ]
        if hdr:
            hdr_params = build_hdr_metadata(info.master_display, info.max_cll, use_nvenc=False, fps=info.fps)
            if '-x265-params' in hdr_params:
                idx = hdr_params.index('-x265-params')
                x265_str = hdr_params[idx + 1]
                x265_params += x265_str.split(':')
        return FFmpegParams('libx265', pix_fmt, profile, level, [], ['-x265-params', ':'.join(x265_params)], [])

COMMON_FFMPEG_FLAGS = [
    '-metadata:s:v:0', 'handler_name=VideoHandler',
    '-metadata:s:a:0', 'handler_name=SoundHandler',
    '-metadata:s:a:0', 'language=und',
    '-metadata:s:a:0', 'title=Main Audio'
]

def get_audio_flags(audio_channels: int) -> List[str]:
    min_bitrate = 128
    per_channel = 64
    max_total = 512
    calculated_bitrate = max(min_bitrate, audio_channels * per_channel)
    calculated_bitrate = min(calculated_bitrate, max_total)
    if audio_channels > 2:
        calculated_bitrate = max(calculated_bitrate, 256)

    audio_flags = ['-c:a', 'aac', '-b:a', f'{calculated_bitrate}k', '-ar', '48000']

    if audio_channels == 6:
        audio_flags += ['-channel_layout', '5.1']
    elif audio_channels == 8:
        audio_flags += ['-channel_layout', '7.1']

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
    elif ff_params.color_flags:
        cmd.extend(ff_params.color_flags)

    if extra_vparams:
        cmd.extend(extra_vparams)
    else:
        cmd.extend(ff_params.vparams)

    common_flags = COMMON_FFMPEG_FLAGS.copy()
    try:
        idx = common_flags.index('language=und')
        common_flags[idx] = f'language={audio_language or "eng"}'
    except ValueError:
        pass

    cmd.extend(common_flags)
    cmd.extend(get_audio_flags(audio_channels))
    cmd.extend(['-ac', str(audio_channels)])
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
    info = probe_video(file_path)
    gpu_name = detect_gpu_type()
    out_path = out_dir / (file_path.stem + '.mp4')
    audio_channels = probe_audio_channels(file_path)
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
        for attempt in range(1, len(NVENC_RETRIES) + 2):
            retry_vparams = adjust_nvenc_params(ff_params.vparams, attempt if attempt <= len(NVENC_RETRIES) else 1)
            cmd = build_ffmpeg_command(
                file_path, out_path, ff_params, audio_channels,
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
            file_path, out_path, ff_params_cpu, audio_channels,
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
