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
from typing import List, Optional, Callable, Dict, Any

from core.probe import probe_video, probe_audio_channels, VideoInfo
from core.utils import has_nvenc, detect_gpu_type, is_hdr, build_hdr_metadata
import config

logger = logging.getLogger(__name__)

# ---- dataclasses ----
@dataclass
class FFmpegParams:
    vcodec: str
    pix_fmt: str
    profile: str
    level: str
    color_flags: List[str]
    vparams: List[str]

# ---- NVENC retry presets (按原逻辑保留) ----
NVENC_RETRIES = [
    {'-bf': '3', '-b_ref_mode': 'middle'},
    {'-bf': '0', '-b_ref_mode': 'disabled'},
    {'-bf': '0', '-b_ref_mode': 'disabled', '-temporal-aq': '0'},
    {'-bf': '0', '-b_ref_mode': 'disabled', '-temporal-aq': '0', '-spatial-aq': '0'}
]

# ---- helper functions (迁移并清理自原脚本) ----
def parse_fps(rate_str: str) -> float:
    try:
        if not rate_str or '/' not in rate_str:
            return 30.0
        num, den = map(int, rate_str.split('/'))
        return num / den if den != 0 else 30.0
    except Exception:
        return 30.0

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

def auto_nvenc_cq(info: VideoInfo, gpu_name: str) -> int:
    if max(info.width, info.height) >= 3840:
        base_cq = 18
    elif max(info.width, info.height) >= 2560:
        base_cq = 19
    else:
        base_cq = 20
    if "rtx" in (gpu_name or "").lower():
        base_cq -= 1
    return base_cq

def select_nvenc_preset(info: VideoInfo, gpu_name: str) -> str:
    res = max(info.width, info.height)
    g = (gpu_name or "").lower()
    if 'rtx' in g:
        if res >= 3840: return 'p7'
        elif res >= 2560: return 'p7'
        else: return 'p6'
    else:
        if res >= 3840: return 'p6'
        elif res >= 2560: return 'p6'
        else: return 'p5'

def build_hdr_color_flags(master_display: str, max_cll: str) -> List[str]:
    return build_hdr_metadata(master_display, max_cll)

def decide_encoder(info: VideoInfo, force_cpu: bool, force_gpu: bool) -> bool:
    """返回 True 表示优先使用 NVENC（若存在），False 表示使用 CPU"""
    hdr_video = is_hdr(info)
    if hdr_video or force_cpu:
        return False
    if force_gpu:
        return has_nvenc()
    return has_nvenc()

def adjust_nvenc_params(params: List[str], attempt: int) -> List[str]:
    new_params = params.copy()
    if 1 <= attempt <= len(NVENC_RETRIES):
        retry_mods = NVENC_RETRIES[attempt-1]
        # 修改 param 列表中的对应项值（假定 params 列表为键、值交替）
        for k, v in retry_mods.items():
            if k in new_params:
                try:
                    idx = new_params.index(k)
                    if idx + 1 < len(new_params):
                        new_params[idx + 1] = str(v)
                except ValueError:
                    pass
    return new_params

def build_ffmpeg_params(info: VideoInfo, use_nvenc: bool, crf: int, gpu_name: str) -> FFmpegParams:
    hdr_video = is_hdr(info)
    is_vertical = info.height > info.width
    level = calculate_apple_hevc_level(info)
    profile = 'main10' if hdr_video else 'main'
    pix_fmt = 'p010le' if hdr_video else 'yuv420p'
    keyint = max(1, int(round(info.fps * (1.2 if is_vertical else 1.5))))
    color_flags = ['-color_primaries', 'bt709', '-color_trc', 'bt709', '-colorspace', 'bt709']
    if hdr_video:
        color_flags = build_hdr_color_flags(info.master_display, info.max_cll)
        # 如果 HDR，通常希望使用 CPU x265（除非你有特殊 NVENC 支持） - 但仍允许尝试 NVENC
        if use_nvenc and not has_nvenc():
            use_nvenc = False

    if use_nvenc:
        vcodec = 'hevc_nvenc'
        cq_base = auto_nvenc_cq(info, gpu_name)
        if is_vertical:
            cq_base = max(15, cq_base - 1)
        preset = select_nvenc_preset(info, gpu_name)
        if is_vertical and preset.startswith("p") and preset[1:].isdigit():
            preset = f"p{min(int(preset[1:]) + 1, 7)}"
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
            'profile=main10' if hdr_video else 'profile=main'
        ]
        if hdr_video:
            x265_params += ['hdr10=1', 'hdr-opt=1', 'colorprim=bt2020', 'transfer=smpte2084', 'colormatrix=bt2020nc']
        if info.master_display:
            x265_params.append(f'master-display={info.master_display}')
        if info.max_cll:
            x265_params.append(f'max-cll={info.max_cll}')
        vparams = ['-preset', 'slow', '-x265-params', ':'.join(x265_params)]

    return FFmpegParams(vcodec=vcodec, pix_fmt=pix_fmt, profile=profile, level=level, color_flags=color_flags, vparams=vparams)

def build_ffmpeg_command(file_path: Path, out_path: Path, ff_params: FFmpegParams, vparams_override: Optional[List[str]], audio_channels: int) -> List[str]:
    vparams_to_use = vparams_override if vparams_override is not None else ff_params.vparams
    cmd = [
        'ffmpeg', '-hide_banner', '-y', '-i', str(file_path),
        '-map_metadata', '0',
        '-c:v', ff_params.vcodec,
        '-pix_fmt', ff_params.pix_fmt,
        '-profile:v', ff_params.profile,
        '-level:v', ff_params.level,
        '-tag:v', 'hvc1'
    ]
    cmd.extend(ff_params.color_flags)
    cmd.extend(vparams_to_use)
    cmd.extend([
        '-ac', str(audio_channels),
        '-metadata:s:v:0', 'handler_name=VideoHandler',
        '-metadata:s:a:0', 'handler_name=SoundHandler',
        '-c:a', 'aac', '-b:a', '192k', '-ar', '48000',
        str(out_path)
    ])
    return cmd

def run_apple_validator(file_path: Path):
    possible_paths = [
        Path('/Applications/Apple Video Tools/AppleHEVCValidator'),
        Path('/usr/local/bin/AppleHEVCValidator'),
        Path('C:/Program Files/Apple/AppleHEVCValidator.exe')
    ]
    validator = next((p for p in possible_paths if p.exists()), None)
    if not validator:
        return
    try:
        subprocess.run([str(validator), str(file_path)], check=True, encoding='utf-8', errors='replace')
    except subprocess.CalledProcessError:
        logger.warning(f"Apple Validator 未通过: {file_path.name}")
    except Exception as e:
        logger.debug(f"运行 Apple Validator 出错: {e}")

# ---- ffmpeg 运行函数，支持 stop_event ----
def run_ffmpeg(cmd: List[str], progress_callback: Optional[Callable[[str, int, int], None]], file_name: str, total_frames: int, stop_event: Optional[threading.Event] = None, debug: bool = False) -> int:
    """
    运行 ffmpeg 命令，逐行读取 stdout 以解析进度（frame=）。
    如果传入 stop_event 并且被 set()，会尝试终止子进程并返回非零。
    """
    if debug:
        logger.debug("运行 FFmpeg 命令: %s", " ".join(cmd))
    try:
        with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace') as proc:
            frame = 0
            # 逐行读取输出
            for line in proc.stdout:
                if stop_event and stop_event.is_set():
                    try:
                        proc.terminate()
                    except Exception:
                        pass
                    return 1
                if "frame=" in line:
                    # 尝试解析 frame 数（保守解析）
                    try:
                        # 分割并寻找 frame= 后的第一个整数
                        part = line.strip().split("frame=")[-1].split()[0]
                        frame = int(part)
                    except Exception:
                        frame = frame
                    if progress_callback:
                        try:
                            progress_callback(file_name, frame, total_frames)
                        except Exception:
                            # 避免进度回调抛异常导致转码中断
                            logger.debug("progress_callback 抛异常", exc_info=True)
            return proc.wait()
    except Exception as e:
        logger.error("运行 ffmpeg 失败: %s — %s", cmd[:3], e)
        return 1

# ---- 主转换函数 ----
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
    crf_attempt = config.DEFAULT_CRF
    use_nvenc = decide_encoder(info, force_cpu, force_gpu)
    audio_channels = probe_audio_channels(file_path)
    hdr_video = is_hdr(info)

    log_entry = {
        "file": file_path.name,
        "status": "FAILED",
        "crf": None,
        "retries": 0,
        "method": "NVENC" if use_nvenc else "CPU",
        "hdr": hdr_video
    }

    ff_params = build_ffmpeg_params(info, use_nvenc, crf_attempt, gpu_name)
    total_frames = max(1, int(info.duration * info.fps)) if info.duration and info.fps else 1

    # 如果使用 NVENC，尝试多次，失败则回退 CPU（和原逻辑一致）
    if use_nvenc:
        # 允许最多 len(NVENC_RETRIES)+1 次尝试（最后一次为回退/最终尝试）
        max_attempts = len(NVENC_RETRIES) + 1
        for attempt in range(1, max_attempts + 1):
            # 当 attempt 超出 NVENC_RETRIES 时，保持最后一组参数（或可回退到 CPU）
            vparams_try = adjust_nvenc_params(ff_params.vparams, attempt if attempt <= len(NVENC_RETRIES) else len(NVENC_RETRIES))
            cmd = build_ffmpeg_command(file_path, out_path, ff_params, vparams_override=vparams_try, audio_channels=audio_channels)
            if debug:
                logger.debug("NVENC FFmpeg 命令 (尝试 %d): %s", attempt, " ".join(cmd))
            ret = run_ffmpeg(cmd, progress_callback, file_path.name, total_frames, stop_event=stop_event, debug=debug)
            if ret == 0:
                log_entry["status"] = "SUCCESS"
                log_entry["crf"] = auto_nvenc_cq(info, gpu_name)
                log_entry["retries"] = attempt if attempt <= len(NVENC_RETRIES) else len(NVENC_RETRIES)
                break
            # 如果是最后一次尝试仍失败，回退到 CPU（保留原脚本行为）
            if attempt == max_attempts:
                logger.info("NVENC 多次失败，准备回退 CPU 编码")
                use_nvenc = False
                ff_params = build_ffmpeg_params(info, False, crf_attempt, gpu_name)

    # CPU 路径（或 NVENC 回退）
    if not use_nvenc:
        cpu_params = build_ffmpeg_params(info, False, crf_attempt, gpu_name)
        cmd_cpu = build_ffmpeg_command(file_path, out_path, cpu_params, vparams_override=None, audio_channels=audio_channels)
        if debug:
            logger.debug("CPU FFmpeg 命令: %s", " ".join(cmd_cpu))
        ret_cpu = run_ffmpeg(cmd_cpu, progress_callback, file_path.name, total_frames, stop_event=stop_event, debug=debug)
        if ret_cpu == 0:
            log_entry["status"] = "SUCCESS"
            log_entry["crf"] = crf_attempt
            log_entry["retries"] = 0
            log_entry["method"] = "CPU"

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
