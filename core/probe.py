# core/probe.py
import json, subprocess, logging
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

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
    audio_language: Optional[str] = 'eng'
    nb_frames: Optional[int] = None
    duration: Optional[float] = None

HDR_PIXFMTS = {'yuv420p10le', 'p010le', 'yuv444p10le'}
HDR_COLOR_SPACES = {'bt2020', 'bt2020-ncl', 'bt2020nc'}
HDR_TRANSFERS = {'smpte2084', 'pq'}
HDR_PRIMARIES = {'bt2020', 'bt2020-ncl'}

def parse_fps(rate_str: str) -> float:
    try:
        if not rate_str or '/' not in rate_str:
            return 30.0
        num, den = map(int, rate_str.split('/'))
        return num / den if den else 30.0
    except Exception:
        return 30.0

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
