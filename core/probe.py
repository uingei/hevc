# core/probe.py
import json, subprocess, logging
from pathlib import Path
from dataclasses import dataclass

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
    duration: float

def parse_fps(rate_str: str) -> float:
    try:
        if not rate_str or '/' not in rate_str:
            return 30.0
        num, den = map(int, rate_str.split('/'))
        return num / den if den else 30.0
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

        # 优先使用视频流内的 tags，再回退到 format.tags
        tags = v.get('tags', {}) or info.get('format', {}).get('tags', {})

        return VideoInfo(
            v.get('width', 1920),
            v.get('height', 1080),
            parse_fps(v.get('avg_frame_rate') or v.get('r_frame_rate', '30/1')),
            v.get('color_primaries', 'bt709'),
            v.get('color_transfer', 'bt709'),
            v.get('color_space', 'bt709'),
            tags.get('master-display', ''),
            tags.get('max-cll', ''),
            float(info.get('format', {}).get('duration', 0) or 0)
        )
    except Exception as e:
        logger.error(f"探测失败: {file_path} - {e}")
        return VideoInfo(1920, 1080, 30, 'bt709', 'bt709', 'bt709', '', '', 0)

def probe_audio_channels(file_path: Path) -> int:
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'quiet', '-select_streams', 'a:0', '-show_entries', 'stream=channels', '-of', 'csv=p=0', str(file_path)],
            capture_output=True, text=True, check=True, encoding='utf-8', errors='replace'
        )
        return int(result.stdout.strip() or 2)
    except Exception:
        return 2
