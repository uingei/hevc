# core/utils.py
import subprocess, logging
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)

def has_nvenc() -> bool:
    try:
        result = subprocess.run(['ffmpeg', '-hide_banner', '-encoders'],
                                capture_output=True, text=True, check=True)
        return 'hevc_nvenc' in result.stdout
    except Exception:
        return False

def detect_gpu_type() -> str:
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                                capture_output=True, text=True, check=True)
        return result.stdout.strip().lower()
    except Exception:
        return "unknown"

def is_hdr(info) -> bool:
    return info.color_transfer.lower() == 'smpte2084' or info.color_primaries.lower() == 'bt2020'

def build_hdr_metadata(master_display: str, max_cll: str) -> List[str]:
    if not master_display:
        master_display = 'G(13250,34500)B(7500,3000)R(34000,16000)WP(15635,16450)L(10000000,50)'
    if not max_cll:
        max_cll = '1000,400'
    return ['-color_primaries', 'bt2020', '-color_trc', 'smpte2084',
            '-colorspace', 'bt2020nc', '-master-display', master_display, '-max-cll', max_cll]
