# core/utils.py
import subprocess, logging
from pathlib import Path
from typing import List
from functools import lru_cache

logger = logging.getLogger(__name__)

def has_nvenc() -> bool:
    try:
        result = subprocess.run(['ffmpeg', '-hide_banner', '-encoders'],
                                capture_output=True, text=True, check=True, encoding='utf-8')
        return 'hevc_nvenc' in result.stdout
    except Exception:
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

def build_hdr_metadata(master_display: str, max_cll: str, use_nvenc: bool, fps: float = 30.0) -> List[str]:
    master_display = master_display.strip() or 'G(13250,34500)B(7500,3000)R(34000,16000)WP(15635,16450)L(10000000,50)'
    max_cll = max_cll.strip() or '1000,400'
    if use_nvenc:
        ordered_tags = [
            ('color_primaries', 'bt2020'),
            ('color_trc', 'smpte2084'),
            ('colorspace', 'bt2020nc'),
            ('master_display', master_display),
            ('max_cll', max_cll)
        ]
        return sum([['-metadata:s:v:0', f"{k}={v}"] for k, v in ordered_tags], [])
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
