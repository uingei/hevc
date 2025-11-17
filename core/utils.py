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
