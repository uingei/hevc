# config.py
import os
from pathlib import Path

APP_NAME = "Apple HEVC 批量转码"
APP_VERSION = "1.7.0"
LOG_FILE = "transcode_log.csv"

INPUT_EXTS = (
    '.mp4', '.mov', '.mkv', '.avi', '.wmv', '.flv', '.ts', '.m2ts', '.mts',
    '.m4v', '.webm', '.3gp', '.f4v', '.ogv', '.vob', '.mpg', '.mpeg'
)

DEFAULT_CRF = 18
MAX_WORKERS_SDR = max(1, os.cpu_count() or 1)
MAX_WORKERS_HDR = 2
