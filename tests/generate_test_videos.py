# tests/generate_test_videos.py
# coding: utf-8
import subprocess
from pathlib import Path

OUTPUT_DIR = Path("tests/sample_videos")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 定义测试视频参数: (文件名, 分辨率, 帧率, 是否 HDR)
VIDEOS = [
    ("1080p_sdr.mp4", "1920x1080", 30, False),
    ("720p_sdr.mp4", "1280x720", 30, False),
    ("4k_sdr.mp4", "3840x2160", 30, False),
    ("1080p_hdr.mp4", "1920x1080", 30, True),
    ("4k_hdr.mp4", "3840x2160", 30, True),
]

for name, res, fps, hdr in VIDEOS:
    out_file = OUTPUT_DIR / name
    if out_file.exists():
        print(f"已存在: {out_file}, 跳过生成")
        continue

    width, height = res.split('x')
    # FFmpeg 命令生成测试视频
    cmd = [
        "ffmpeg",
        "-y",
        "-f", "lavfi",
        "-i", f"testsrc=size={res}:rate={fps}:duration=5",
    ]

    if hdr:
        # HDR 模拟：使用 bt2020 和 smpte2084 转换
        cmd += ["-color_primaries", "bt2020", "-color_trc", "smpte2084", "-colorspace", "bt2020nc"]

    cmd += [str(out_file)]
    print(f"生成测试视频: {out_file}")
    subprocess.run(cmd, check=True)
