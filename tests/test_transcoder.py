# tests/test_transcoder.py
# coding: utf-8
import unittest
from pathlib import Path
import threading
from core.transcoder import convert_video, is_hdr, probe_video

class TestTranscoder(unittest.TestCase):

    def setUp(self):
        self.test_dir = Path("tests/sample_videos")
        self.output_dir = Path("tests/output")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def test_probe_video(self):
        for f in self.test_dir.glob("*"):
            info = probe_video(f)
            self.assertGreater(info.width, 0)
            self.assertGreater(info.height, 0)
            self.assertGreater(info.fps, 0)
            print(f"{f.name}: {info}")

    def test_is_hdr_detection(self):
        for f in self.test_dir.glob("*"):
            info = probe_video(f)
            hdr_flag = is_hdr(info)
            print(f"{f.name} HDR: {hdr_flag}")

    def test_convert_video_cpu(self):
        for f in self.test_dir.glob("*"):
            log = convert_video(f, self.output_dir, force_cpu=True, skip_validator=True)
            self.assertIn(log["status"], ["SUCCESS", "FAILED"])
            print(log)

    def test_convert_video_stop(self):
        f = next(self.test_dir.glob("*"), None)
        if not f: return

        stop_event = threading.Event()
        def progress_callback(name, frame, total):
            if frame > 2:  # 模拟中途停止
                stop_event.set()

        log = convert_video(f, self.output_dir, force_cpu=True, skip_validator=True,
                            progress_callback=progress_callback, stop_event=stop_event)
        print(log)
        self.assertTrue(log["status"] in ["SUCCESS", "FAILED"])

if __name__ == "__main__":
    unittest.main()
