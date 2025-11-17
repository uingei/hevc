import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch, cv2, subprocess, time, os, sys, logging
from itertools import cycle
from queue import Queue
import platform

# ==== 日志设置 ====
logging.basicConfig(filename='video_upscale_gui.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# ==== 资源路径兼容 PyInstaller ====
def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# ==== GPU 初始化 ====
gpu_list = list(range(torch.cuda.device_count()))
gpu_memory = {}
gpu_model_map = {}
gpu_batch_map = {}
gpu_cycle = None
gpu_lock = threading.Lock()

def init_gpu():
    global gpu_cycle
    if not gpu_list:
        messagebox.showerror("错误", "未检测到 GPU，请安装 CUDA")
        return False
    for i in gpu_list:
        props = torch.cuda.get_device_properties(i)
        total_gb = props.total_memory / (1024**3)
        gpu_memory[i] = total_gb
        if total_gb >= 16:
            gpu_model_map[i] = "RealESRGAN_x4plus.pth"
            gpu_batch_map[i] = 2
        elif total_gb >= 8:
            gpu_model_map[i] = "RealESRGAN_x2plus.pth"
            gpu_batch_map[i] = 1
        else:
            gpu_model_map[i] = "RealESRGANv2-anime_6B.pth"
            gpu_batch_map[i] = 1
    gpu_cycle = cycle(gpu_list)
    logging.info(f"GPU 初始化完成: { {i:(gpu_memory[i], gpu_model_map[i], gpu_batch_map[i]) for i in gpu_list} }")
    return True

def gpu_load(gpu_id):
    try:
        mem = torch.cuda.memory_allocated(gpu_id)
        total = torch.cuda.get_device_properties(gpu_id).total_memory
        return mem/total
    except:
        return 0

# ==== 跨平台打开目录 ====
def open_folder(path):
    if platform.system() == "Windows":
        os.startfile(path)
    elif platform.system() == "Darwin":
        subprocess.run(["open", path])
    else:
        subprocess.run(["xdg-open", path])

# ==== 视频处理函数 ====
def process_video(video_path: Path, output_dir: Path, noise:int, frame_rate:int, frame_interp:bool, interp_multiplier:int, target_height:int,
                  retry:int, progress_var, gpu_label, eta_label, pause_flag, stop_flag):
    start_time = time.time()
    cap = cv2.VideoCapture(str(video_path))
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 自动或自定义分辨率
    if target_height == 0:  # 0 表示自动
        if orig_height < 720: target_height = 1080
        elif orig_height < 1080: target_height = 1080
        elif orig_height < 2160: target_height = 2160
        else: target_height = orig_height
    scale = target_height / orig_height
    target_width = int(orig_width * scale)

    tmp_dir = Path("tmp_processing")
    tmp_dir.mkdir(exist_ok=True)
    tmp_video_path = tmp_dir / f"tmp_{video_path.stem}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(tmp_video_path), fourcc, frame_rate, (target_width, target_height))

    frame_queue = Queue(maxsize=32)
    result_dict = {}
    reader_stop = threading.Event()

    def frame_reader():
        idx = 0
        while True:
            if stop_flag.is_set():
                break
            ret, frame = cap.read()
            if not ret:
                break
            frame_queue.put((idx, frame))
            idx += 1
        cap.release()
        reader_stop.set()

    def frame_worker():
        while not (reader_stop.is_set() and frame_queue.empty()) and not stop_flag.is_set():
            while pause_flag.is_set() and not stop_flag.is_set():
                time.sleep(0.5)
            try:
                idx, frame = frame_queue.get(timeout=1)
            except:
                continue
            attempt = 0
            while attempt <= retry:
                try:
                    with gpu_lock:
                        gpu_id = next(gpu_cycle)
                        model = gpu_model_map[gpu_id]
                        batch_size = gpu_batch_map[gpu_id]

                    tmp_in = tmp_dir / f"frame_{video_path.stem}_{idx:06d}.png"
                    tmp_out = tmp_dir / f"up_{video_path.stem}_{idx:06d}.png"
                    cv2.imwrite(str(tmp_in), frame)

                    cmd = [
                        "python", resource_path("inference_realesrgan.py"),
                        "-i", str(tmp_in),
                        "-o", str(tmp_out),
                        "-s", str(scale),
                        "-n", str(noise),
                        "--model-path", resource_path(f"models/{model}"),
                        "--gpu-id", str(gpu_id)
                    ]
                    result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
                    if result.returncode != 0:
                        raise RuntimeError(result.stderr.decode())

                    up_frame = cv2.imread(str(tmp_out))
                    tmp_in.unlink(missing_ok=True)
                    tmp_out.unlink(missing_ok=True)
                    result_dict[idx] = up_frame
                    break
                except Exception as e:
                    logging.exception(f"处理帧 {idx} 错误: {e}")
                    attempt += 1
                    if attempt > retry:
                        result_dict[idx] = frame

    reader_thread = threading.Thread(target=frame_reader)
    reader_thread.start()
    workers = []
    for _ in range(len(gpu_list)):
        t = threading.Thread(target=frame_worker)
        t.start()
        workers.append(t)

    next_idx = 0
    while next_idx < total_frames and not stop_flag.is_set():
        while pause_flag.is_set() and not stop_flag.is_set():
            time.sleep(0.5)
        if next_idx in result_dict:
            out.write(result_dict.pop(next_idx))
            next_idx += 1
            progress_var.set(int(next_idx / total_frames * 100))
            elapsed = time.time() - start_time
            eta = elapsed / next_idx * (total_frames - next_idx)
            gpu_status = ", ".join([f"GPU{i}:{gpu_load(i)*100:.0f}%" for i in gpu_list])
            gpu_label.config(text=f"GPU 使用: {gpu_status}")
            eta_label.config(text=f"预计剩余: {int(eta)} 秒")
        else:
            time.sleep(0.1)

    reader_thread.join()
    for t in workers:
        t.join()
    out.release()

    if stop_flag.is_set():
        logging.info(f"视频处理被取消: {video_path.name}")
        return

    if frame_interp:
        tmp_interp = tmp_dir / f"interp_{video_path.stem}.mp4"
        subprocess.run([
            "ffmpeg",
            "-i", str(tmp_video_path),
            "-filter:v", f"minterpolate='mi_mode=mci:mc_mode=aobmc:vsbmc=1:fps={frame_rate*interp_multiplier}'",
            str(tmp_interp)
        ])
        tmp_video_path.unlink()
        tmp_video_path = tmp_interp

    final_output = output_dir / video_path.name
    subprocess.run([
        "ffmpeg",
        "-i", str(tmp_video_path),
        "-i", str(video_path),
        "-c:v", "copy",
        "-c:a", "aac",
        "-map", "0:v:0",
        "-map", "1:a:0",
        str(final_output)
    ])
    tmp_video_path.unlink()
    logging.info(f"完成视频: {video_path.name}")

# ==== GUI ====
class VideoUpscaleGUI:
    def __init__(self, master):
        self.master = master
        master.title("最终增强版视频高清修复工具")

        self.input_dir = tk.StringVar()
        self.output_dir = tk.StringVar()
        self.noise = tk.IntVar(value=2)
        self.frame_rate = tk.IntVar(value=30)
        self.frame_interp = tk.BooleanVar()
        self.interp_multiplier = tk.IntVar(value=2)
        self.target_height = tk.IntVar(value=0)  # 0 表示自动
        self.retry = tk.IntVar(value=2)
        self.progress_var = tk.IntVar()
        self.pause_flag = threading.Event()
        self.stop_flag = threading.Event()

        # --- 基本设置 ---
        tk.Label(master, text="输入目录").grid(row=0, column=0)
        tk.Entry(master, textvariable=self.input_dir, width=50).grid(row=0, column=1)
        tk.Button(master, text="浏览", command=self.browse_input).grid(row=0, column=2)

        tk.Label(master, text="输出目录").grid(row=1, column=0)
        tk.Entry(master, textvariable=self.output_dir, width=50).grid(row=1, column=1)
        tk.Button(master, text="浏览", command=self.browse_output).grid(row=1, column=2)

        tk.Label(master, text="降噪等级 (0-3)").grid(row=2, column=0)
        tk.Entry(master, textvariable=self.noise, width=5).grid(row=2, column=1, sticky='w')

        tk.Label(master, text="输出帧率").grid(row=3, column=0)
        tk.Entry(master, textvariable=self.frame_rate, width=5).grid(row=3, column=1, sticky='w')

        # --- 高级设置 ---
        tk.Checkbutton(master, text="启用帧插值", variable=self.frame_interp).grid(row=4, column=1, sticky='w')
        tk.Label(master, text="插值倍数").grid(row=5, column=0)
        tk.Entry(master, textvariable=self.interp_multiplier, width=5).grid(row=5, column=1, sticky='w')

        tk.Label(master, text="目标高度 (0=自动)").grid(row=6, column=0)
        tk.Entry(master, textvariable=self.target_height, width=5).grid(row=6, column=1, sticky='w')

        tk.Label(master, text="失败重试次数").grid(row=7, column=0)
        tk.Entry(master, textvariable=self.retry, width=5).grid(row=7, column=1, sticky='w')

        tk.Label(master, text="处理进度").grid(row=8, column=0)
        self.progress = ttk.Progressbar(master, orient='horizontal', length=400, mode='determinate', variable=self.progress_var)
        self.progress.grid(row=8, column=1, columnspan=2, pady=5)

        self.gpu_label = tk.Label(master, text="GPU 使用: ")
        self.gpu_label.grid(row=9, column=0, columnspan=3)
        self.eta_label = tk.Label(master, text="预计剩余: ")
        self.eta_label.grid(row=10, column=0, columnspan=3)

        tk.Button(master, text="一键理想配置", command=self.reset_defaults).grid(row=11, column=0)
        tk.Button(master, text="开始处理", command=self.start_process).grid(row=11, column=1)
        tk.Button(master, text="打开输出目录", command=self.open_output).grid(row=11, column=2)

        self.pause_btn = tk.Button(master, text="暂停", command=self.toggle_pause)
        self.pause_btn.grid(row=12, column=1)
        self.stop_btn = tk.Button(master, text="取消", command=self.stop_process)
        self.stop_btn.grid(row=12, column=2)

    # ==== GUI 功能方法 ====
    def browse_input(self):
        dir_path = filedialog.askdirectory()
        if dir_path: self.input_dir.set(dir_path)

    def browse_output(self):
        dir_path = filedialog.askdirectory()
        if dir_path: self.output_dir.set(dir_path)

    def toggle_pause(self):
        if not self.pause_flag.is_set():
            self.pause_flag.set()
            self.pause_btn.config(text="继续")
        else:
            self.pause_flag.clear()
            self.pause_btn.config(text="暂停")

    def stop_process(self):
        self.stop_flag.set()
        messagebox.showinfo("提示", "处理中止，将尽快停止当前视频")

    def reset_defaults(self):
        """一键恢复理想默认值"""
        self.noise.set(2)
        self.frame_rate.set(30)
        self.frame_interp.set(False)
        self.interp_multiplier.set(2)
        self.target_height.set(0)
        self.retry.set(2)
        messagebox.showinfo("提示", "已恢复为推荐默认配置")

    def start_process(self):
        if not self.input_dir.get() or not self.output_dir.get():
            messagebox.showwarning("警告", "请选择输入和输出目录")
            return
        self.pause_flag.clear()
        self.stop_flag.clear()
        threading.Thread(target=self.process_all_videos, daemon=True).start()

    def process_all_videos(self):
        if not init_gpu(): 
            return

        input_path = Path(self.input_dir.get())
        output_path = Path(self.output_dir.get())
        video_files = list(input_path.glob("*.*"))
        if not video_files:
            messagebox.showwarning("警告", "输入目录没有视频文件")
            return

        def worker(video_file):
            self.progress_var.set(0)
            process_video(
                video_file, output_path,
                self.noise.get(),
                self.frame_rate.get(),
                self.frame_interp.get(),
                self.interp_multiplier.get(),
                self.target_height.get(),
                self.retry.get(),
                self.progress_var,
                self.gpu_label,
                self.eta_label,
                self.pause_flag,
                self.stop_flag
            )
            logging.info(f"视频完成: {video_file.name}")

        def run_pool():
            with ThreadPoolExecutor(max_workers=min(len(video_files), len(gpu_list))) as executor:
                futures = [executor.submit(worker, v) for v in video_files]
                for f in as_completed(futures):
                    try:
                        f.result()
                    except Exception as e:
                        logging.exception(f"处理视频失败: {e}")

            if not self.stop_flag.is_set():
                messagebox.showinfo("完成", "所有视频处理完成！")
                open_folder(output_path)

        threading.Thread(target=run_pool, daemon=True).start()

    def open_output(self):
        if self.output_dir.get():
            open_folder(self.output_dir.get())

# ==== 启动 GUI ====
if __name__ == "__main__":
    root = tk.Tk()
    app = VideoUpscaleGUI(root)
    root.mainloop()
