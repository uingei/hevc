import torch
import subprocess
import sys

def check_cuda():
    try:
        # 检查 PyTorch GPU 支持
        cuda_available = torch.cuda.is_available()
        gpu_count = torch.cuda.device_count() if cuda_available else 0
        print(f"CUDA 可用: {cuda_available}")
        print(f"检测到 GPU 数量: {gpu_count}")

        for i in range(gpu_count):
            name = torch.cuda.get_device_name(i)
            mem = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            print(f"GPU {i}: {name}, 显存: {mem:.1f} GB")
        
        if not cuda_available:
            print("\nPyTorch 检测不到 CUDA，请确认已安装 NVIDIA 驱动和 CUDA Toolkit")
        
        return cuda_available

    except Exception as e:
        print("检查 CUDA 时出现异常:", e)
        return False

def check_nvcc():
    try:
        # 检查 nvcc 命令
        result = subprocess.run(["nvcc", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode == 0:
            print("\nnvcc 版本信息:")
            print(result.stdout)
            return True
        else:
            print("\n未检测到 nvcc，请确认 CUDA Toolkit 已正确安装并加入 PATH")
            return False
    except FileNotFoundError:
        print("\n未检测到 nvcc，请确认 CUDA Toolkit 已正确安装并加入 PATH")
        return False

if __name__ == "__main__":
    print("===============================")
    print(" GPU + CUDA + PyTorch 环境检测 ")
    print("===============================\n")
    cuda_ok = check_cuda()
    nvcc_ok = check_nvcc()

    if cuda_ok and nvcc_ok:
        print("\n环境检测通过，GPU 可用！")
    else:
        print("\n环境未完全就绪，请按照提示安装或配置 CUDA 驱动和 Toolkit。")
    
    input("\n按回车键退出...")
