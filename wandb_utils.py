import psutil
import wandb
import torch
import os
import pynvml

def is_gpu_available():
    try:
        pynvml.nvmlInit()
        return True
    except pynvml.NVMLError:
        return False

def log_memory_usage():
    process = psutil.Process()
    mem_info = process.memory_info()
    log_data = {'memory_usage': mem_info.rss / (1024 * 1024)}

    if torch.cuda.is_available():
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # Assumes a single GPU, index 0
        gpu_mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        log_data['gpu_memory_usage'] = gpu_mem_info.used / (1024 * 1024)

        log_data['cuda_memory_allocated'] = torch.cuda.memory_allocated()
        log_data['cuda_memory_reserved'] = torch.cuda.memory_reserved()
        log_data['cuda_max_memory_allocated'] = torch.cuda.max_memory_allocated()

    wandb.log(log_data)

def write_file(str):
    with open(f'/user/ml4723/Prj/NIC/progress_log/log{os.getpid()}.txt', 'a') as f:
        f.write(str + '\n')
