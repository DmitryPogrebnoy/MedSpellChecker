import numpy as np
import pynvml
import torch

# 8 Gb
MINIMAL_REQUIRED_GRU_MEMORY = 8192


def set_device() -> bool:
    """Set the most appropriate device for torch

       Returns:
            True if CUDA device selected, otherwise False
   """

    def set_cpu_device():
        torch.device("cpu")
        print(f"We will use device: CPU")

    if torch.cuda.is_available():
        gpus_free_mem_list = []
        for device_num in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(device_num)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpus_free_mem_list.append((info.total - info.used) // 1024 ** 3)
        selected_gpu_device_number = np.argmax(gpus_free_mem_list)
        handle = pynvml.nvmlDeviceGetHandleByIndex(selected_gpu_device_number)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        selected_gpu_device_memory = (info.total - info.used) // 1024 ** 2

        if selected_gpu_device_memory < MINIMAL_REQUIRED_GRU_MEMORY:
            print("All available GPUs are busy and don't have required free memory")
            set_cpu_device()
            return False

        print(selected_gpu_device_number)
        torch.cuda.set_device(torch.device(selected_gpu_device_number))
        print(f"Selected GPU number: {torch.cuda.current_device()}")
        print(f"Will use device {torch.cuda.current_device()}: "
              f"{torch.cuda.get_device_name(torch.cuda.current_device())}")
        print(f"Device has {selected_gpu_device_memory} Gb free memory")
        return True
    else:
        print("There is no available GPU")
        set_cpu_device()
        return False


def print_gpu_memory_stats():
    current_device = torch.cuda.current_device()
    handle = pynvml.nvmlDeviceGetHandleByIndex(current_device)
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    print(f"All GPU memory occupied: {info.used // 1024 ** 3}/{info.total // 1024 ** 3}  Gb.")
    print(f"Torch GPU {current_device} memory allocated: {torch.cuda.memory_allocated(current_device) // 1024 ** 3} Gb")
