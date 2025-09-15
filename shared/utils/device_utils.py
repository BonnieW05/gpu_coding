"""
设备工具函数
用于GPU设备管理和信息获取
"""

import torch
import subprocess
import platform

def get_gpu_info():
    """获取GPU信息"""
    if not torch.cuda.is_available():
        return {"available": False, "message": "CUDA不可用"}
    
    gpu_info = {
        "available": True,
        "device_count": torch.cuda.device_count(),
        "current_device": torch.cuda.current_device(),
        "device_name": torch.cuda.get_device_name(),
        "memory_allocated": torch.cuda.memory_allocated(),
        "memory_reserved": torch.cuda.memory_reserved(),
        "max_memory_allocated": torch.cuda.max_memory_allocated(),
        "max_memory_reserved": torch.cuda.max_memory_reserved(),
    }
    
    return gpu_info

def print_gpu_info():
    """打印GPU信息"""
    info = get_gpu_info()
    
    if not info["available"]:
        print("❌ CUDA不可用")
        return
    
    print("🖥️  GPU信息:")
    print(f"   设备数量: {info['device_count']}")
    print(f"   当前设备: {info['current_device']}")
    print(f"   设备名称: {info['device_name']}")
    print(f"   已分配内存: {info['memory_allocated'] / 1024**3:.2f} GB")
    print(f"   已保留内存: {info['memory_reserved'] / 1024**3:.2f} GB")
    print(f"   最大分配内存: {info['max_memory_allocated'] / 1024**3:.2f} GB")
    print(f"   最大保留内存: {info['max_memory_reserved'] / 1024**3:.2f} GB")

def get_nvidia_smi_info():
    """获取nvidia-smi信息"""
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout
        else:
            return "nvidia-smi命令执行失败"
    except FileNotFoundError:
        return "nvidia-smi命令未找到"

def clear_gpu_memory():
    """清理GPU内存"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("✅ GPU内存已清理")

def set_device(device_id=None):
    """设置CUDA设备"""
    if not torch.cuda.is_available():
        print("❌ CUDA不可用，无法设置设备")
        return None
    
    if device_id is None:
        device_id = 0
    
    if device_id >= torch.cuda.device_count():
        print(f"❌ 设备ID {device_id} 超出范围")
        return None
    
    torch.cuda.set_device(device_id)
    print(f"✅ 已设置设备为: {torch.cuda.get_device_name(device_id)}")
    return device_id

def benchmark_function(func, *args, **kwargs):
    """函数性能基准测试"""
    import time
    
    # 预热
    for _ in range(10):
        func(*args, **kwargs)
    
    # 同步GPU
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # 计时
    start_time = time.time()
    for _ in range(100):
        result = func(*args, **kwargs)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    end_time = time.time()
    avg_time = (end_time - start_time) / 100
    
    print(f"⏱️  平均执行时间: {avg_time*1000:.2f} ms")
    return result, avg_time

if __name__ == "__main__":
    print_gpu_info()
    print("\n" + "="*50)
    print(get_nvidia_smi_info())
