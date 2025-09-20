"""
工具函数
"""

import yaml
import logging
import os
from pathlib import Path
from typing import Dict, Any, List
import json


def load_config(config_path: str) -> Dict[str, Any]:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def setup_logging(config: Dict[str, Any]) -> None:
    """设置日志系统"""
    log_config = config.get("logging", {})
    
    # 创建日志目录
    log_file = log_config.get("log_file", "./logs/kevin_evaluation.log")
    log_dir = Path(log_file).parent
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # 配置日志
    logging.basicConfig(
        level=getattr(logging, log_config.get("level", "INFO")),
        format=log_config.get("log_format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"),
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )


def load_test_cases(dataset_path: str) -> List[Dict[str, Any]]:
    """加载测试用例"""
    # 简化的测试用例
    test_cases = [
        {
            "name": "vector_add",
            "description": "向量加法内核",
            "input_size": 1024 * 1024,
            "prompt": "Write a CUDA kernel for vector addition: c[i] = a[i] + b[i]"
        },
        {
            "name": "matrix_multiply",
            "description": "矩阵乘法内核",
            "input_size": 1024,
            "prompt": "Write a CUDA kernel for matrix multiplication: C = A * B"
        },
        {
            "name": "reduction_sum",
            "description": "归约求和内核",
            "input_size": 1024 * 1024,
            "prompt": "Write a CUDA kernel for reduction sum of an array"
        }
    ]
    return test_cases


def save_generated_kernels(kernels: List[str], test_cases: List[Dict[str, Any]], 
                          output_dir: str) -> None:
    """保存生成的内核代码"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for kernel, test_case in zip(kernels, test_cases):
        if kernel:
            filename = f"{test_case['name']}_generated.cu"
            filepath = output_path / filename
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"// Generated kernel for: {test_case['description']}\n")
                f.write(f"// Test case: {test_case['name']}\n\n")
                f.write(kernel)


def check_system_requirements() -> Dict[str, Any]:
    """检查系统要求"""
    import torch
    import subprocess
    
    system_info = {
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "gpu_info": []
    }
    
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            gpu_info = {
                "device_id": i,
                "name": torch.cuda.get_device_name(i),
                "memory_total": torch.cuda.get_device_properties(i).total_memory / 1024**3,
                "compute_capability": torch.cuda.get_device_properties(i).major
            }
            system_info["gpu_info"].append(gpu_info)
    
    # 检查NVCC
    try:
        result = subprocess.run(["nvcc", "--version"], capture_output=True, text=True)
        system_info["nvcc_available"] = result.returncode == 0
        if result.returncode == 0:
            system_info["nvcc_version"] = result.stdout.split('\n')[0]
    except FileNotFoundError:
        system_info["nvcc_available"] = False
        system_info["nvcc_version"] = None
    
    return system_info