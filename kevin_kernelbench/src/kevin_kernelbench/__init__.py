"""
Kevin-32B CUDA Kernel Generation and KernelBench Evaluation Framework

这个包提供了使用Kevin-32B模型生成CUDA内核代码，
并使用KernelBench进行性能评估的完整框架。
"""

__version__ = "1.0.0"
__author__ = "Wang Han"

from .model import KevinModel
from .evaluator import KernelBenchEvaluator
from .utils import setup_logging, load_config

__all__ = [
    "KevinModel",
    "KernelBenchEvaluator", 
    "setup_logging",
    "load_config",
]
