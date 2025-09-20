#!/usr/bin/env python3
"""
Kevin-32B 和 KernelBench 评估环境安装脚本
"""

from setuptools import setup, find_packages

setup(
    name="kevin-kernelbench",
    version="1.0.0",
    description="Kevin-32B CUDA kernel generation and KernelBench evaluation framework",
    author="Wang Han",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.1.0",
        "transformers>=4.35.0",
        "accelerate>=0.24.0",
        "numpy>=1.24.0",
        "tqdm>=4.66.0",
        "pyyaml>=6.0",
        "requests>=2.31.0",
        "huggingface-hub>=0.17.0",
    ],
    extras_require={
        "dev": [
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
        ],
        "monitoring": [
            "swanlab>=0.3.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "kevin-eval=kevin_kernelbench.cli:main",
        ],
    },
)
