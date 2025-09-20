#!/usr/bin/env python3
"""
Kevin-32B 和 KernelBench 评估运行脚本

这个脚本提供了接口来运行完整的评估流程。
"""

import sys
import os
from pathlib import Path

# 添加src目录到Python路径
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

from kevin_kernelbench.cli import main

if __name__ == "__main__":
    main()
