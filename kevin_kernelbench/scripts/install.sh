#!/bin/bash
# Kevin-32B 和 KernelBench 评估环境安装脚本

set -e

echo "开始安装Kevin-32B和KernelBench评估环境..."

# 检查Python版本
python_version=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "错误: 需要Python 3.8或更高版本，当前版本: $python_version"
    exit 1
fi

echo "Python版本检查通过: $python_version"

# 检查CUDA
if ! command -v nvcc &> /dev/null; then
    echo "错误: 未找到NVCC，请确保CUDA已正确安装"
    exit 1
fi

echo "CUDA检查通过: $(nvcc --version | head -n1)"

# 检查GPU
if ! python3 -c "import torch; print('CUDA可用:', torch.cuda.is_available())" 2>/dev/null; then
    echo "错误: PyTorch未安装或CUDA不可用"
    exit 1
fi

echo "GPU检查通过"

# 安装依赖
echo "安装Python依赖..."
pip install --upgrade pip
pip install -r requirements.txt

# 创建必要的目录
echo "创建项目目录..."
mkdir -p data/models
mkdir -p data/kernelbench
mkdir -p results
mkdir -p logs

# 设置权限
chmod +x run_evaluation.py

echo "安装完成！"
echo ""
echo "使用方法:"
echo "1. 检查系统要求: python3 run_evaluation.py --check-system"
echo "2. 运行评估: python3 run_evaluation.py"
echo ""
echo "配置文件: configs/config.yaml"
echo "日志文件: logs/kevin_evaluation.log"