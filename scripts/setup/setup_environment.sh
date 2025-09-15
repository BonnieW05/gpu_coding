#!/bin/bash

# GPU编程环境设置脚本
# 自动安装和配置PyTorch、Triton、ThunderKitten等依赖

set -e

echo "🚀 开始设置GPU编程学习环境..."

# 检查Python版本
echo "🐍 检查Python版本..."
python_version=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "Python版本: $python_version"

if [[ $(echo "$python_version < 3.8" | bc -l) -eq 1 ]]; then
    echo "❌ Python版本过低，需要3.8+"
    exit 1
fi

# 检查CUDA
echo "🔍 检查CUDA环境..."
if command -v nvcc &> /dev/null; then
    cuda_version=$(nvcc --version | grep "release" | cut -d' ' -f6 | cut -d',' -f1)
    echo "✅ CUDA版本: $cuda_version"
else
    echo "⚠️  CUDA未安装，某些功能可能不可用"
fi

# 检查GPU
echo "🖥️  检查GPU..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits
else
    echo "⚠️  nvidia-smi未找到，可能没有NVIDIA GPU"
fi

# 创建虚拟环境
echo "📦 创建虚拟环境..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✅ 虚拟环境已创建"
else
    echo "✅ 虚拟环境已存在"
fi

# 激活虚拟环境
echo "🔧 激活虚拟环境..."
source venv/bin/activate

# 升级pip
echo "⬆️  升级pip..."
pip install --upgrade pip

# 安装PyTorch
echo "🔥 安装PyTorch..."
if command -v nvcc &> /dev/null; then
    # 有CUDA，安装CUDA版本
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
else
    # 无CUDA，安装CPU版本
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# 安装Triton
echo "⚡ 安装Triton..."
pip install triton

# 安装ThunderKitten (如果可用)
echo "🐱 尝试安装ThunderKitten..."
pip install thunderkitten || echo "⚠️  ThunderKitten安装失败，可能暂不可用"

# 安装其他依赖
echo "📚 安装其他依赖..."
pip install -r requirements.txt

# 验证安装
echo "✅ 验证安装..."
python3 -c "
import torch
print(f'PyTorch版本: {torch.__version__}')
print(f'CUDA可用: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA版本: {torch.version.cuda}')
    print(f'GPU数量: {torch.cuda.device_count()}')
    print(f'GPU名称: {torch.cuda.get_device_name()}')

try:
    import triton
    print(f'Triton版本: {triton.__version__}')
except ImportError:
    print('Triton未安装')

try:
    import thunderkitten
    print('ThunderKitten已安装')
except ImportError:
    print('ThunderKitten未安装')
"

echo ""
echo "🎉 环境设置完成!"
echo ""
echo "📝 使用方法:"
echo "  source venv/bin/activate  # 激活虚拟环境"
echo "  python pytorch/basics/01_tensor_basics.py  # 运行PyTorch示例"
echo "  python triton/basics/01_vector_add.py  # 运行Triton示例"
echo "  python shared/utils/device_utils.py  # 查看GPU信息"
echo ""
echo "🔧 编译CUDA示例:"
echo "  nvcc -o hello_cuda cuda/basics/01_hello_cuda.cu"
echo "  ./hello_cuda"
