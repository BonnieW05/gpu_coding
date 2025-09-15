#!/bin/bash

# 服务器特定配置文件
# 基于服务器信息.md中的硬件配置
# 用户: wanghan, 服务器: 双RTX 4090 GPU服务器

# 服务器基本信息
export SERVER_USER="wanghan"
export SERVER_UID="1010"
export SERVER_GID="1010"
export SERVER_OS="Ubuntu Linux"
export SERVER_KERNEL="5.15.0-86-generic"
export SERVER_ARCH="x86_64"
export SERVER_CONTAINER="autodl-container"

# 硬件配置
export SERVER_CPU="AMD EPYC 9654"
export SERVER_CPU_CORES="96"
export SERVER_CPU_THREADS="192"
export SERVER_MEMORY_TOTAL="755GB"
export SERVER_MEMORY_AVAILABLE="736GB"

# GPU配置
export SERVER_GPU_COUNT="2"
export SERVER_GPU_MODEL="NVIDIA GeForce RTX 4090"
export SERVER_GPU_MEMORY="24GB"
export SERVER_CUDA_VERSION="12.4"
export SERVER_DRIVER_VERSION="550.78"

# 存储配置
export SERVER_ROOT_PARTITION="30GB"
export SERVER_ROOT_USAGE="76%"
export SERVER_DATA_PARTITION="7TB"
export SERVER_DATA_USAGE="54%"
export SERVER_SHARED_DATA="20TB"
export SERVER_SHARED_USAGE="9%"

# 目录配置
export SERVER_USER_HOME="/root/autodl-tmp/wanghan"
export SERVER_CACHE_DIR="/root/autodl-tmp/wanghan/.cache"
export SERVER_LEARNING_DIR="/root/autodl-tmp/wanghan/gpu_learning"
export SERVER_PROJECTS_DIR="/root/autodl-tmp/wanghan/projects"

# 网络配置
export SERVER_INTERNAL_IP="172.17.0.2"
export SERVER_LOOPBACK="127.0.0.1"
export SERVER_INTERFACE="eth0"

# 环境变量
export CUDA_HOME="/usr/local/cuda-12.4"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

# PyTorch配置
export PYTORCH_CUDA_VERSION="cu124"
export PYTORCH_INDEX_URL="https://download.pytorch.org/whl/cu124"

# 多用户环境提醒
export SERVER_MULTI_USER="true"
export SERVER_OTHER_USERS="chunyu,pengfei,qingchen,ranwen,rundong,tianyu,wanghaoxu,weijia,weiqi,yicheng,yiwei"

# 系统负载信息
export SERVER_LOAD_1MIN="0.64"
export SERVER_LOAD_5MIN="0.88"
export SERVER_LOAD_15MIN="1.34"
export SERVER_UPTIME="445天10小时56分钟"

# 功能函数
function show_server_info() {
    echo "=== 服务器配置信息 ==="
    echo "用户: $SERVER_USER (UID: $SERVER_UID)"
    echo "操作系统: $SERVER_OS $SERVER_KERNEL"
    echo "CPU: $SERVER_CPU ($SERVER_CPU_CORES核/$SERVER_CPU_THREADS线程)"
    echo "内存: $SERVER_MEMORY_TOTAL (可用: $SERVER_MEMORY_AVAILABLE)"
    echo "GPU: $SERVER_GPU_COUNT x $SERVER_GPU_MODEL ($SERVER_GPU_MEMORY显存)"
    echo "CUDA: $SERVER_CUDA_VERSION (驱动: $SERVER_DRIVER_VERSION)"
    echo "用户目录: $SERVER_USER_HOME"
    echo "学习目录: $SERVER_LEARNING_DIR"
    echo "多用户环境: $SERVER_MULTI_USER"
    echo "========================"
}

function check_gpu_status() {
    echo "=== GPU状态检查 ==="
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
    else
        echo "nvidia-smi未找到"
    fi
    echo "=================="
}

function check_disk_usage() {
    echo "=== 磁盘使用情况 ==="
    echo "根分区: $SERVER_ROOT_PARTITION (使用率: $SERVER_ROOT_USAGE)"
    echo "数据分区: $SERVER_DATA_PARTITION (使用率: $SERVER_DATA_USAGE)"
    echo "共享数据: $SERVER_SHARED_DATA (使用率: $SERVER_SHARED_USAGE)"
    echo "=================="
}

function setup_environment() {
    echo "=== 设置环境变量 ==="
    echo "设置CUDA环境..."
    export CUDA_HOME="/usr/local/cuda-12.4"
    export PATH="$CUDA_HOME/bin:$PATH"
    export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
    
    echo "设置PyTorch环境..."
    export PYTORCH_CUDA_VERSION="cu124"
    export PYTORCH_INDEX_URL="https://download.pytorch.org/whl/cu124"
    
    echo "创建必要目录..."
    mkdir -p "$SERVER_USER_HOME"/{gpu_learning,projects,.cache}
    
    echo "环境设置完成"
    echo "================"
}

# 如果直接运行此脚本，显示服务器信息
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    show_server_info
    check_gpu_status
    check_disk_usage
    echo ""
    echo "使用方法:"
    echo "  source $0  # 加载环境变量"
    echo "  show_server_info  # 显示服务器信息"
    echo "  check_gpu_status  # 检查GPU状态"
    echo "  check_disk_usage  # 检查磁盘使用"
    echo "  setup_environment  # 设置环境"
fi
