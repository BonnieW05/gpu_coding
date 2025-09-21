#!/bin/bash

# Kevin-32B 模型下载策略脚本
# 基于Hugging Face页面分析：https://huggingface.co/cognition-ai/Kevin-32B/tree/main

echo "========================================="
echo "Kevin-32B 模型下载策略"
echo "模型大小: 131GB"
echo "分析时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================="
echo ""

# 设置变量
MODEL_NAME="cognition-ai/Kevin-32B"
MODEL_SIZE_GB=131
PROJECT_DIR="/root/autodl-tmp/wanghan/gpu_coding/kevin_kernelbench"
MODEL_DIR="$PROJECT_DIR/data/models/kevin-32b"

echo "1. 存储空间检查..."
echo "==================="

# 检查可用空间
AVAILABLE_SPACE=$(df /dev/md0 | tail -1 | awk '{print $4}')
AVAILABLE_SPACE_GB=$((AVAILABLE_SPACE / 1024 / 1024))
REMAINING_SPACE=$((AVAILABLE_SPACE_GB - MODEL_SIZE_GB))
USAGE_AFTER=$((100 - (REMAINING_SPACE * 100 / 2500)))

echo "当前可用空间: ${AVAILABLE_SPACE_GB}GB"
echo "模型大小: ${MODEL_SIZE_GB}GB"
echo "下载后剩余空间: ${REMAINING_SPACE}GB"
echo "下载后使用率: ${USAGE_AFTER}%"
echo ""

# 风险评估
if [ $REMAINING_SPACE -gt 200 ]; then
    echo "✅ 存储风险评估: 低风险"
    echo "   剩余空间充足，可以安全下载"
elif [ $REMAINING_SPACE -gt 100 ]; then
    echo "⚠️  存储风险评估: 中等风险"
    echo "   剩余空间基本够用，建议监控"
else
    echo "❌ 存储风险评估: 高风险"
    echo "   剩余空间不足，不建议下载"
    exit 1
fi
echo ""

echo "2. 创建下载目录..."
echo "=================="
mkdir -p "$MODEL_DIR"
echo "✅ 创建目录: $MODEL_DIR"
echo ""

echo "3. 下载策略选择..."
echo "=================="
echo "推荐使用以下方法之一："
echo ""
echo "方法一：使用 huggingface-hub (推荐)"
echo "-----------------------------------"
echo "pip install huggingface-hub"
echo "huggingface-cli download $MODEL_NAME \\"
echo "    --local-dir $MODEL_DIR \\"
echo "    --local-dir-use-symlinks False"
echo ""

echo "方法二：使用 transformers 库"
echo "---------------------------"
echo "pip install transformers torch"
echo "python -c \""
echo "from transformers import AutoTokenizer, AutoModelForCausalLM"
echo "model_name = '$MODEL_NAME'"
echo "tokenizer = AutoTokenizer.from_pretrained(model_name)"
echo "model = AutoModelForCausalLM.from_pretrained("
echo "    model_name,"
echo "    torch_dtype='auto',"
echo "    device_map='auto',"
echo "    cache_dir='$MODEL_DIR'"
echo ")"
echo "\""
echo ""

echo "方法三：使用 git lfs"
echo "-------------------"
echo "git lfs install"
echo "git clone https://huggingface.co/$MODEL_NAME $MODEL_DIR"
echo ""

echo "4. 下载监控..."
echo "============="
echo "下载过程中可以使用以下命令监控："
echo ""
echo "# 监控下载进度"
echo "watch -n 5 'du -sh $MODEL_DIR'"
echo ""
echo "# 监控磁盘使用"
echo "watch -n 10 'df -h /dev/md0'"
echo ""
echo "# 监控网络流量"
echo "iftop -i eth0"
echo ""

echo "5. 下载后验证..."
echo "==============="
echo "下载完成后，验证文件完整性："
echo ""
echo "# 检查文件数量"
echo "ls -la $MODEL_DIR/*.safetensors | wc -l"
echo "echo '应该有29个.safetensors文件'"
echo ""
echo "# 检查总大小"
echo "du -sh $MODEL_DIR"
echo "echo '应该约为131GB'"
echo ""
echo "# 测试模型加载"
echo "python -c \""
echo "from transformers import AutoTokenizer"
echo "tokenizer = AutoTokenizer.from_pretrained('$MODEL_DIR')"
echo "print('模型加载成功！')"
echo "\""
echo ""

echo "6. 存储管理建议..."
echo "=================="
echo "下载完成后："
echo "✅ 定期监控存储使用情况"
echo "✅ 清理临时文件和缓存"
echo "✅ 考虑压缩不常用的模型文件"
echo "✅ 准备外部存储备选方案"
echo ""

echo "7. 配置文件更新..."
echo "=================="
echo "更新项目配置文件："
echo ""
echo "# configs/config.yaml"
echo "model:"
echo "  name: \"$MODEL_NAME\""
echo "  local_path: \"$MODEL_DIR\""
echo "  local_files_only: true"
echo "  torch_dtype: \"auto\""
echo "  device_map: \"auto\""
echo ""

echo "========================================="
echo "下载策略制定完成！"
echo "========================================="
echo ""
echo "⚠️  重要提醒："
echo "1. 模型大小131GB，下载时间较长"
echo "2. 建议使用支持断点续传的方法"
echo "3. 下载过程中保持网络稳定"
echo "4. 定期监控存储使用情况"
echo ""
echo "🚀 开始下载前，请确保："
echo "1. 网络连接稳定"
echo "2. 存储空间充足"
echo "3. 有足够的下载时间"
echo ""
