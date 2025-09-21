#!/bin/bash

# 详细存储分析脚本
# 输出文件：storage_analysis_detailed.txt

OUTPUT_FILE="storage_analysis_detailed.txt"
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

echo "=========================================" > $OUTPUT_FILE
echo "服务器存储详细分析报告" >> $OUTPUT_FILE
echo "分析时间: $TIMESTAMP" >> $OUTPUT_FILE
echo "用户: $(whoami)" >> $OUTPUT_FILE
echo "=========================================" >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE

echo "正在分析服务器存储使用情况..."

# 1. 整体磁盘使用情况
echo "1. 整体磁盘使用情况" >> $OUTPUT_FILE
echo "===================" >> $OUTPUT_FILE
echo "所有挂载点:" >> $OUTPUT_FILE
df -h >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE

# 2. 主存储设备详细信息
echo "2. 主存储设备详细信息" >> $OUTPUT_FILE
echo "=====================" >> $OUTPUT_FILE
echo "主存储设备 /dev/md0 详细信息:" >> $OUTPUT_FILE
df -h /dev/md0 >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE

# 3. 用户目录使用情况
echo "3. 用户目录使用情况" >> $OUTPUT_FILE
echo "===================" >> $OUTPUT_FILE
USER_HOME="/root/autodl-tmp/wanghan"
echo "用户主目录: $USER_HOME" >> $OUTPUT_FILE
echo "用户主目录大小:" >> $OUTPUT_FILE
du -sh "$USER_HOME" >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE

echo "用户主目录详细使用情况:" >> $OUTPUT_FILE
du -sh "$USER_HOME"/* 2>/dev/null | sort -hr >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE

# 4. 项目目录使用情况
echo "4. 项目目录使用情况" >> $OUTPUT_FILE
echo "===================" >> $OUTPUT_FILE
PROJECT_DIR="/root/autodl-tmp/wanghan/gpu_coding/kevin_kernelbench"
echo "项目目录: $PROJECT_DIR" >> $OUTPUT_FILE
if [ -d "$PROJECT_DIR" ]; then
    echo "项目目录大小:" >> $OUTPUT_FILE
    du -sh "$PROJECT_DIR" >> $OUTPUT_FILE
    echo "" >> $OUTPUT_FILE
    
    echo "项目目录详细使用情况:" >> $OUTPUT_FILE
    du -sh "$PROJECT_DIR"/* 2>/dev/null | sort -hr >> $OUTPUT_FILE
    echo "" >> $OUTPUT_FILE
    
    echo "data目录使用情况:" >> $OUTPUT_FILE
    if [ -d "$PROJECT_DIR/data" ]; then
        du -sh "$PROJECT_DIR/data"/* 2>/dev/null | sort -hr >> $OUTPUT_FILE
    else
        echo "data目录不存在" >> $OUTPUT_FILE
    fi
    echo "" >> $OUTPUT_FILE
else
    echo "项目目录不存在" >> $OUTPUT_FILE
    echo "" >> $OUTPUT_FILE
fi

# 5. 其他用户目录使用情况
echo "5. 其他用户目录使用情况" >> $OUTPUT_FILE
echo "=======================" >> $OUTPUT_FILE
AUTODL_TMP="/root/autodl-tmp"
echo "autodl-tmp目录总大小:" >> $OUTPUT_FILE
du -sh "$AUTODL_TMP" >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE

echo "autodl-tmp下各用户目录大小:" >> $OUTPUT_FILE
du -sh "$AUTODL_TMP"/* 2>/dev/null | sort -hr >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE

# 6. 系统目录使用情况
echo "6. 系统目录使用情况" >> $OUTPUT_FILE
echo "===================" >> $OUTPUT_FILE
echo "根目录使用情况:" >> $OUTPUT_FILE
du -sh /* 2>/dev/null | sort -hr | head -10 >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE

# 7. 大文件查找
echo "7. 大文件查找" >> $OUTPUT_FILE
echo "=============" >> $OUTPUT_FILE
echo "用户目录下大于1GB的文件:" >> $OUTPUT_FILE
find "$USER_HOME" -type f -size +1G -exec ls -lh {} \; 2>/dev/null | head -10 >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE

echo "项目目录下大于100MB的文件:" >> $OUTPUT_FILE
find "$PROJECT_DIR" -type f -size +100M -exec ls -lh {} \; 2>/dev/null | head -10 >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE

# 8. 存储风险评估
echo "8. 存储风险评估" >> $OUTPUT_FILE
echo "===============" >> $OUTPUT_FILE

# 获取可用空间
AVAILABLE_SPACE=$(df /dev/md0 | tail -1 | awk '{print $4}')
AVAILABLE_SPACE_GB=$((AVAILABLE_SPACE / 1024 / 1024))

echo "当前可用空间: ${AVAILABLE_SPACE_GB}GB" >> $OUTPUT_FILE
echo "Kevin-32B模型预估大小: 60-80GB" >> $OUTPUT_FILE
echo "用户当前使用: 13GB" >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE

if [ $AVAILABLE_SPACE_GB -gt 100 ]; then
    echo "✅ 存储风险评估: 低风险" >> $OUTPUT_FILE
    echo "   可用空间充足，可以安全存储Kevin-32B模型" >> $OUTPUT_FILE
elif [ $AVAILABLE_SPACE_GB -gt 80 ]; then
    echo "⚠️  存储风险评估: 中等风险" >> $OUTPUT_FILE
    echo "   可用空间基本够用，但建议监控使用情况" >> $OUTPUT_FILE
else
    echo "❌ 存储风险评估: 高风险" >> $OUTPUT_FILE
    echo "   可用空间不足，不建议存储大型模型" >> $OUTPUT_FILE
fi
echo "" >> $OUTPUT_FILE

# 9. 存储建议
echo "9. 存储建议" >> $OUTPUT_FILE
echo "===========" >> $OUTPUT_FILE
echo "基于分析结果，提供以下建议:" >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE

if [ $AVAILABLE_SPACE_GB -gt 100 ]; then
    echo "✅ 推荐方案:" >> $OUTPUT_FILE
    echo "   1. 在项目目录下存储模型: $PROJECT_DIR/data/models/kevin-32b" >> $OUTPUT_FILE
    echo "   2. 定期清理临时文件和缓存" >> $OUTPUT_FILE
    echo "   3. 监控存储使用情况" >> $OUTPUT_FILE
    echo "" >> $OUTPUT_FILE
    echo "⚠️  注意事项:" >> $OUTPUT_FILE
    echo "   1. 模型文件较大，上传时间较长" >> $OUTPUT_FILE
    echo "   2. 建议使用rsync进行断点续传" >> $OUTPUT_FILE
    echo "   3. 定期备份重要数据" >> $OUTPUT_FILE
else
    echo "❌ 不推荐在项目目录存储大型模型" >> $OUTPUT_FILE
    echo "   建议使用外部存储或云存储" >> $OUTPUT_FILE
fi
echo "" >> $OUTPUT_FILE

# 10. 存储监控命令
echo "10. 存储监控命令" >> $OUTPUT_FILE
echo "================" >> $OUTPUT_FILE
echo "定期监控存储使用情况的命令:" >> $OUTPUT_FILE
echo "df -h                    # 查看整体磁盘使用" >> $OUTPUT_FILE
echo "du -sh /root/autodl-tmp/wanghan  # 查看用户目录大小" >> $OUTPUT_FILE
echo "du -sh /root/autodl-tmp/wanghan/gpu_coding/kevin_kernelbench  # 查看项目大小" >> $OUTPUT_FILE
echo "find /root/autodl-tmp/wanghan -type f -size +1G  # 查找大文件" >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE

echo "=========================================" >> $OUTPUT_FILE
echo "分析完成时间: $(date '+%Y-%m-%d %H:%M:%S')" >> $OUTPUT_FILE
echo "=========================================" >> $OUTPUT_FILE

echo "存储分析完成！结果已保存到: $OUTPUT_FILE"
echo "请查看详细分析结果，我会根据结果为您提供专业的存储建议。"
