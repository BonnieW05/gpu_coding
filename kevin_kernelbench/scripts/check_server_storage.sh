#!/bin/bash

# 服务器存储检查脚本
# 输出文件：server_storage_info.txt

OUTPUT_FILE="server_storage_info.txt"
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

echo "=========================================" > $OUTPUT_FILE
echo "服务器存储检查报告" >> $OUTPUT_FILE
echo "检查时间: $TIMESTAMP" >> $OUTPUT_FILE
echo "=========================================" >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE

echo "正在检查服务器存储情况，请稍候..."

# 1. 磁盘使用情况
echo "1. 磁盘使用情况" >> $OUTPUT_FILE
echo "=================" >> $OUTPUT_FILE
df -h >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE

# 2. 当前目录信息
echo "2. 当前目录信息" >> $OUTPUT_FILE
echo "=================" >> $OUTPUT_FILE
echo "当前工作目录: $(pwd)" >> $OUTPUT_FILE
echo "当前目录大小:" >> $OUTPUT_FILE
du -sh . >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE

# 3. 当前目录下各子目录大小
echo "3. 当前目录下各子目录大小" >> $OUTPUT_FILE
echo "=========================" >> $OUTPUT_FILE
du -sh * 2>/dev/null | sort -hr >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE

# 4. 查找 kevin_kernelbench 项目
echo "4. 查找 kevin_kernelbench 项目" >> $OUTPUT_FILE
echo "=============================" >> $OUTPUT_FILE
KEVIN_PATH=$(find / -name "kevin_kernelbench" -type d 2>/dev/null | head -5)
if [ -n "$KEVIN_PATH" ]; then
    echo "找到的 kevin_kernelbench 项目路径:" >> $OUTPUT_FILE
    echo "$KEVIN_PATH" >> $OUTPUT_FILE
    echo "" >> $OUTPUT_FILE
    
    # 检查项目大小
    for path in $KEVIN_PATH; do
        echo "项目路径: $path" >> $OUTPUT_FILE
        echo "项目大小:" >> $OUTPUT_FILE
        du -sh "$path" >> $OUTPUT_FILE
        echo "" >> $OUTPUT_FILE
        
        # 检查 data 目录
        if [ -d "$path/data" ]; then
            echo "data 目录大小:" >> $OUTPUT_FILE
            du -sh "$path/data" >> $OUTPUT_FILE
            echo "data 目录结构:" >> $OUTPUT_FILE
            find "$path/data" -type d -maxdepth 2 2>/dev/null >> $OUTPUT_FILE
        else
            echo "data 目录不存在" >> $OUTPUT_FILE
        fi
        echo "" >> $OUTPUT_FILE
    done
else
    echo "未找到 kevin_kernelbench 项目" >> $OUTPUT_FILE
    echo "" >> $OUTPUT_FILE
fi

# 5. 系统存储位置检查
echo "5. 系统存储位置检查" >> $OUTPUT_FILE
echo "===================" >> $OUTPUT_FILE

# 检查 /tmp
echo "/tmp 目录:" >> $OUTPUT_FILE
df -h /tmp >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE

# 检查 /opt
echo "/opt 目录:" >> $OUTPUT_FILE
if [ -d "/opt" ]; then
    df -h /opt >> $OUTPUT_FILE
    echo "/opt 目录大小:" >> $OUTPUT_FILE
    du -sh /opt >> $OUTPUT_FILE
else
    echo "/opt 目录不存在" >> $OUTPUT_FILE
fi
echo "" >> $OUTPUT_FILE

# 检查 /usr/local
echo "/usr/local 目录:" >> $OUTPUT_FILE
df -h /usr/local >> $OUTPUT_FILE
echo "/usr/local 目录大小:" >> $OUTPUT_FILE
du -sh /usr/local >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE

# 检查用户主目录
echo "用户主目录:" >> $OUTPUT_FILE
df -h ~ >> $OUTPUT_FILE
echo "用户主目录大小:" >> $OUTPUT_FILE
du -sh ~ >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE

# 6. 常见模型存储目录
echo "6. 常见模型存储目录" >> $OUTPUT_FILE
echo "===================" >> $OUTPUT_FILE

for dir in /models /data /storage /mnt; do
    echo "$dir 目录:" >> $OUTPUT_FILE
    if [ -d "$dir" ]; then
        df -h "$dir" >> $OUTPUT_FILE
        echo "$dir 目录大小:" >> $OUTPUT_FILE
        du -sh "$dir" >> $OUTPUT_FILE
        echo "$dir 目录内容:" >> $OUTPUT_FILE
        ls -la "$dir" 2>/dev/null | head -10 >> $OUTPUT_FILE
    else
        echo "$dir 目录不存在" >> $OUTPUT_FILE
    fi
    echo "" >> $OUTPUT_FILE
done

# 7. 检查 /mnt 下的挂载点
echo "7. /mnt 下的挂载点" >> $OUTPUT_FILE
echo "==================" >> $OUTPUT_FILE
if [ -d "/mnt" ]; then
    echo "/mnt 目录内容:" >> $OUTPUT_FILE
    ls -la /mnt/ >> $OUTPUT_FILE
    echo "" >> $OUTPUT_FILE
    
    # 检查 /mnt 下每个子目录的大小
    for subdir in /mnt/*; do
        if [ -d "$subdir" ]; then
            echo "$subdir 大小:" >> $OUTPUT_FILE
            du -sh "$subdir" >> $OUTPUT_FILE
        fi
    done
else
    echo "/mnt 目录不存在" >> $OUTPUT_FILE
fi
echo "" >> $OUTPUT_FILE

# 8. 内存信息
echo "8. 系统内存信息" >> $OUTPUT_FILE
echo "===============" >> $OUTPUT_FILE
free -h >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE

# 9. GPU 信息
echo "9. GPU 信息" >> $OUTPUT_FILE
echo "===========" >> $OUTPUT_FILE
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi >> $OUTPUT_FILE
else
    echo "nvidia-smi 不可用" >> $OUTPUT_FILE
fi
echo "" >> $OUTPUT_FILE

# 10. 总结和建议
echo "10. 存储建议" >> $OUTPUT_FILE
echo "=============" >> $OUTPUT_FILE
echo "基于以上信息，建议的模型存储位置:" >> $OUTPUT_FILE
echo "1. 优先选择有足够空间（>150GB）的分区" >> $OUTPUT_FILE
echo "2. 考虑使用 /opt、/usr/local 或专门的 /data 目录" >> $OUTPUT_FILE
echo "3. 避免使用 /tmp 目录（可能被清理）" >> $OUTPUT_FILE
echo "4. 确保目标目录有读写权限" >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE

echo "=========================================" >> $OUTPUT_FILE
echo "检查完成时间: $(date '+%Y-%m-%d %H:%M:%S')" >> $OUTPUT_FILE
echo "=========================================" >> $OUTPUT_FILE

echo "检查完成！结果已保存到: $OUTPUT_FILE"
echo "请查看文件内容并告诉我结果，我会为您推荐最佳的模型存储位置。"
