#!/bin/bash

# 用户权限检查脚本
# 输出文件：user_permissions_info.txt

OUTPUT_FILE="user_permissions_info.txt"
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

echo "=========================================" > $OUTPUT_FILE
echo "用户权限检查报告" >> $OUTPUT_FILE
echo "检查时间: $TIMESTAMP" >> $OUTPUT_FILE
echo "用户: $(whoami)" >> $OUTPUT_FILE
echo "=========================================" >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE

echo "正在检查用户权限和可访问目录..."

# 1. 用户基本信息
echo "1. 用户基本信息" >> $OUTPUT_FILE
echo "=================" >> $OUTPUT_FILE
echo "当前用户: $(whoami)" >> $OUTPUT_FILE
echo "用户ID: $(id)" >> $OUTPUT_FILE
echo "主目录: $HOME" >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE

# 2. 检查 /autodl-pub 目录权限
echo "2. /autodl-pub 目录权限检查" >> $OUTPUT_FILE
echo "===========================" >> $OUTPUT_FILE
if [ -d "/autodl-pub" ]; then
    echo "/autodl-pub 目录存在" >> $OUTPUT_FILE
    ls -ld /autodl-pub >> $OUTPUT_FILE
    echo "" >> $OUTPUT_FILE
    
    # 检查 /autodl-pub/data 权限
    if [ -d "/autodl-pub/data" ]; then
        echo "/autodl-pub/data 目录权限:" >> $OUTPUT_FILE
        ls -ld /autodl-pub/data >> $OUTPUT_FILE
        echo "" >> $OUTPUT_FILE
        
        # 尝试创建测试目录
        echo "尝试在 /autodl-pub/data 创建测试目录:" >> $OUTPUT_FILE
        if mkdir -p /autodl-pub/data/test_$(whoami)_$(date +%s) 2>/dev/null; then
            echo "✅ 可以在 /autodl-pub/data 创建目录" >> $OUTPUT_FILE
            rmdir /autodl-pub/data/test_$(whoami)_* 2>/dev/null
        else
            echo "❌ 无法在 /autodl-pub/data 创建目录" >> $OUTPUT_FILE
        fi
        echo "" >> $OUTPUT_FILE
    fi
    
    # 检查 /autodl-pub 根目录权限
    echo "尝试在 /autodl-pub 创建测试目录:" >> $OUTPUT_FILE
    if mkdir -p /autodl-pub/test_$(whoami)_$(date +%s) 2>/dev/null; then
        echo "✅ 可以在 /autodl-pub 创建目录" >> $OUTPUT_FILE
        rmdir /autodl-pub/test_$(whoami)_* 2>/dev/null
    else
        echo "❌ 无法在 /autodl-pub 创建目录" >> $OUTPUT_FILE
    fi
    echo "" >> $OUTPUT_FILE
else
    echo "/autodl-pub 目录不存在" >> $OUTPUT_FILE
    echo "" >> $OUTPUT_FILE
fi

# 3. 检查用户主目录空间
echo "3. 用户主目录空间检查" >> $OUTPUT_FILE
echo "=====================" >> $OUTPUT_FILE
echo "主目录: $HOME" >> $OUTPUT_FILE
df -h $HOME >> $OUTPUT_FILE
echo "主目录大小:" >> $OUTPUT_FILE
du -sh $HOME >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE

# 4. 检查 /tmp 目录权限
echo "4. /tmp 目录权限检查" >> $OUTPUT_FILE
echo "===================" >> $OUTPUT_FILE
echo "/tmp 目录权限:" >> $OUTPUT_FILE
ls -ld /tmp >> $OUTPUT_FILE
echo "尝试在 /tmp 创建测试目录:" >> $OUTPUT_FILE
if mkdir -p /tmp/test_$(whoami)_$(date +%s) 2>/dev/null; then
    echo "✅ 可以在 /tmp 创建目录" >> $OUTPUT_FILE
    rmdir /tmp/test_$(whoami)_* 2>/dev/null
else
    echo "❌ 无法在 /tmp 创建目录" >> $OUTPUT_FILE
fi
echo "" >> $OUTPUT_FILE

# 5. 检查当前项目目录权限
echo "5. 当前项目目录权限检查" >> $OUTPUT_FILE
echo "=======================" >> $OUTPUT_FILE
echo "当前目录: $(pwd)" >> $OUTPUT_FILE
ls -ld . >> $OUTPUT_FILE
echo "data 目录权限:" >> $OUTPUT_FILE
if [ -d "data" ]; then
    ls -ld data >> $OUTPUT_FILE
    echo "data/models 目录权限:" >> $OUTPUT_FILE
    if [ -d "data/models" ]; then
        ls -ld data/models >> $OUTPUT_FILE
    else
        echo "data/models 目录不存在，尝试创建:" >> $OUTPUT_FILE
        if mkdir -p data/models 2>/dev/null; then
            echo "✅ 成功创建 data/models 目录" >> $OUTPUT_FILE
        else
            echo "❌ 无法创建 data/models 目录" >> $OUTPUT_FILE
        fi
    fi
else
    echo "data 目录不存在，尝试创建:" >> $OUTPUT_FILE
    if mkdir -p data/models 2>/dev/null; then
        echo "✅ 成功创建 data 目录" >> $OUTPUT_FILE
    else
        echo "❌ 无法创建 data 目录" >> $OUTPUT_FILE
    fi
fi
echo "" >> $OUTPUT_FILE

# 6. 检查其他可能的存储位置
echo "6. 其他可能的存储位置" >> $OUTPUT_FILE
echo "=====================" >> $OUTPUT_FILE

for dir in "/scratch" "/data" "/workspace" "/shared" "/mnt/shared"; do
    echo "$dir 目录:" >> $OUTPUT_FILE
    if [ -d "$dir" ]; then
        ls -ld "$dir" >> $OUTPUT_FILE
        echo "尝试在 $dir 创建测试目录:" >> $OUTPUT_FILE
        if mkdir -p "$dir/test_$(whoami)_$(date +%s)" 2>/dev/null; then
            echo "✅ 可以在 $dir 创建目录" >> $OUTPUT_FILE
            rmdir "$dir/test_$(whoami)_*" 2>/dev/null
        else
            echo "❌ 无法在 $dir 创建目录" >> $OUTPUT_FILE
        fi
    else
        echo "$dir 目录不存在" >> $OUTPUT_FILE
    fi
    echo "" >> $OUTPUT_FILE
done

# 7. 检查环境变量中的存储路径
echo "7. 环境变量中的存储路径" >> $OUTPUT_FILE
echo "=======================" >> $OUTPUT_FILE
echo "TMPDIR: $TMPDIR" >> $OUTPUT_FILE
echo "XDG_DATA_HOME: $XDG_DATA_HOME" >> $OUTPUT_FILE
echo "XDG_CACHE_HOME: $XDG_CACHE_HOME" >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE

# 8. 推荐方案
echo "8. 推荐存储方案" >> $OUTPUT_FILE
echo "===============" >> $OUTPUT_FILE
echo "基于权限检查结果，推荐以下方案:" >> $OUTPUT_FILE
echo "1. 如果 /autodl-pub/data 可写：使用 /autodl-pub/data/models/kevin-32b" >> $OUTPUT_FILE
echo "2. 如果 /autodl-pub 可写：使用 /autodl-pub/models/kevin-32b" >> $OUTPUT_FILE
echo "3. 如果都不可写：使用用户主目录下的 models 目录" >> $OUTPUT_FILE
echo "4. 临时方案：使用 /tmp 目录（注意可能被清理）" >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE

echo "=========================================" >> $OUTPUT_FILE
echo "检查完成时间: $(date '+%Y-%m-%d %H:%M:%S')" >> $OUTPUT_FILE
echo "=========================================" >> $OUTPUT_FILE

echo "权限检查完成！结果已保存到: $OUTPUT_FILE"
echo "请查看文件内容，我会根据结果为您提供最佳的无sudo权限存储方案。"
