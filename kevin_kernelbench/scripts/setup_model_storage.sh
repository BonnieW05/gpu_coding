#!/bin/bash

# 无sudo权限的模型存储设置脚本
# 适用于用户 wanghan

echo "========================================="
echo "Kevin-32B 模型存储设置脚本"
echo "用户: $(whoami)"
echo "时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================="
echo ""

# 设置变量
PROJECT_DIR="/root/autodl-tmp/wanghan/gpu_coding/kevin_kernelbench"
MODEL_DIR="$PROJECT_DIR/data/models/kevin-32b"
USER_HOME="/root/autodl-tmp/wanghan"
USER_MODEL_DIR="$USER_HOME/models/kevin-32b"

echo "1. 检查当前目录和权限..."
echo "当前目录: $(pwd)"
echo "项目目录: $PROJECT_DIR"
echo ""

# 检查项目目录是否存在
if [ ! -d "$PROJECT_DIR" ]; then
    echo "❌ 项目目录不存在: $PROJECT_DIR"
    echo "请确保您在正确的项目目录中运行此脚本"
    exit 1
fi

echo "2. 创建模型存储目录..."

# 方案一：在项目目录下创建（推荐）
echo "方案一：在项目目录下创建模型存储"
if [ ! -d "$MODEL_DIR" ]; then
    mkdir -p "$MODEL_DIR"
    if [ $? -eq 0 ]; then
        echo "✅ 成功创建目录: $MODEL_DIR"
    else
        echo "❌ 创建目录失败: $MODEL_DIR"
    fi
else
    echo "✅ 目录已存在: $MODEL_DIR"
fi

# 检查空间
echo ""
echo "3. 检查存储空间..."
echo "项目目录可用空间:"
df -h "$PROJECT_DIR" | tail -1
echo ""
echo "模型目录大小:"
du -sh "$MODEL_DIR" 2>/dev/null || echo "目录为空"

# 方案二：在用户主目录下创建（备选）
echo ""
echo "方案二：在用户主目录下创建模型存储"
if [ ! -d "$USER_MODEL_DIR" ]; then
    mkdir -p "$USER_MODEL_DIR"
    if [ $? -eq 0 ]; then
        echo "✅ 成功创建目录: $USER_MODEL_DIR"
    else
        echo "❌ 创建目录失败: $USER_MODEL_DIR"
    fi
else
    echo "✅ 目录已存在: $USER_MODEL_DIR"
fi

echo ""
echo "4. 设置目录权限..."
chmod 755 "$MODEL_DIR" 2>/dev/null
chmod 755 "$USER_MODEL_DIR" 2>/dev/null
echo "✅ 权限设置完成"

echo ""
echo "5. 创建软链接（可选）..."
# 在项目目录中创建指向用户主目录的软链接
if [ -d "$USER_MODEL_DIR" ] && [ ! -L "$PROJECT_DIR/data/models/kevin-32b-user" ]; then
    ln -s "$USER_MODEL_DIR" "$PROJECT_DIR/data/models/kevin-32b-user"
    echo "✅ 创建软链接: $PROJECT_DIR/data/models/kevin-32b-user -> $USER_MODEL_DIR"
fi

echo ""
echo "6. 验证设置..."
echo "项目目录下的模型存储:"
ls -la "$PROJECT_DIR/data/models/"
echo ""
echo "用户主目录下的模型存储:"
ls -la "$USER_HOME/models/" 2>/dev/null || echo "用户主目录下无models目录"

echo ""
echo "========================================="
echo "设置完成！"
echo "========================================="
echo ""
echo "推荐使用以下路径存储Kevin-32B模型:"
echo "主选: $MODEL_DIR"
echo "备选: $USER_MODEL_DIR"
echo ""
echo "下一步操作:"
echo "1. 在本地下载Kevin-32B模型"
echo "2. 使用以下命令上传到服务器:"
echo "   rsync -avz --progress /path/to/local/Kevin-32B/ username@server:$MODEL_DIR/"
echo "   或"
echo "   scp -r /path/to/local/Kevin-32B/* username@server:$MODEL_DIR/"
echo ""
echo "3. 更新配置文件中的模型路径"
echo "   model.local_path: \"$MODEL_DIR\""
echo ""
