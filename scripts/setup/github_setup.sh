#!/bin/bash

# GitHub仓库设置脚本
# 使用方法: ./github_setup.sh <repository_name>

set -e

# 检查参数
if [ $# -eq 0 ]; then
    echo "使用方法: $0 <repository_name>"
    echo "例如: $0 gpu-programming-learning"
    exit 1
fi

REPO_NAME=$1
USERNAME=$(git config user.name)

echo "🚀 设置GitHub仓库: $REPO_NAME"
echo "👤 用户名: $USERNAME"

# 检查GitHub CLI是否安装
if ! command -v gh &> /dev/null; then
    echo "❌ GitHub CLI (gh) 未安装"
    echo "请先安装GitHub CLI:"
    echo "  macOS: brew install gh"
    echo "  Ubuntu: sudo apt install gh"
    echo "  Windows: winget install GitHub.cli"
    exit 1
fi

# 检查是否已登录GitHub
if ! gh auth status &> /dev/null; then
    echo "❌ 未登录GitHub"
    echo "请先登录: gh auth login"
    exit 1
fi

echo "✅ GitHub CLI已就绪"

# 创建GitHub仓库
echo "📦 创建GitHub仓库..."
gh repo create $REPO_NAME --public --description "GPU编程学习项目 - PyTorch, Triton, ThunderKitten, CUDA" --clone=false

# 添加远程仓库
echo "🔗 添加远程仓库..."
git remote add origin git@github.com:$USERNAME/$REPO_NAME.git

# 设置默认分支为main
echo "🌿 设置默认分支..."
git branch -M main

# 推送到GitHub
echo "📤 推送到GitHub..."
git push -u origin main

echo "✅ GitHub仓库设置完成!"
echo "🔗 仓库地址: https://github.com/$USERNAME/$REPO_NAME"
echo ""
echo "📝 后续操作:"
echo "  git add ."
echo "  git commit -m 'your commit message'"
echo "  git push"
