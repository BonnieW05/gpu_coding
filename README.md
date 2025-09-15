# GPU编程学习项目

这个仓库用于学习和实验PyTorch、Triton、ThunderKitten和CUDA编程。

## 📁 项目结构

```
gpu_coding/
├── pytorch/                    # PyTorch学习
│   ├── basics/                # 基础概念和操作
│   ├── advanced/              # 高级特性和优化
│   ├── examples/              # 示例代码
│   └── projects/              # 完整项目
├── triton/                    # Triton学习
│   ├── basics/                # 基础GPU内核编程
│   ├── advanced/              # 高级优化技巧
│   ├── examples/              # 示例内核
│   └── projects/              # 完整项目
├── thunderkitten/             # ThunderKitten学习
│   ├── basics/                # 基础概念
│   ├── advanced/              # 高级特性
│   ├── examples/              # 示例代码
│   └── projects/              # 完整项目
├── cuda/                      # CUDA学习
│   ├── basics/                # CUDA基础编程
│   ├── advanced/              # 高级CUDA优化
│   ├── examples/              # 示例程序
│   └── projects/              # 完整项目
├── shared/                    # 共享资源
│   ├── utils/                 # 通用工具函数
│   ├── data/                  # 数据集和样本数据
│   └── models/                # 共享模型定义
├── docs/                      # 文档
│   ├── notes/                 # 学习笔记
│   ├── tutorials/             # 教程文档
│   └── references/            # 参考资料
├── experiments/               # 实验
│   ├── benchmarks/            # 性能基准测试
│   └── comparisons/           # 不同框架对比
├── scripts/                   # 脚本
│   ├── setup/                 # 环境设置脚本
│   └── helpers/               # 辅助脚本
└── README.md                  # 项目说明
```

## 🚀 快速开始

### 环境要求
- Python 3.8+
- CUDA 11.0+
- PyTorch
- Triton
- ThunderKitten

### 安装依赖
```bash
# 安装基础依赖
pip install torch torchvision torchaudio

# 安装Triton
pip install triton

# 安装ThunderKitten
pip install thunderkitten

# 安装其他依赖
pip install -r requirements.txt
```

## 📚 学习路径

### 1. PyTorch
- **基础**: 张量操作、自动微分、神经网络基础
- **进阶**: 模型优化、分布式训练、自定义操作

### 2. CUDA
- **基础**: CUDA编程模型、内存管理、线程组织
- **进阶**: 性能优化、高级内存模式、多GPU编程

### 3. Triton
- **基础**: GPU内核编写、内存访问模式
- **进阶**: 高级优化、复杂算法实现

### 4. ThunderKitten
- **基础**: 框架特性、基本用法
- **进阶**: 高级功能、性能优化

## 🔬 实验和基准测试

在`experiments/`目录下进行各种性能测试和框架对比实验。

## 📝 学习笔记

在`docs/notes/`目录下记录学习过程中的重要概念和心得。

## 🤝 贡献

欢迎提交学习笔记、示例代码和改进建议！

## 📄 许可证

MIT License
