# Kevin-32B CUDA内核生成和KernelBench评估框架

这个项目提供了使用Kevin-32B模型生成CUDA内核代码，并使用KernelBench进行性能评估的完整框架。

## 🖥️ 系统要求

### 硬件要求
- **GPU**: NVIDIA GPU，推荐RTX 4090或更高（24GB+显存）
- **内存**: 至少32GB系统内存，推荐64GB+
- **存储**: 至少100GB可用空间

### 软件要求
- **操作系统**: Ubuntu 20.04/22.04, CentOS 7/8, 或 Windows 10/11
- **Python**: 3.8或更高版本
- **CUDA**: 12.4或更高版本
- **NVIDIA驱动**: 535或更高版本

## 🚀 快速开始

### 1. 克隆项目
```bash
git clone <repository-url>
cd kevin_kernelbench
```

### 2. 安装依赖
```bash
# 使用安装脚本（推荐）
chmod +x scripts/install.sh
./scripts/install.sh

# 或手动安装
pip install -r requirements.txt
```

### 3. 检查系统环境
```bash
python3 run_evaluation.py --check-system
```

### 4. 运行评估
```bash
python3 run_evaluation.py
```

## 📁 项目结构

```
kevin_kernelbench/
├── src/kevin_kernelbench/          # 主要源代码
│   ├── __init__.py                 # 包初始化
│   ├── model.py                    # Kevin-32B模型接口
│   ├── evaluator.py                # KernelBench评估器
│   ├── utils.py                    # 工具函数
│   └── cli.py                      # 命令行接口
├── configs/                        # 配置文件
│   └── config.yaml                 # 主配置文件
├── scripts/                        # 脚本文件
│   └── install.sh                  # 安装脚本
├── data/                           # 数据目录
│   ├── models/                     # 模型缓存
│   └── kernelbench/                # KernelBench数据集
├── results/                        # 结果输出
├── logs/                           # 日志文件
├── requirements.txt                # Python依赖
├── run_evaluation.py               # 主运行脚本
└── README.md                       # 项目说明
```

## ⚙️ 配置说明

主要配置文件是`configs/config.yaml`，包含以下配置项：

### 模型配置
```yaml
model:
  name: "cognition-ai/Kevin-32B"    # 模型名称
  cache_dir: "./data/models"        # 模型缓存目录
  torch_dtype: "float16"            # 数据类型
  device_map: "auto"                # 设备映射
```

### 生成配置
```yaml
generation:
  max_new_tokens: 1024              # 最大生成token数
  temperature: 0.1                  # 温度参数
  top_p: 0.9                        # top-p采样
  top_k: 50                         # top-k采样
```

### 评估配置
```yaml
evaluation:
  num_samples: 3                    # 评估样本数量
  save_generated_kernels: true      # 保存生成的内核
  compile_kernels: true             # 编译内核
  run_benchmarks: true              # 运行基准测试
```

## 📊 预期结果

### 内存使用情况
- **模型加载**: 约60-80GB GPU显存
- **推理过程**: 额外10-20GB GPU显存
- **系统内存**: 约20-30GB

### 运行时间估算
- **模型加载**: 5-10分钟
- **内核生成**: 每个内核10-30秒
- **编译评估**: 每个内核1-5分钟
- **总时间**: 3个样本约30-60分钟

### 输出结果
评估完成后会生成以下文件：

1. **详细结果**: `results/kevin_evaluation_results_YYYYMMDD_HHMMSS.json`
2. **生成的内核**: `results/generated_kernels/*.cu`
3. **日志文件**: `logs/kevin_evaluation.log`

## 🔧 高级用法

### 自定义测试用例
在`src/kevin_kernelbench/utils.py`中的`load_test_cases`函数中添加自定义测试用例：

```python
test_cases = [
    {
        "name": "custom_kernel",
        "description": "自定义内核描述",
        "input_size": 1024,
        "prompt": "Write a CUDA kernel for..."
    }
]
```

### 调整生成参数
修改`configs/config.yaml`中的生成参数：

```yaml
generation:
  temperature: 0.2      # 增加创造性
  max_new_tokens: 2048  # 生成更长的代码
```

### 批量评估
```bash
# 评估更多样本
python3 run_evaluation.py --num-samples 10
```

## 🐛 故障排除

### 常见问题

1. **CUDA内存不足**
   - 减少batch_size
   - 使用更小的模型精度
   - 清理GPU缓存

2. **编译失败**
   - 检查CUDA版本兼容性
   - 确保NVCC可用
   - 检查内核代码语法

3. **模型加载失败**
   - 检查网络连接
   - 确保有足够的磁盘空间
   - 验证Hugging Face访问权限

### 日志分析
查看详细日志：
```bash
tail -f logs/kevin_evaluation.log
```

## 📚 参考文献

- [Kevin-32B模型](https://huggingface.co/cognition-ai/Kevin-32B)
- [CognitionAI博客](https://cognition.ai/blog/kevin-32b)
- [KernelBench论文](https://github.com/KernelBench/KernelBench)

## 📄 许可证

MIT License

## 🤝 贡献

欢迎提交问题报告和改进建议！