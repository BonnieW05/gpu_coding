# SwanLab 使用说明

## 安装和登录
```bash
# 安装
pip install swanlab>=0.3.0

# 登录
swanlab login
```

## 启用跟踪
在 `configs/config.yaml` 中设置：
```yaml
monitoring:
  use_swanlab: true
```

## 运行
```bash
kevin-eval
```

## 查看结果
访问 [SwanLab平台](https://swanlab.cn) 查看实验数据

## 跟踪的指标
- 生成时间、速度
- 编译成功率
- 执行性能
- 汇总统计