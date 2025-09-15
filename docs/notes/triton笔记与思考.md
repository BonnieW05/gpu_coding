@triton.jit指示运行时（JIT）编译内核，灵活性更高。

```
普通 Python 程序调用 GPU（例如 PyTorch）
------------------------------------------------
Python 代码 (x.cuda()) 
        │
        ▼
PyTorch/CuPy 封装好的 C++/CUDA 内核
        │
        ▼
CUDA 驱动 (Driver API / Runtime API)
        │
        ▼
GPU 执行 (PTX → SASS → 硬件)
```
```
Triton @triton.jit 调用 GPU
------------------------------------------------
Python 代码 (调用 add_kernel[grid](...))
        │
        ▼
@triton.jit 拦截函数定义
        │
        ▼
第一次调用: Python AST → Triton IR → LLVM IR → PTX
编译完成并缓存
        │
        ▼
CUDA 驱动 (加载 PTX 内核)
        │
        ▼
GPU 执行 (PTX → SASS → 硬件)
```

