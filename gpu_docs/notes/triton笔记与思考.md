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

## 典例：Matrix multiplication: tiled implementation

运算示意图
https://www.youtube.com/watch?v=aMvCEEBIBto&skip_registered_account_check=true
![[matrix multiply tile implementation.png]]
通过A与B转制的块乘来实现，提高缓存利用率

python版本：
```python
# Do in parallel
for m in range(0, M, BLOCK_SIZE_M):
  # Do in parallel
  for n in range(0, N, BLOCK_SIZE_N):
    acc = zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=float32)
    for k in range(0, K, BLOCK_SIZE_K):
      a = A[m : m+BLOCK_SIZE_M, k : k+BLOCK_SIZE_K]
      b = B[k : k+BLOCK_SIZE_K, n : n+BLOCK_SIZE_N]
      acc += dot(a, b)
    C[m : m+BLOCK_SIZE_M, n : n+BLOCK_SIZE_N] = acc
```

伪代码：
```
&A[m : m+BLOCK_SIZE_M, k:k+BLOCK_SIZE_K] = 
a_ptr + (m : m+BLOCK_SIZE_M)[:, None]*A.stride(0)
 + (k : k+BLOCK_SIZE_K)[None, :] * A.stride(1);

&B[k : k+BLOCK_SIZE_K, n:n+BLOCK_SIZE_N] = 
b_ptr + (k : k+BLOCK_SIZE_K)[:, None] * B.stride(0) + 
(n : n+BLOCK_SIZE_N)[None, :] * B.stride(1);
```
`(...)[:, None]` 将其变为一个列向量——增加了一个纬度（行、列）
`A.stride(0)` 是矩阵一行的大小（以字节计），`A.stride(1)` 是一个元素的大小
(m : m+BLOCK_SIZE_M)[:, None]是列偏移，(k : k+BLOCK_SIZE_K)[None, :]是行偏移，加在一起就得到了矩阵的位置偏移！（也是广播的作用所在）


triton代码：
```python
offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M  
offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N  
offs_k = tl.arange(0, BLOCK_SIZE_K)  
a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k [None, :] * stride_ak)  
b_ptrs = b_ptr + (offs_k [:, None] * stride_bk + offs_bn[None, :] * stride_bn)
```

**`... + tl.arange(...)`**：是 Triton 的一个关键特性，叫做**广播（Broadcasting）**。我们将一个标量（`pid_m * BLOCK_SIZE_M`）加到一个向量（`tl.arange(...)`）上
**`... % M`**: 这是取模运算，避免越界访问。但这里存在一个问题：这会导致溢出值发生回绕。因此，进行掩码处理至关重要！


对于行读取顺序的优化：
```python
# Program ID  
pid = tl.program_id(axis=0)  
# Number of program ids along the M axis  
num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)  
# Number of programs ids along the N axis  
num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)  
# Number of programs in group  
num_pid_in_group = GROUP_SIZE_M * num_pid_n  
# Id of the group this program is in  
group_id = pid // num_pid_in_group  
# Row-id of the first program in the group  
first_pid_m = group_id * GROUP_SIZE_M  
# If `num_pid_m` isn't divisible by `GROUP_SIZE_M`, the last group is smaller  
group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)  
# *Within groups*, programs are ordered in a column-major order  
# Row-id of the program in the *launch grid*  
pid_m = first_pid_m + (pid % group_size_m)  
# Col-id of the program in the *launch grid*  
pid_n = (pid % num_pid_in_group) // group_size_m
```

GPU 内核（Kernel）启动时，会创建大量并行的“程序实例”，每个实例都有一个从0开始的唯一的一维ID. 我们的目标是把这个一维的 `pid` 聪明地映射成一个二维的 `(pid_m, pid_n)` 坐标，这个坐标决定了该程序实例负责计算输出矩阵 C 的哪一个块。

`tl.cdiv(a, b)` 是向上取整的除法 (`ceiling(a/b)`)

todo: 感觉还是有点费解。等我清醒一点再看一次（















