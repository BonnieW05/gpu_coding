"""
Triton基础 - 向量加法
学习使用Triton编写GPU内核
"""

import torch
import triton
import triton.language as tl

@triton.jit
def vector_add_kernel(
    x_ptr,  # 输入向量x的指针
    y_ptr,  # 输入向量y的指针
    output_ptr,  # 输出向量的指针
    n_elements,  # 向量长度
    BLOCK_SIZE: tl.constexpr,  # 块大小
):
    """
    Triton向量加法内核
    """
    # 获取程序ID
    pid = tl.program_id(axis=0)
    
    # 计算这个程序块要处理的元素范围
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # 创建掩码，防止越界访问
    mask = offsets < n_elements
    
    # 从全局内存加载数据
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    
    # 执行向量加法
    output = x + y
    
    # 将结果写回全局内存
    tl.store(output_ptr + offsets, output, mask=mask)

def vector_add(x: torch.Tensor, y: torch.Tensor):
    """
    向量加法函数
    """
    # 确保输入张量在GPU上
    x = x.cuda()
    y = y.cuda()
    
    # 创建输出张量
    output = torch.empty_like(x)
    
    # 设置块大小
    BLOCK_SIZE = 1024
    
    # 计算网格大小
    grid = (triton.cdiv(x.numel(), BLOCK_SIZE),)
    
    # 启动内核
    vector_add_kernel[grid](
        x, y, output, x.numel(), BLOCK_SIZE
    )
    
    return output

def main():
    """主函数"""
    print("=== Triton向量加法示例 ===")
    
    # 创建测试数据
    size = 10000
    x = torch.randn(size, device='cuda')
    y = torch.randn(size, device='cuda')
    
    # 使用Triton计算
    triton_result = vector_add(x, y)
    
    # 使用PyTorch计算作为参考
    pytorch_result = x + y
    
    # 验证结果
    print(f"Triton结果形状: {triton_result.shape}")
    print(f"PyTorch结果形状: {pytorch_result.shape}")
    print(f"结果是否相等: {torch.allclose(triton_result, pytorch_result)}")
    
    # 计算误差
    error = torch.abs(triton_result - pytorch_result).max()
    print(f"最大误差: {error}")

if __name__ == "__main__":
    main()
