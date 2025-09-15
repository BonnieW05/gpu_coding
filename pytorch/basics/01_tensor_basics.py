"""
PyTorch基础 - 张量操作
学习PyTorch的基本张量操作和属性
"""

import torch
import numpy as np

def tensor_creation():
    """张量创建示例"""
    print("=== 张量创建 ===")
    
    # 从列表创建
    tensor1 = torch.tensor([1, 2, 3, 4])
    print(f"从列表创建: {tensor1}")
    
    # 从numpy数组创建
    np_array = np.array([1, 2, 3, 4])
    tensor2 = torch.from_numpy(np_array)
    print(f"从numpy创建: {tensor2}")
    
    # 创建零张量
    zeros = torch.zeros(2, 3)
    print(f"零张量:\n{zeros}")
    
    # 创建单位张量
    ones = torch.ones(2, 3)
    print(f"单位张量:\n{ones}")
    
    # 创建随机张量
    random_tensor = torch.randn(2, 3)
    print(f"随机张量:\n{random_tensor}")

def tensor_operations():
    """张量操作示例"""
    print("\n=== 张量操作 ===")
    
    a = torch.tensor([1, 2, 3])
    b = torch.tensor([4, 5, 6])
    
    # 基本运算
    print(f"a + b = {a + b}")
    print(f"a * b = {a * b}")
    print(f"a @ b = {a @ b}")  # 点积
    
    # 矩阵乘法
    matrix_a = torch.randn(2, 3)
    matrix_b = torch.randn(3, 2)
    result = torch.mm(matrix_a, matrix_b)
    print(f"矩阵乘法结果形状: {result.shape}")

def tensor_properties():
    """张量属性示例"""
    print("\n=== 张量属性 ===")
    
    tensor = torch.randn(2, 3, 4)
    print(f"张量形状: {tensor.shape}")
    print(f"张量维度: {tensor.dim()}")
    print(f"张量大小: {tensor.size()}")
    print(f"张量数据类型: {tensor.dtype}")
    print(f"张量设备: {tensor.device}")
    print(f"张量是否连续: {tensor.is_contiguous()}")

if __name__ == "__main__":
    tensor_creation()
    tensor_operations()
    tensor_properties()
