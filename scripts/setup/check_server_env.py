#!/usr/bin/env python3
"""
服务器环境检查脚本
基于服务器信息.md中的配置进行环境验证
"""

import os
import sys
import subprocess
import torch
import platform
import psutil
from pathlib import Path

def print_header(title):
    """打印标题"""
    print(f"\n{'='*50}")
    print(f" {title}")
    print(f"{'='*50}")

def check_system_info():
    """检查系统基本信息"""
    print_header("系统信息")
    
    print(f"操作系统: {platform.system()} {platform.release()}")
    print(f"架构: {platform.machine()}")
    print(f"内核版本: {platform.version()}")
    print(f"Python版本: {sys.version}")
    print(f"当前用户: {os.getenv('USER', 'unknown')}")
    print(f"用户ID: {os.getuid()}")
    print(f"工作目录: {os.getcwd()}")
    
    # 检查容器环境
    if os.path.exists('/.dockerenv'):
        print("环境: Docker容器")
    else:
        print("环境: 物理机/虚拟机")

def check_cpu_info():
    """检查CPU信息"""
    print_header("CPU信息")
    
    print(f"CPU核心数: {psutil.cpu_count(logical=False)}")
    print(f"逻辑处理器数: {psutil.cpu_count(logical=True)}")
    print(f"CPU使用率: {psutil.cpu_percent(interval=1):.1f}%")
    
    # 尝试获取CPU型号
    try:
        with open('/proc/cpuinfo', 'r') as f:
            for line in f:
                if 'model name' in line:
                    cpu_model = line.split(':')[1].strip()
                    print(f"CPU型号: {cpu_model}")
                    break
    except:
        print("CPU型号: 无法获取")

def check_memory_info():
    """检查内存信息"""
    print_header("内存信息")
    
    memory = psutil.virtual_memory()
    print(f"总内存: {memory.total / (1024**3):.1f} GB")
    print(f"可用内存: {memory.available / (1024**3):.1f} GB")
    print(f"已使用内存: {memory.used / (1024**3):.1f} GB")
    print(f"内存使用率: {memory.percent:.1f}%")
    
    # 预期配置检查
    expected_memory = 755  # GB
    actual_memory = memory.total / (1024**3)
    if abs(actual_memory - expected_memory) < 10:
        print("✅ 内存配置符合预期 (755GB)")
    else:
        print(f"⚠️  内存配置与预期不符 (预期: 755GB, 实际: {actual_memory:.1f}GB)")

def check_gpu_info():
    """检查GPU信息"""
    print_header("GPU信息")
    
    try:
        # 使用nvidia-smi获取GPU信息
        result = subprocess.run(['nvidia-smi', '--query-gpu=index,name,memory.total,driver_version', 
                               '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            gpu_lines = result.stdout.strip().split('\n')
            print(f"检测到 {len(gpu_lines)} 个GPU:")
            
            for i, line in enumerate(gpu_lines):
                parts = line.split(', ')
                if len(parts) >= 4:
                    gpu_id, name, memory, driver = parts[0], parts[1], parts[2], parts[3]
                    print(f"  GPU {gpu_id}: {name}")
                    print(f"    显存: {memory} MB")
                    print(f"    驱动版本: {driver}")
            
            # 检查是否符合预期配置
            if len(gpu_lines) == 2 and 'RTX 4090' in result.stdout:
                print("✅ GPU配置符合预期 (双RTX 4090)")
            else:
                print("⚠️  GPU配置与预期不符 (预期: 双RTX 4090)")
        else:
            print("❌ nvidia-smi执行失败")
            
    except FileNotFoundError:
        print("❌ nvidia-smi未找到，请检查NVIDIA驱动安装")
    except Exception as e:
        print(f"❌ GPU检查失败: {e}")

def check_cuda_info():
    """检查CUDA信息"""
    print_header("CUDA信息")
    
    try:
        # 检查nvcc
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            version_line = [line for line in result.stdout.split('\n') if 'release' in line][0]
            cuda_version = version_line.split('release')[1].split(',')[0].strip()
            print(f"CUDA版本: {cuda_version}")
            
            # 检查是否符合预期
            if '12.4' in cuda_version:
                print("✅ CUDA版本符合预期 (12.4)")
            else:
                print(f"⚠️  CUDA版本与预期不符 (预期: 12.4, 实际: {cuda_version})")
        else:
            print("❌ nvcc未找到或执行失败")
    except FileNotFoundError:
        print("❌ nvcc未找到，请检查CUDA安装")
    except Exception as e:
        print(f"❌ CUDA检查失败: {e}")

def check_pytorch_info():
    """检查PyTorch信息"""
    print_header("PyTorch信息")
    
    try:
        print(f"PyTorch版本: {torch.__version__}")
        print(f"CUDA可用: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"PyTorch CUDA版本: {torch.version.cuda}")
            print(f"GPU数量: {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"  GPU {i}: {props.name}")
                print(f"    显存: {props.total_memory / (1024**3):.1f} GB")
                print(f"    计算能力: {props.major}.{props.minor}")
            
            # 简单GPU测试
            try:
                x = torch.randn(100, 100, device='cuda')
                y = torch.mm(x, x)
                print("✅ GPU计算测试通过")
            except Exception as e:
                print(f"❌ GPU计算测试失败: {e}")
        else:
            print("❌ CUDA不可用")
            
    except ImportError:
        print("❌ PyTorch未安装")
    except Exception as e:
        print(f"❌ PyTorch检查失败: {e}")

def check_triton_info():
    """检查Triton信息"""
    print_header("Triton信息")
    
    try:
        import triton
        print(f"Triton版本: {triton.__version__}")
        
        # 检查Triton后端
        try:
            import triton.language as tl
            print("✅ Triton语言模块可用")
        except ImportError as e:
            print(f"❌ Triton语言模块不可用: {e}")
        
        # 检查Triton编译
        try:
            import triton.compiler as tc
            print("✅ Triton编译器可用")
        except ImportError as e:
            print(f"❌ Triton编译器不可用: {e}")
            
        # 简单Triton测试
        try:
            import triton.language as tl
            import triton.testing as tt
            
            @triton.jit
            def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
                pid = tl.program_id(axis=0)
                block_start = pid * BLOCK_SIZE
                offsets = block_start + tl.arange(0, BLOCK_SIZE)
                mask = offsets < n_elements
                x = tl.load(x_ptr + offsets, mask=mask)
                y = tl.load(y_ptr + offsets, mask=mask)
                output = x + y
                tl.store(output_ptr + offsets, output, mask=mask)
            
            print("✅ Triton内核编译测试通过")
        except Exception as e:
            print(f"❌ Triton内核测试失败: {e}")
            
    except ImportError:
        print("❌ Triton未安装")
        print("安装命令: pip install triton")
    except Exception as e:
        print(f"❌ Triton检查失败: {e}")

def check_thunderkitten_info():
    """检查ThunderKitten信息"""
    print_header("ThunderKitten信息")
    
    try:
        import thunderkitten
        print(f"ThunderKitten版本: {thunderkitten.__version__}")
        
        # 检查ThunderKitten模块
        try:
            from thunderkitten import ops
            print("✅ ThunderKitten操作模块可用")
        except ImportError as e:
            print(f"❌ ThunderKitten操作模块不可用: {e}")
        
        # 检查ThunderKitten后端
        try:
            from thunderkitten import backend
            print("✅ ThunderKitten后端可用")
        except ImportError as e:
            print(f"❌ ThunderKitten后端不可用: {e}")
            
        # 简单ThunderKitten测试
        try:
            import torch
            from thunderkitten import ops
            
            # 创建测试张量
            x = torch.randn(100, 100, device='cuda')
            y = torch.randn(100, 100, device='cuda')
            
            # 测试ThunderKitten操作
            result = ops.add(x, y)
            print("✅ ThunderKitten操作测试通过")
        except Exception as e:
            print(f"❌ ThunderKitten操作测试失败: {e}")
            
    except ImportError:
        print("❌ ThunderKitten未安装")
        print("安装命令: pip install thunderkitten")
    except Exception as e:
        print(f"❌ ThunderKitten检查失败: {e}")

def check_disk_info():
    """检查磁盘信息"""
    print_header("磁盘信息")
    
    try:
        # 检查根分区
        root_usage = psutil.disk_usage('/')
        print(f"根分区 (/):")
        print(f"  总容量: {root_usage.total / (1024**3):.1f} GB")
        print(f"  已使用: {root_usage.used / (1024**3):.1f} GB")
        print(f"  可用空间: {root_usage.free / (1024**3):.1f} GB")
        print(f"  使用率: {(root_usage.used / root_usage.total) * 100:.1f}%")
        
        # 检查用户目录
        user_home = os.path.expanduser('~')
        if os.path.exists(user_home):
            home_usage = psutil.disk_usage(user_home)
            print(f"用户目录 ({user_home}):")
            print(f"  总容量: {home_usage.total / (1024**3):.1f} GB")
            print(f"  已使用: {home_usage.used / (1024**3):.1f} GB")
            print(f"  可用空间: {home_usage.free / (1024**3):.1f} GB")
            print(f"  使用率: {(home_usage.used / home_usage.total) * 100:.1f}%")
        
        # 检查数据分区
        data_paths = ['/root/autodl-tmp', '/data', '/mnt/data']
        for path in data_paths:
            if os.path.exists(path):
                try:
                    data_usage = psutil.disk_usage(path)
                    print(f"数据分区 ({path}):")
                    print(f"  总容量: {data_usage.total / (1024**3):.1f} GB")
                    print(f"  已使用: {data_usage.used / (1024**3):.1f} GB")
                    print(f"  可用空间: {data_usage.free / (1024**3):.1f} GB")
                    print(f"  使用率: {(data_usage.used / data_usage.total) * 100:.1f}%")
                except PermissionError:
                    print(f"数据分区 ({path}): 权限不足，无法访问")
                    
    except Exception as e:
        print(f"❌ 磁盘检查失败: {e}")

def check_network_info():
    """检查网络信息"""
    print_header("网络信息")
    
    try:
        # 获取网络接口信息
        net_io = psutil.net_io_counters()
        print(f"网络统计:")
        print(f"  发送字节: {net_io.bytes_sent / (1024**2):.1f} MB")
        print(f"  接收字节: {net_io.bytes_recv / (1024**2):.1f} MB")
        
        # 获取网络接口
        net_ifaces = psutil.net_if_addrs()
        print(f"网络接口:")
        for interface, addresses in net_ifaces.items():
            for addr in addresses:
                if addr.family == 2:  # IPv4
                    print(f"  {interface}: {addr.address}")
                    
    except Exception as e:
        print(f"❌ 网络检查失败: {e}")

def check_environment_variables():
    """检查环境变量"""
    print_header("环境变量")
    
    important_vars = [
        'CUDA_HOME', 'PATH', 'LD_LIBRARY_PATH', 
        'PYTORCH_CUDA_VERSION', 'PYTORCH_INDEX_URL',
        'HOME', 'USER', 'SHELL'
    ]
    
    for var in important_vars:
        value = os.getenv(var, '未设置')
        if var in ['PATH', 'LD_LIBRARY_PATH'] and len(value) > 100:
            value = value[:100] + "..."
        print(f"{var}: {value}")

def check_directories():
    """检查重要目录"""
    print_header("目录检查")
    
    important_dirs = [
        os.path.expanduser('~'),
        '/root/autodl-tmp/wanghan',
        '/usr/local/cuda-12.4',
        '/usr/local/cuda',
    ]
    
    for dir_path in important_dirs:
        if os.path.exists(dir_path):
            try:
                stat = os.stat(dir_path)
                print(f"✅ {dir_path} (存在)")
            except PermissionError:
                print(f"⚠️  {dir_path} (存在，但权限不足)")
        else:
            print(f"❌ {dir_path} (不存在)")

def main():
    """主函数"""
    print("服务器环境检查脚本")
    print("基于服务器信息.md中的配置")
    print(f"检查时间: {subprocess.run(['date'], capture_output=True, text=True).stdout.strip()}")
    
    check_system_info()
    check_cpu_info()
    check_memory_info()
    check_gpu_info()
    check_cuda_info()
    check_pytorch_info()
    check_triton_info()
    check_thunderkitten_info()
    check_disk_info()
    check_network_info()
    check_environment_variables()
    check_directories()
    
    print_header("检查完成")
    print("如果发现配置与预期不符，请检查:")
    print("1. NVIDIA驱动是否正确安装")
    print("2. CUDA版本是否匹配")
    print("3. PyTorch是否正确安装")
    print("4. Triton是否正确安装 (pip install triton)")
    print("5. ThunderKitten是否正确安装 (pip install thunderkitten)")
    print("6. 环境变量是否正确设置")
    print("7. 目录权限是否正确")

if __name__ == "__main__":
    main()
