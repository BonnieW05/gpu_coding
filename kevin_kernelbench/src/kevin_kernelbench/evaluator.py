"""
KernelBench 评估器
"""

import logging
import subprocess
import tempfile
import os
from pathlib import Path
from typing import List, Dict, Any
import json
import time

logger = logging.getLogger(__name__)

# 可选导入swanlab
try:
    import swanlab
    SWANLAB_AVAILABLE = True
except ImportError:
    SWANLAB_AVAILABLE = False
    swanlab = None


class KernelBenchEvaluator:
    """KernelBench 评估器，用于编译和测试CUDA内核"""
    
    def __init__(self, config: Dict[str, Any]):
        """初始化评估器"""
        self.config = config
        self.results_path = Path(config["kernelbench"]["results_path"])
        self.results_path.mkdir(parents=True, exist_ok=True)
        
        # 初始化SwanLab（如果可用）
        self.swanlab_run = None
        if SWANLAB_AVAILABLE and config.get("monitoring", {}).get("use_swanlab", False):
            try:
                self.swanlab_run = swanlab.init(
                    project="kevin-kernelbench",
                    experiment_name=f"evaluation_{int(time.time())}",
                    description="Kevin-32B CUDA kernel generation evaluation"
                )
                logger.info("SwanLab实验跟踪已初始化")
            except Exception as e:
                logger.warning(f"SwanLab初始化失败: {e}")
                self.swanlab_run = None
    
    def compile_kernel(self, kernel_code: str) -> Dict[str, Any]:
        """编译CUDA内核"""
        result = {
            "success": False,
            "errors": [],
            "executable_path": None
        }
        
        if not kernel_code.strip():
            result["errors"].append("空的内核代码")
            return result
        
        # 创建临时文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.cu', delete=False) as f:
            # 添加必要的头文件和main函数包装
            full_code = f"""
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

{kernel_code}

int main() {{
    printf("CUDA kernel compiled successfully\\n");
    return 0;
}}
"""
            f.write(full_code)
            cu_file = f.name
        
        exe_file = cu_file.replace('.cu', '.exe')
        
        try:
            # 编译命令
            cmd = [
                "nvcc", cu_file, "-o", exe_file,
                "-O3", "-std=c++17", "-lcudart"
            ]
            
            # 执行编译
            result_proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=60
            )
            
            if result_proc.returncode == 0:
                result["success"] = True
                result["executable_path"] = exe_file
            else:
                result["errors"].append(f"编译错误: {result_proc.stderr}")
        
        except subprocess.TimeoutExpired:
            result["errors"].append("编译超时")
        except Exception as e:
            result["errors"].append(f"编译异常: {str(e)}")
        finally:
            # 清理临时文件
            if os.path.exists(cu_file):
                os.unlink(cu_file)
        
        return result
    
    def run_benchmark(self, executable_path: str) -> Dict[str, Any]:
        """运行基准测试"""
        result = {
            "success": False,
            "execution_time_ms": None,
            "throughput_gflops": None,
            "errors": []
        }
        
        try:
            # 运行可执行文件
            start_time = time.time()
            proc = subprocess.run(
                [executable_path], 
                capture_output=True, 
                text=True, 
                timeout=30
            )
            end_time = time.time()
            
            if proc.returncode == 0:
                result["success"] = True
                result["execution_time_ms"] = (end_time - start_time) * 1000
                # 简化的吞吐量计算
                result["throughput_gflops"] = 1000.0 / result["execution_time_ms"]
            else:
                result["errors"].append(f"执行错误: {proc.stderr}")
        
        except subprocess.TimeoutExpired:
            result["errors"].append("执行超时")
        except Exception as e:
            result["errors"].append(f"执行异常: {str(e)}")
        finally:
            # 清理可执行文件
            if os.path.exists(executable_path):
                os.unlink(executable_path)
        
        return result
    
    def evaluate_kernel(self, kernel_code: str, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """评估单个内核"""
        start_time = time.time()
        
        result = {
            "test_case": test_case,
            "compilation_success": False,
            "execution_success": False,
            "performance_metrics": {},
            "errors": [],
            "evaluation_time": 0
        }
        
        # 编译内核
        compile_result = self.compile_kernel(kernel_code)
        result["compilation_success"] = compile_result["success"]
        result["errors"].extend(compile_result["errors"])
        
        # 如果编译成功，运行基准测试
        if compile_result["success"]:
            benchmark_result = self.run_benchmark(compile_result["executable_path"])
            result["execution_success"] = benchmark_result["success"]
            result["performance_metrics"] = {
                "execution_time_ms": benchmark_result["execution_time_ms"],
                "throughput_gflops": benchmark_result["throughput_gflops"]
            }
            result["errors"].extend(benchmark_result["errors"])
        
        result["evaluation_time"] = time.time() - start_time
        
        # 记录到SwanLab
        if self.swanlab_run:
            try:
                self.swanlab_run.log({
                    "test_case": test_case["name"],
                    "compilation_success": result["compilation_success"],
                    "execution_success": result["execution_success"],
                    "evaluation_time": result["evaluation_time"],
                    **result["performance_metrics"]
                })
            except Exception as e:
                logger.warning(f"SwanLab日志记录失败: {e}")
        
        return result
    
    def evaluate_batch(self, kernels: List[str], test_cases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """批量评估内核"""
        results = []
        for kernel, test_case in zip(kernels, test_cases):
            result = self.evaluate_kernel(kernel, test_case)
            results.append(result)
        return results
    
    def generate_report(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """生成评估报告"""
        total_kernels = len(results)
        compilation_success = sum(1 for r in results if r["compilation_success"])
        execution_success = sum(1 for r in results if r["execution_success"])
        
        # 计算性能指标
        execution_times = [r["performance_metrics"].get("execution_time_ms", 0) 
                          for r in results if r["execution_success"]]
        throughputs = [r["performance_metrics"].get("throughput_gflops", 0) 
                      for r in results if r["execution_success"]]
        
        report = {
            "summary": {
                "total_kernels": total_kernels,
                "compilation_success_rate": compilation_success / total_kernels if total_kernels > 0 else 0,
                "execution_success_rate": execution_success / total_kernels if total_kernels > 0 else 0,
                "avg_evaluation_time": sum(r["evaluation_time"] for r in results) / total_kernels if total_kernels > 0 else 0
            },
            "performance": {
                "avg_execution_time_ms": sum(execution_times) / len(execution_times) if execution_times else 0,
                "avg_throughput_gflops": sum(throughputs) / len(throughputs) if throughputs else 0,
                "min_execution_time_ms": min(execution_times) if execution_times else 0,
                "max_execution_time_ms": max(execution_times) if execution_times else 0,
                "min_throughput_gflops": min(throughputs) if throughputs else 0,
                "max_throughput_gflops": max(throughputs) if throughputs else 0
            },
            "detailed_results": results
        }
        
        # 记录汇总报告到SwanLab
        if self.swanlab_run:
            try:
                self.swanlab_run.log({
                    "summary/compilation_success_rate": report["summary"]["compilation_success_rate"],
                    "summary/execution_success_rate": report["summary"]["execution_success_rate"],
                    "summary/avg_evaluation_time": report["summary"]["avg_evaluation_time"],
                    "performance/avg_execution_time_ms": report["performance"]["avg_execution_time_ms"],
                    "performance/avg_throughput_gflops": report["performance"]["avg_throughput_gflops"],
                    "performance/min_execution_time_ms": report["performance"]["min_execution_time_ms"],
                    "performance/max_execution_time_ms": report["performance"]["max_execution_time_ms"]
                })
            except Exception as e:
                logger.warning(f"SwanLab汇总日志记录失败: {e}")
        
        return report
    
    def save_results(self, results: List[Dict[str, Any]]) -> str:
        """保存评估结果"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"kevin_evaluation_results_{timestamp}.json"
        filepath = self.results_path / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        return str(filepath)