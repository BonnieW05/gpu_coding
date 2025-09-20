"""
Kevin-32B 和 KernelBench 评估命令行接口
"""

import argparse
import sys
import time
import logging
from pathlib import Path

from .model import KevinModel
from .evaluator import KernelBenchEvaluator
from .utils import (
    load_config, setup_logging, load_test_cases, 
    save_generated_kernels, check_system_requirements
)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="Kevin-32B CUDA Kernel Generation and KernelBench Evaluation"
    )
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/config.yaml",
        help="配置文件路径"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="评估样本数量"
    )
    parser.add_argument(
        "--check-system",
        action="store_true",
        help="检查系统要求后退出"
    )
    
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 设置日志
    setup_logging(config)
    logger = logging.getLogger(__name__)
    
    # 检查系统要求
    if args.check_system:
        system_info = check_system_requirements()
        logger.info("系统要求检查:")
        logger.info(f"CUDA可用: {system_info['cuda_available']}")
        logger.info(f"GPU数量: {system_info['gpu_count']}")
        for gpu in system_info['gpu_info']:
            logger.info(f"GPU {gpu['device_id']}: {gpu['name']}, "
                       f"内存: {gpu['memory_total']:.1f}GB")
        logger.info(f"NVCC可用: {system_info['nvcc_available']}")
        return
    
    # 运行评估
    try:
        run_evaluation(config, args)
    except KeyboardInterrupt:
        logger.info("用户中断评估")
        sys.exit(1)
    except Exception as e:
        logger.error(f"评估过程中发生错误: {e}")
        sys.exit(1)


def run_evaluation(config: dict, args: argparse.Namespace):
    """运行完整的评估流程"""
    logger = logging.getLogger(__name__)
    
    # 覆盖配置参数
    if args.num_samples is not None:
        config["evaluation"]["num_samples"] = args.num_samples
    
    logger.info("开始Kevin-32B CUDA内核生成和KernelBench评估")
    logger.info(f"评估样本数量: {config['evaluation']['num_samples']}")
    
    # 检查系统要求
    system_info = check_system_requirements()
    if not system_info["cuda_available"]:
        logger.error("CUDA不可用，无法运行评估")
        return
    
    if not system_info["nvcc_available"]:
        logger.error("NVCC不可用，无法编译CUDA内核")
        return
    
    # 初始化模型
    logger.info("初始化Kevin-32B模型...")
    model = KevinModel(config)
    
    # 加载测试用例
    logger.info("加载测试用例...")
    test_cases = load_test_cases(config["kernelbench"]["dataset_path"])
    
    # 限制测试用例数量
    num_samples = config["evaluation"]["num_samples"]
    if num_samples < len(test_cases):
        test_cases = test_cases[:num_samples]
        logger.info(f"限制测试用例数量为: {num_samples}")
    
    # 生成内核代码
    logger.info("开始生成CUDA内核代码...")
    prompts = [test_case["prompt"] for test_case in test_cases]
    
    start_time = time.time()
    generated_kernels = model.generate_batch_kernels(prompts)
    generation_time = time.time() - start_time
    
    logger.info(f"内核代码生成完成，耗时: {generation_time:.2f}秒")
    
    # 保存生成的内核代码
    if config["evaluation"]["save_generated_kernels"]:
        output_dir = Path(config["kernelbench"]["results_path"]) / "generated_kernels"
        save_generated_kernels(generated_kernels, test_cases, str(output_dir))
        logger.info(f"生成的内核代码已保存到: {output_dir}")
    
    # 初始化评估器
    logger.info("初始化KernelBench评估器...")
    evaluator = KernelBenchEvaluator(config)
    
    # 运行评估
    if config["evaluation"]["compile_kernels"] or config["evaluation"]["run_benchmarks"]:
        logger.info("开始KernelBench评估...")
        
        start_time = time.time()
        results = evaluator.evaluate_batch(generated_kernels, test_cases)
        evaluation_time = time.time() - start_time
        
        logger.info(f"KernelBench评估完成，耗时: {evaluation_time:.2f}秒")
        
        # 生成报告
        logger.info("生成评估报告...")
        report = evaluator.generate_report(results)
        
        # 保存结果
        results_file = evaluator.save_results(results)
        logger.info(f"详细结果已保存到: {results_file}")
        
        # 打印摘要
        print_evaluation_summary(report)
    
    logger.info("评估完成")


def print_evaluation_summary(report: dict):
    """打印评估摘要"""
    summary = report["summary"]
    performance = report["performance"]
    
    print("\n" + "="*60)
    print("Kevin-32B CUDA内核生成和KernelBench评估摘要")
    print("="*60)
    print(f"总内核数量: {summary['total_kernels']}")
    print(f"编译成功率: {summary['compilation_success_rate']:.2%}")
    print(f"执行成功率: {summary['execution_success_rate']:.2%}")
    print(f"平均评估时间: {summary['avg_evaluation_time']:.2f}秒")
    print()
    print("性能指标:")
    print(f"  平均执行时间: {performance['avg_execution_time_ms']:.3f} ms")
    print(f"  平均吞吐量: {performance['avg_throughput_gflops']:.3f} GFLOPS")
    print("="*60)


if __name__ == "__main__":
    main()