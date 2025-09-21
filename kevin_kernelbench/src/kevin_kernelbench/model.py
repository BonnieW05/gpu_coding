"""
Kevin-32B 模型接口
"""

import torch
import logging
import time
from typing import Dict, Any, List
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

# 可选导入swanlab
try:
    import swanlab
    SWANLAB_AVAILABLE = True
except ImportError:
    SWANLAB_AVAILABLE = False
    swanlab = None


class KevinModel:
    """Kevin-32B 模型包装器，用于生成CUDA内核代码"""
    
    def __init__(self, config: Dict[str, Any]):
        """初始化Kevin模型"""
        self.config = config
        self.model = None
        self.tokenizer = None
        self.swanlab_run = None
        self._load_model()
        
        # 初始化SwanLab（如果可用）
        if SWANLAB_AVAILABLE and config.get("monitoring", {}).get("use_swanlab", False):
            try:
                self.swanlab_run = swanlab.init(
                    project="kevin-kernelbench",
                    experiment_name=f"model_generation_{int(time.time())}",
                    description="Kevin-32B CUDA kernel generation"
                )
                logger.info("SwanLab模型跟踪已初始化")
            except Exception as e:
                logger.warning(f"SwanLab模型跟踪初始化失败: {e}")
                self.swanlab_run = None
    
    def _load_model(self):
        """加载Kevin-32B模型和分词器，采用渐进式加载策略"""
        logger.info("开始加载Kevin-32B模型...")
        
        # 策略1: 尝试内存安全加载（从HuggingFace Hub）
        if self._try_load_from_hub():
            logger.info("✅ 成功从HuggingFace Hub加载模型（内存安全模式）")
            return
        
        # 策略2: 尝试本地加载（内存安全模式）
        if self._try_load_from_local():
            logger.info("✅ 成功从本地加载模型（内存安全模式）")
            return
        
        # 策略3: 放弃加载
        logger.error("❌ 所有加载策略都失败，无法加载模型")
        raise RuntimeError("无法加载Kevin-32B模型，请检查网络连接和本地文件")
    
    def _try_load_from_hub(self):
        """尝试从HuggingFace Hub加载模型（内存安全模式）"""
        try:
            logger.info("🔄 尝试从HuggingFace Hub加载模型...")
            
            # 内存安全的分词器加载
            tokenizer_kwargs = {
                "cache_dir": self.config["model"]["cache_dir"],
                "trust_remote_code": True,
                "local_files_only": False  # 允许从网络下载
            }
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config["model"]["name"],
                **tokenizer_kwargs
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # 内存安全的模型加载
            model_kwargs = {
                "cache_dir": self.config["model"]["cache_dir"],
                "torch_dtype": torch.float16,
                "device_map": "auto",
                "trust_remote_code": True,
                "low_cpu_mem_usage": True,
                "local_files_only": False  # 允许从网络下载
            }
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config["model"]["name"],
                **model_kwargs
            )
            
            return True
            
        except Exception as e:
            logger.warning(f"⚠️ 从HuggingFace Hub加载失败: {e}")
            self.model = None
            self.tokenizer = None
            return False
    
    def _try_load_from_local(self):
        """尝试从本地加载模型（内存安全模式）"""
        try:
            logger.info("🔄 尝试从本地加载模型...")
            
            # 检查本地模型路径
            local_model_path = self.config["model"].get("local_path")
            if not local_model_path:
                # 如果没有指定本地路径，尝试从cache_dir加载
                local_model_path = self.config["model"]["cache_dir"]
            
            # 内存安全的本地分词器加载
            tokenizer_kwargs = {
                "cache_dir": local_model_path,
                "trust_remote_code": True,
                "local_files_only": True  # 只使用本地文件
            }
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                local_model_path,
                **tokenizer_kwargs
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # 内存安全的本地模型加载
            model_kwargs = {
                "cache_dir": local_model_path,
                "torch_dtype": torch.float16,
                "device_map": "auto",
                "trust_remote_code": True,
                "low_cpu_mem_usage": True,
                "local_files_only": True  # 只使用本地文件
            }
            
            self.model = AutoModelForCausalLM.from_pretrained(
                local_model_path,
                **model_kwargs
            )
            
            return True
            
        except Exception as e:
            logger.warning(f"⚠️ 从本地加载失败: {e}")
            self.model = None
            self.tokenizer = None
            return False
    
    def generate_kernel(self, prompt: str) -> str:
        """生成单个CUDA内核"""
        start_time = time.time()
        
        inputs = self.tokenizer(prompt, return_tensors="pt")
        
        # 将输入移动到正确的设备
        if hasattr(self.model, 'device'):
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config["generation"]["max_new_tokens"],
                temperature=self.config["generation"]["temperature"],
                top_p=self.config["generation"]["top_p"],
                top_k=self.config["generation"]["top_k"],
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        result = generated_text[len(prompt):].strip()
        
        # 记录到SwanLab
        if self.swanlab_run:
            try:
                generation_time = time.time() - start_time
                self.swanlab_run.log({
                    "generation_time": generation_time,
                    "prompt_length": len(prompt),
                    "generated_length": len(result),
                    "tokens_per_second": len(result.split()) / generation_time if generation_time > 0 else 0
                })
            except Exception as e:
                logger.warning(f"SwanLab生成日志记录失败: {e}")
        
        return result
    
    def generate_batch_kernels(self, prompts: List[str]) -> List[str]:
        """批量生成CUDA内核"""
        results = []
        for prompt in prompts:
            try:
                kernel = self.generate_kernel(prompt)
                results.append(kernel)
            except Exception as e:
                logger.error(f"生成内核失败: {e}")
                results.append("")
        return results