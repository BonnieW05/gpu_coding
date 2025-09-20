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
        """加载Kevin-32B模型和分词器"""
        logger.info("开始加载Kevin-32B模型...")
        
        try:
            # 加载分词器
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config["model"]["name"],
                cache_dir=self.config["model"]["cache_dir"],
                trust_remote_code=True
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # 加载模型
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config["model"]["name"],
                cache_dir=self.config["model"]["cache_dir"],
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            logger.info("Kevin-32B模型加载完成")
            
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise
    
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