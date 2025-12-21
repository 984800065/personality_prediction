"""
基座模型加载器
统一加载base_model和获取hidden_size，方便更换基座模型
"""
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from typing import Tuple


def load_base_model(
    base_model_name: str,
    freeze_base: bool = False,
    local_files_only: bool = False
) -> Tuple[nn.Module, int]:
    """
    加载基座模型并获取hidden_size
    
    Args:
        base_model_name: 基座模型名称
        freeze_base: 是否冻结基座模型参数
        local_files_only: 是否只使用本地模型文件
    
    Returns:
        base_model: 加载的基座模型
        hidden_size: 模型的hidden_size
    """
    # 加载基座模型
    base_model = AutoModel.from_pretrained(
        base_model_name,
        trust_remote_code=True,
        local_files_only=local_files_only
    )
    
    # 获取配置以获取hidden_size
    config = AutoConfig.from_pretrained(
        base_model_name,
        trust_remote_code=True,
        local_files_only=local_files_only
    )
    hidden_size = config.hidden_size
    
    # 如果冻结基座模型
    if freeze_base:
        for param in base_model.parameters():
            param.requires_grad = False
    
    return base_model, hidden_size
