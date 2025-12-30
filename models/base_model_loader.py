"""
基座模型加载器
统一加载base_model和获取hidden_size，方便更换基座模型
支持从 BertForSequenceClassification 模型中提取 BERT encoder
"""
import torch.nn as nn
from transformers import AutoModel, AutoConfig, BertForSequenceClassification
from typing import Tuple
from loguru import logger


def load_base_model(
    base_model_name: str,
    freeze_base: bool = False,
    local_files_only: bool = False
) -> Tuple[nn.Module, int]:
    """
    加载基座模型并获取hidden_size
    
    支持以下模型类型：
    - 标准模型（如 roberta-base, bert-base-uncased）：直接加载
    - 带分类头的模型（如 Minej/bert-base-personality）：提取 encoder 部分
    
    Args:
        base_model_name: 基座模型名称
        freeze_base: 是否冻结基座模型参数
        local_files_only: 是否只使用本地模型文件
    
    Returns:
        base_model: 加载的基座模型（只包含 encoder，不包含分类头）
        hidden_size: 模型的hidden_size
    """
    # 特殊处理：如果是指定的人格预测模型，需要从分类模型中提取 BERT encoder
    if base_model_name == "Minej/bert-base-personality":
        logger.info(f"检测到人格预测模型 {base_model_name}，将从 BertForSequenceClassification 中提取 BERT encoder")
        
        # 加载带分类头的模型
        classification_model = BertForSequenceClassification.from_pretrained(
            base_model_name,
            trust_remote_code=True,
            local_files_only=local_files_only
        )
        
        # 提取 BERT encoder（这是实际的 base model）
        # BertForSequenceClassification 的结构是: bert (BertModel) + classifier (Linear)
        base_model = classification_model.bert
        
        # 获取配置以获取hidden_size
        config = AutoConfig.from_pretrained(
            base_model_name,
            trust_remote_code=True,
            local_files_only=local_files_only
        )
        hidden_size = config.hidden_size
        
        logger.info(f"已从 {base_model_name} 中提取 BERT encoder，hidden_size={hidden_size}")
    else:
        # 标准模型加载方式
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
