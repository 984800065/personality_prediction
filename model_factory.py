"""
模型工厂
统一创建模型实例，供 train.py 和 test.py 复用
"""
import torch
import torch.nn as nn
from typing import Optional, Dict, Any
from loguru import logger


def create_model(
    base_model_name: str,
    num_labels: int = 5,
    use_multi_instance: bool = False,
    checkpoint_config: Optional[Dict[str, Any]] = None,
    config: Optional[Any] = None,
    device: Optional[torch.device] = None
) -> nn.Module:
    """
    根据配置创建模型实例
    
    Args:
        base_model_name: 基座模型名称
        num_labels: 输出标签数量（默认5）
        use_multi_instance: 是否使用聚合模式（多实例学习）
        checkpoint_config: checkpoint中的配置字典（可选）
        config: 当前配置对象（可选，用于获取默认值）
        device: 设备（可选，如果提供会将模型移动到该设备）
    
    Returns:
        model: 创建的模型实例
    """
    if use_multi_instance:
        # 聚合模式（多实例学习）
        try:
            from models.multi_instance_late_fusion_model import MultiInstanceLateFusionPersonalityPredictor
        except ImportError as e:
            logger.error(f"无法导入 Multi-Instance Late Fusion 模型: {e}")
            raise
        
        # 从checkpoint或config获取参数（优先使用checkpoint中的值）
        def get_param(key: str, default: Any) -> Any:
            if checkpoint_config and key in checkpoint_config:
                return checkpoint_config[key]
            if config and hasattr(config, key):
                return getattr(config, key)
            return default
        
        freeze_base = get_param('freeze_base', False)
        use_mlp_head = get_param('use_mlp_head', True)
        mlp_hidden_size = get_param('mlp_hidden_size', 256)
        local_files_only = config.local_files_only if config else False
        use_metadata = get_param('use_metadata', True)
        aggregation_method = get_param('aggregation_method', 'mean')
        aggregation_hidden_size = get_param('aggregation_hidden_size', 256)
        
        if checkpoint_config:
            logger.info("使用聚合模式（Multi-Instance Late Fusion，从checkpoint恢复）")
        else:
            logger.info("使用聚合模式（Multi-Instance Late Fusion）")
        
        model = MultiInstanceLateFusionPersonalityPredictor(
            base_model_name=base_model_name,
            num_labels=num_labels,
            freeze_base=freeze_base,
            use_improved_pooling=False,  # 聚合模式默认不使用改进池化
            use_mlp_head=use_mlp_head,
            mlp_hidden_size=mlp_hidden_size,
            local_files_only=local_files_only,
            use_metadata=use_metadata,
            use_gender=True,  # 默认全部使用
            use_education=True,
            use_race=True,
            use_age=True,
            use_income=True,
            aggregation_method=aggregation_method,
            aggregation_hidden_size=aggregation_hidden_size
        )
    else:
        # 原始模型模式
        try:
            from models.vanilla_model import PersonalityPredictor
        except ImportError as e:
            logger.error(f"无法导入 PersonalityPredictor 模型: {e}")
            raise
        
        # 从checkpoint或config获取参数（优先使用checkpoint中的值）
        def get_param(key: str, default: Any) -> Any:
            if checkpoint_config and key in checkpoint_config:
                return checkpoint_config[key]
            if config and hasattr(config, key):
                return getattr(config, key)
            return default
        
        freeze_base = get_param('freeze_base', False)
        use_mlp_head = get_param('use_mlp_head', True)
        mlp_hidden_size = get_param('mlp_hidden_size', 256)
        local_files_only = config.local_files_only if config else False
        use_metadata = get_param('use_metadata', True)
        
        if checkpoint_config:
            logger.info("使用原始模型模式（PersonalityPredictor with Late Fusion，从checkpoint恢复）")
        else:
            logger.info("使用原始模型模式（PersonalityPredictor with Late Fusion）")
        
        model = PersonalityPredictor(
            base_model_name=base_model_name,
            num_labels=num_labels,
            freeze_base=freeze_base,
            use_mlp_head=use_mlp_head,
            mlp_hidden_size=mlp_hidden_size,
            local_files_only=local_files_only,
            use_metadata=use_metadata
        )
    
    # 如果提供了device，将模型移动到该设备
    if device is not None:
        model.to(device)
    
    return model

