"""
标签归一化和反归一化工具
用于将训练标签从原始范围归一化到[0, 1]，并在测试时反归一化
"""
import numpy as np
import json
from pathlib import Path
from typing import Dict, Optional
from loguru import logger


class LabelNormalizer:
    """标签归一化器"""
    
    def __init__(self, label_mins: Optional[Dict[str, float]] = None,
                 label_maxs: Optional[Dict[str, float]] = None):
        """
        初始化归一化器
        
        Args:
            label_mins: 每个标签维度的最小值字典
            label_maxs: 每个标签维度的最大值字典
        """
        self.label_mins = label_mins or {}
        self.label_maxs = label_maxs or {}
        
        # 标签维度顺序（与训练数据一致）
        self.label_columns = [
            'personality_conscientiousness',
            'personality_openess',
            'personality_extraversion',
            'personality_agreeableness',
            'personality_stability'
        ]
    
    def fit(self, labels: np.ndarray) -> 'LabelNormalizer':
        """
        从训练数据中计算每个维度的min和max
        
        Args:
            labels: 标签数组，shape为 [n_samples, n_dimensions]
        
        Returns:
            self
        """
        if labels.ndim != 2:
            raise ValueError(f"labels应该是2维数组，shape为[n_samples, n_dimensions]，但得到shape: {labels.shape}")
        
        if labels.shape[1] != len(self.label_columns):
            raise ValueError(f"标签维度数不匹配：期望{len(self.label_columns)}，得到{labels.shape[1]}")
        
        # 计算每个维度的min和max
        for i, col_name in enumerate(self.label_columns):
            col_data = labels[:, i]
            # 排除NaN值
            valid_data = col_data[~np.isnan(col_data)]
            if len(valid_data) > 0:
                self.label_mins[col_name] = float(np.min(valid_data))
                self.label_maxs[col_name] = float(np.max(valid_data))
                logger.info(f"{col_name}: 范围 [{self.label_mins[col_name]:.4f}, {self.label_maxs[col_name]:.4f}]")
            else:
                logger.warning(f"{col_name}: 没有有效数据，无法计算min/max")
                self.label_mins[col_name] = 0.0
                self.label_maxs[col_name] = 1.0
        
        return self
    
    def normalize(self, labels: np.ndarray) -> np.ndarray:
        """
        将标签从原始范围归一化到[0, 1]
        
        Args:
            labels: 标签数组，shape为 [n_samples, n_dimensions] 或 [n_dimensions]
        
        Returns:
            归一化后的标签数组
        """
        labels = np.asarray(labels, dtype=np.float32)
        original_shape = labels.shape
        
        # 如果是1维，转换为2维
        if labels.ndim == 1:
            labels = labels.reshape(1, -1)
        
        if labels.shape[1] != len(self.label_columns):
            raise ValueError(f"标签维度数不匹配：期望{len(self.label_columns)}，得到{labels.shape[1]}")
        
        normalized = np.zeros_like(labels)
        
        for i, col_name in enumerate(self.label_columns):
            col_data = labels[:, i]
            min_val = self.label_mins.get(col_name, 0.0)
            max_val = self.label_maxs.get(col_name, 1.0)
            
            # Min-max归一化: (x - min) / (max - min)
            if max_val > min_val:
                normalized[:, i] = (col_data - min_val) / (max_val - min_val)
            else:
                # 如果min == max，保持原值或设为0.5
                normalized[:, i] = 0.5
        
        # 限制在[0, 1]范围内
        normalized = np.clip(normalized, 0.0, 1.0)
        
        # 恢复原始shape
        if len(original_shape) == 1:
            normalized = normalized.reshape(-1)
        
        return normalized
    
    def denormalize(self, labels: np.ndarray) -> np.ndarray:
        """
        将标签从[0, 1]反归一化回原始范围
        
        Args:
            labels: 归一化后的标签数组，shape为 [n_samples, n_dimensions] 或 [n_dimensions]
        
        Returns:
            反归一化后的标签数组
        """
        labels = np.asarray(labels, dtype=np.float32)
        original_shape = labels.shape
        
        # 如果是1维，转换为2维
        if labels.ndim == 1:
            labels = labels.reshape(1, -1)
        
        if labels.shape[1] != len(self.label_columns):
            raise ValueError(f"标签维度数不匹配：期望{len(self.label_columns)}，得到{labels.shape[1]}")
        
        denormalized = np.zeros_like(labels)
        
        for i, col_name in enumerate(self.label_columns):
            col_data = labels[:, i]
            min_val = self.label_mins.get(col_name, 0.0)
            max_val = self.label_maxs.get(col_name, 1.0)
            
            # 反归一化: x * (max - min) + min
            denormalized[:, i] = col_data * (max_val - min_val) + min_val
        
        # 恢复原始shape
        if len(original_shape) == 1:
            denormalized = denormalized.reshape(-1)
        
        return denormalized
    
    def save(self, filepath: str) -> None:
        """
        保存归一化参数到JSON文件
        
        Args:
            filepath: 保存路径
        """
        data = {
            'label_mins': self.label_mins,
            'label_maxs': self.label_maxs,
            'label_columns': self.label_columns
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"归一化参数已保存到: {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'LabelNormalizer':
        """
        从JSON文件加载归一化参数
        
        Args:
            filepath: 文件路径
        
        Returns:
            LabelNormalizer实例
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        normalizer = cls(
            label_mins=data['label_mins'],
            label_maxs=data['label_maxs']
        )
        
        logger.info(f"归一化参数已从 {filepath} 加载")
        return normalizer
    
    def to_dict(self) -> Dict:
        """转换为字典（用于保存到checkpoint）"""
        return {
            'label_mins': self.label_mins,
            'label_maxs': self.label_maxs,
            'label_columns': self.label_columns
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'LabelNormalizer':
        """从字典创建（用于从checkpoint加载）"""
        return cls(
            label_mins=data.get('label_mins', {}),
            label_maxs=data.get('label_maxs', {})
        )

