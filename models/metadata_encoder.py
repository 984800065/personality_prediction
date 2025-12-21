"""
元数据编码器
统一处理人口统计学元数据的编码（gender, education, race, age, income）
"""
import torch
import torch.nn as nn
from typing import Optional


class MetadataEncoder(nn.Module):
    """
    元数据编码器
    统一编码所有元数据字段（默认全部使用）
    """
    
    def __init__(
        self,
        gender_embed_dim: int = 8,
        education_embed_dim: int = 16,
        race_embed_dim: int = 8,
        age_norm_dim: int = 16,
        income_norm_dim: int = 16,
        dropout: float = 0.1
    ):
        """
        Args:
            gender_embed_dim: 性别嵌入维度
            education_embed_dim: 教育程度嵌入维度
            race_embed_dim: 种族嵌入维度
            age_norm_dim: 年龄归一化后的MLP输出维度
            income_norm_dim: 收入归一化后的MLP输出维度
            dropout: dropout率
        """
        super(MetadataEncoder, self).__init__()
        
        # 性别编码（分类，1-2）
        self.gender_embedding = nn.Embedding(
            num_embeddings=3,  # 0: unknown, 1: male, 2: female
            embedding_dim=gender_embed_dim
        )
        
        # 教育程度编码（分类，2-7）
        self.education_embedding = nn.Embedding(
            num_embeddings=8,  # 0: unknown, 1-7: education levels
            embedding_dim=education_embed_dim
        )
        
        # 种族编码（分类，1-6）
        self.race_embedding = nn.Embedding(
            num_embeddings=7,  # 0: unknown, 1-6: race categories
            embedding_dim=race_embed_dim
        )
        
        # 年龄编码（数值，归一化后通过MLP）
        self.age_mlp = nn.Sequential(
            nn.Linear(1, age_norm_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(age_norm_dim, age_norm_dim)
        )
        
        # 收入编码（数值，归一化后通过MLP）
        self.income_mlp = nn.Sequential(
            nn.Linear(1, income_norm_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(income_norm_dim, income_norm_dim)
        )
        
        # 计算总特征维度
        self.feature_dim = (
            gender_embed_dim +
            education_embed_dim +
            race_embed_dim +
            age_norm_dim +
            income_norm_dim
        )
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)
    
    def forward(
        self,
        gender: Optional[torch.Tensor] = None,
        education: Optional[torch.Tensor] = None,
        race: Optional[torch.Tensor] = None,
        age: Optional[torch.Tensor] = None,
        income: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        编码元数据特征
        
        Args:
            gender: [batch_size] 或 None (1: 男性, 2: 女性, 0/unknown: 0)
            education: [batch_size] 或 None (2-7, unknown: 0)
            race: [batch_size] 或 None (1-6, unknown: 0)
            age: [batch_size] 或 None (已归一化到[0,1])
            income: [batch_size] 或 None (已归一化到[0,1])
        
        Returns:
            metadata_features: [batch_size, feature_dim]
        """
        metadata_features = []
        
        if gender is not None:
            # gender: 1->1, 2->2, unknown/0->0
            gender_idx = gender.long().clamp(0, 2)
            gender_emb = self.gender_embedding(gender_idx)
            metadata_features.append(gender_emb)
        
        if education is not None:
            # education: 2-7 -> 2-7, unknown/0/1 -> 0
            education_idx = education.long().clamp(0, 7)
            education_emb = self.education_embedding(education_idx)
            metadata_features.append(education_emb)
        
        if race is not None:
            # race: 1->1, 2->2, 3->3, 5->5, 6->6, unknown/0/4 -> 0
            race_idx = race.long().clamp(0, 6)
            race_emb = self.race_embedding(race_idx)
            metadata_features.append(race_emb)
        
        if age is not None:
            # age: 已归一化的数值，通过MLP
            age_input = age.unsqueeze(-1)  # [batch_size, 1]
            age_feat = self.age_mlp(age_input)
            metadata_features.append(age_feat)
        
        if income is not None:
            # income: 已归一化的数值，通过MLP
            income_input = income.unsqueeze(-1)  # [batch_size, 1]
            income_feat = self.income_mlp(income_input)
            metadata_features.append(income_feat)
        
        if len(metadata_features) == 0:
            # 如果没有提供任何元数据，返回空tensor
            batch_size = gender.size(0) if gender is not None else (
                education.size(0) if education is not None else (
                    race.size(0) if race is not None else (
                        age.size(0) if age is not None else income.size(0)
                    )
                )
            )
            return torch.zeros(batch_size, 0, device=next(self.parameters()).device)
        
        # 拼接所有元数据特征
        return torch.cat(metadata_features, dim=1)

