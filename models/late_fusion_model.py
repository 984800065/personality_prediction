"""
Late Fusion 架构模型
融合 RoBERTa 编码的文本特征与人口统计学元数据
"""
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from typing import Optional, Dict


class LateFusionPersonalityPredictor(nn.Module):
    """
    Late Fusion 性格预测模型
    融合文本特征和人口统计学元数据
    """
    
    def __init__(
        self,
        base_model_name: str = "roberta-base",
        num_labels: int = 5,
        dropout: float = 0.1,
        freeze_base: bool = False,
        use_improved_pooling: bool = True,
        use_mlp_head: bool = True,
        mlp_hidden_size: int = 256,
        local_files_only: bool = False,
        # 元数据字段启用/禁用
        use_gender: bool = True,
        use_education: bool = True,
        use_race: bool = True,
        use_age: bool = True,
        use_income: bool = True,
        # 元数据编码维度
        gender_embed_dim: int = 8,
        education_embed_dim: int = 16,
        race_embed_dim: int = 8,
        age_norm_dim: int = 16,  # 归一化后的年龄经过MLP后的维度
        income_norm_dim: int = 16,  # 归一化后的收入经过MLP后的维度
    ):
        """
        Args:
            base_model_name: 基座模型名称
            num_labels: 输出标签数量（5个维度）
            dropout: dropout率
            freeze_base: 是否冻结基座模型参数
            use_improved_pooling: 是否使用改进的池化（mean + max + cls）
            use_mlp_head: 是否使用MLP回归头
            mlp_hidden_size: MLP隐藏层大小
            local_files_only: 是否只使用本地模型文件
            use_gender: 是否使用性别特征
            use_education: 是否使用教育程度特征
            use_race: 是否使用种族特征
            use_age: 是否使用年龄特征
            use_income: 是否使用收入特征
            gender_embed_dim: 性别嵌入维度
            education_embed_dim: 教育程度嵌入维度
            race_embed_dim: 种族嵌入维度
            age_norm_dim: 年龄归一化后的MLP输出维度
            income_norm_dim: 收入归一化后的MLP输出维度
        """
        super(LateFusionPersonalityPredictor, self).__init__()
        
        self.base_model_name = base_model_name
        self.num_labels = num_labels
        self.use_improved_pooling = use_improved_pooling
        self.use_mlp_head = use_mlp_head
        
        # 保存元数据字段启用状态
        self.use_gender = use_gender
        self.use_education = use_education
        self.use_race = use_race
        self.use_age = use_age
        self.use_income = use_income
        
        # 加载基座模型
        self.base_model = AutoModel.from_pretrained(
            base_model_name,
            trust_remote_code=True,
            local_files_only=local_files_only
        )
        config = AutoConfig.from_pretrained(
            base_model_name,
            trust_remote_code=True,
            local_files_only=local_files_only
        )
        self.hidden_size = config.hidden_size
        
        if freeze_base:
            for param in self.base_model.parameters():
                param.requires_grad = False
        
        self.dropout = nn.Dropout(dropout)
        
        # 文本特征维度
        if use_improved_pooling:
            text_feature_dim = self.hidden_size * 3  # mean + max + cls
        else:
            text_feature_dim = self.hidden_size
        
        # ========== 元数据编码层 ==========
        metadata_feature_dim = 0
        
        # 性别编码（分类，1-2）
        if use_gender:
            self.gender_embedding = nn.Embedding(
                num_embeddings=3,  # 0: unknown, 1: male, 2: female
                embedding_dim=gender_embed_dim
            )
            metadata_feature_dim += gender_embed_dim
        else:
            self.gender_embedding = None
        
        # 教育程度编码（分类，2-7）
        if use_education:
            self.education_embedding = nn.Embedding(
                num_embeddings=8,  # 0: unknown, 1-7: education levels
                embedding_dim=education_embed_dim
            )
            metadata_feature_dim += education_embed_dim
        else:
            self.education_embedding = None
        
        # 种族编码（分类，1-6）
        if use_race:
            self.race_embedding = nn.Embedding(
                num_embeddings=7,  # 0: unknown, 1-6: race categories
                embedding_dim=race_embed_dim
            )
            metadata_feature_dim += race_embed_dim
        else:
            self.race_embedding = None
        
        # 年龄编码（数值，归一化后通过MLP）
        if use_age:
            self.age_mlp = nn.Sequential(
                nn.Linear(1, age_norm_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(age_norm_dim, age_norm_dim)
            )
            metadata_feature_dim += age_norm_dim
        else:
            self.age_mlp = None
        
        # 收入编码（数值，归一化后通过MLP）
        if use_income:
            self.income_mlp = nn.Sequential(
                nn.Linear(1, income_norm_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(income_norm_dim, income_norm_dim)
            )
            metadata_feature_dim += income_norm_dim
        else:
            self.income_mlp = None
        
        # ========== 融合层 ==========
        # 文本特征 + 元数据特征的总维度
        fused_feature_dim = text_feature_dim + metadata_feature_dim
        
        # 回归头
        if use_mlp_head:
            self.regressor = nn.Sequential(
                nn.Linear(fused_feature_dim, mlp_hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(mlp_hidden_size, mlp_hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(mlp_hidden_size // 2, num_labels)
            )
        else:
            self.regressor = nn.Linear(fused_feature_dim, num_labels)
        
        # 初始化
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
    
    def _pool_features(self, last_hidden_state, attention_mask):
        """
        改进的特征池化：mean + max + cls
        """
        # [CLS] token (第一个token)
        cls_output = last_hidden_state[:, 0, :]  # [batch_size, hidden_size]
        
        # Mean pooling (考虑attention mask)
        mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        mean_output = sum_embeddings / sum_mask  # [batch_size, hidden_size]
        
        # Max pooling (考虑attention mask)
        mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        last_hidden_state_masked = last_hidden_state.clone()
        last_hidden_state_masked[mask_expanded == 0] = float('-inf')
        max_output = torch.max(last_hidden_state_masked, dim=1)[0]  # [batch_size, hidden_size]
        
        # 拼接三种池化方式
        pooled_output = torch.cat([cls_output, mean_output, max_output], dim=1)
        
        return pooled_output
    
    def _encode_metadata(
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
            gender: [batch_size] 或 None
            education: [batch_size] 或 None
            race: [batch_size] 或 None
            age: [batch_size] 或 None (已归一化)
            income: [batch_size] 或 None (已归一化)
        
        Returns:
            metadata_features: [batch_size, metadata_feature_dim]
        """
        metadata_features = []
        
        if self.use_gender and gender is not None:
            # gender: 1->1, 2->2, unknown/0->0
            gender_idx = gender.long().clamp(0, 2)
            gender_emb = self.gender_embedding(gender_idx)
            metadata_features.append(gender_emb)
        
        if self.use_education and education is not None:
            # education: 2-7 -> 2-7, unknown/0/1 -> 0
            education_idx = education.long().clamp(0, 7)
            education_emb = self.education_embedding(education_idx)
            metadata_features.append(education_emb)
        
        if self.use_race and race is not None:
            # race: 1->1, 2->2, 3->3, 5->5, 6->6, unknown/0/4 -> 0
            race_idx = race.long().clamp(0, 6)
            race_emb = self.race_embedding(race_idx)
            metadata_features.append(race_emb)
        
        if self.use_age and age is not None:
            # age: 已归一化的数值，通过MLP
            age_input = age.unsqueeze(-1)  # [batch_size, 1]
            age_feat = self.age_mlp(age_input)
            metadata_features.append(age_feat)
        
        if self.use_income and income is not None:
            # income: 已归一化的数值，通过MLP
            income_input = income.unsqueeze(-1)  # [batch_size, 1]
            income_feat = self.income_mlp(income_input)
            metadata_features.append(income_feat)
        
        if len(metadata_features) == 0:
            # 如果没有使用任何元数据，返回空tensor
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
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        # 元数据（可选）
        gender: Optional[torch.Tensor] = None,
        education: Optional[torch.Tensor] = None,
        race: Optional[torch.Tensor] = None,
        age: Optional[torch.Tensor] = None,
        income: Optional[torch.Tensor] = None
    ):
        """
        Args:
            input_ids: token ids
            attention_mask: attention mask
            labels: 真实标签（训练时使用）
            gender: 性别 (1: 男性, 2: 女性, 0/unknown: 0)
            education: 教育程度 (2-7, unknown: 0)
            race: 种族 (1-6, unknown: 0)
            age: 年龄（已归一化到[0,1]）
            income: 收入（已归一化到[0,1]）
        
        Returns:
            outputs: 包含logits和loss的字典
        """
        # 通过基座模型获取文本特征
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        last_hidden_state = outputs.last_hidden_state
        
        # 文本特征池化
        if self.use_improved_pooling:
            text_features = self._pool_features(last_hidden_state, attention_mask)
        else:
            text_features = last_hidden_state[:, 0, :]  # [CLS] token
        
        # Dropout
        text_features = self.dropout(text_features)
        
        # 编码元数据特征
        metadata_features = self._encode_metadata(
            gender=gender,
            education=education,
            race=race,
            age=age,
            income=income
        )
        
        # Late Fusion: 拼接文本特征和元数据特征
        if metadata_features.size(1) > 0:
            fused_features = torch.cat([text_features, metadata_features], dim=1)
        else:
            fused_features = text_features
        
        # 回归预测
        logits = self.regressor(fused_features)
        
        result = {'logits': logits}
        
        # 计算损失
        if labels is not None:
            loss_fn = nn.SmoothL1Loss()
            loss = loss_fn(logits, labels)
            result['loss'] = loss
        
        return result

