"""
原始模型架构（支持元数据的 Late Fusion）
支持即插即用的基座模型，融合文本特征和人口统计学元数据
"""
import torch
import torch.nn as nn
from typing import Optional
from models.metadata_encoder import MetadataEncoder
from models.base_model_loader import load_base_model


class PersonalityPredictor(nn.Module):
    """
    性格预测模型（原始模式）
    支持不同的预训练模型作为基座（BERT, RoBERTa, DeBERTa等）
    融合文本特征和人口统计学元数据（默认全部使用）
    """
    
    def __init__(
        self,
        base_model_name: str = "roberta-base",
        num_labels: int = 5,
        dropout: float = 0.1,
        freeze_base: bool = False,
        use_mlp_head: bool = True,
        mlp_hidden_size: int = 256,
        local_files_only: bool = False,
        # 元数据编码维度
        gender_embed_dim: int = 8,
        education_embed_dim: int = 16,
        race_embed_dim: int = 8,
        age_norm_dim: int = 16,
        income_norm_dim: int = 16,
    ):
        """
        Args:
            base_model_name: 基座模型名称
            num_labels: 输出标签数量（5个维度）
            dropout: dropout率
            freeze_base: 是否冻结基座模型参数
            use_mlp_head: 是否使用MLP回归头（默认True）
            mlp_hidden_size: MLP隐藏层大小
            local_files_only: 是否只使用本地模型文件
            gender_embed_dim: 性别嵌入维度
            education_embed_dim: 教育程度嵌入维度
            race_embed_dim: 种族嵌入维度
            age_norm_dim: 年龄归一化后的MLP输出维度
            income_norm_dim: 收入归一化后的MLP输出维度
        """
        super(PersonalityPredictor, self).__init__()
        
        self.base_model_name = base_model_name
        self.num_labels = num_labels
        self.use_mlp_head = use_mlp_head
        
        # 加载基座模型（使用统一的加载函数）
        self.base_model, self.hidden_size = load_base_model(
            base_model_name=base_model_name,
            freeze_base=freeze_base,
            local_files_only=local_files_only
        )
        
        self.dropout = nn.Dropout(dropout)
        
        # 文本特征维度（使用[CLS] token）
        text_feature_dim = self.hidden_size
        
        # ========== 元数据编码器（默认全部使用）==========
        self.metadata_encoder = MetadataEncoder(
            gender_embed_dim=gender_embed_dim,
            education_embed_dim=education_embed_dim,
            race_embed_dim=race_embed_dim,
            age_norm_dim=age_norm_dim,
            income_norm_dim=income_norm_dim,
            dropout=dropout
        )
        
        # ========== 融合层 ==========
        fused_feature_dim = text_feature_dim + self.metadata_encoder.feature_dim
        
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
        
        # 初始化回归头
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重（仅回归头，元数据编码器有自己的初始化）"""
        if isinstance(self.regressor, nn.Linear):
            nn.init.xavier_uniform_(self.regressor.weight)
            if self.regressor.bias is not None:
                nn.init.zeros_(self.regressor.bias)
        elif isinstance(self.regressor, nn.Sequential):
            for module in self.regressor:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        # 元数据（默认全部使用）
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
        
        # 使用[CLS] token的表示
        text_features = outputs.last_hidden_state[:, 0, :]
        
        # Dropout
        text_features = self.dropout(text_features)
        
        # 编码元数据特征（使用统一的编码器）
        metadata_features = self.metadata_encoder(
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

