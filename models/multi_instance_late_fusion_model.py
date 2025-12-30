"""
Multi-Instance Late Fusion 架构模型
基于 Late Fusion 模型，支持多实例学习（Multi-Instance Learning）
对同一个 speaker 的所有评论进行聚合，预测统一的大五人格
"""
import torch
import torch.nn as nn
from typing import Optional, Dict, List
import math
from models.metadata_encoder import MetadataEncoder
from models.base_model_loader import load_base_model


class MultiInstanceLateFusionPersonalityPredictor(nn.Module):
    """
    Multi-Instance Late Fusion 性格预测模型
    对同一个 speaker 的所有评论进行聚合，预测统一的大五人格
    融合文本特征和人口统计学元数据
    """
    
    def __init__(
        self,
        base_model_name: str = "roberta-base",
        num_labels: int = 5,
        dropout: float = 0.1,
        freeze_base: bool = False,
        use_improved_pooling: bool = False,  # 聚合模式默认不使用改进池化
        use_mlp_head: bool = True,  # 聚合模式默认使用MLP回归头
        mlp_hidden_size: int = 256,
        local_files_only: bool = False,
        use_metadata: bool = True,  # 是否使用metadata
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
        age_norm_dim: int = 16,
        income_norm_dim: int = 16,
        # 多实例聚合参数
        aggregation_method: str = "attention",  # "attention", "mean", "max", "transformer"
        aggregation_hidden_size: int = 256,  # 用于 attention 和 transformer 聚合
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
            use_metadata: 是否使用metadata（控制分类头上是否有metadata的维度拼接），默认True
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
            aggregation_method: 聚合方法 ("attention", "mean", "max", "transformer")
            aggregation_hidden_size: 聚合层的隐藏层大小
        """
        super(MultiInstanceLateFusionPersonalityPredictor, self).__init__()
        
        self.base_model_name = base_model_name
        self.num_labels = num_labels
        self.use_improved_pooling = use_improved_pooling
        self.use_mlp_head = use_mlp_head
        self.aggregation_method = aggregation_method
        self.use_metadata = use_metadata
        
        # 保存元数据字段启用状态
        self.use_gender = use_gender
        self.use_education = use_education
        self.use_race = use_race
        self.use_age = use_age
        self.use_income = use_income
        
        # 加载基座模型（使用统一的加载函数）
        self.base_model, self.hidden_size = load_base_model(
            base_model_name=base_model_name,
            freeze_base=freeze_base,
            local_files_only=local_files_only
        )
        
        self.dropout = nn.Dropout(dropout)
        
        # 文本特征维度
        if use_improved_pooling:
            text_feature_dim = self.hidden_size * 3  # mean + max + cls
        else:
            text_feature_dim = self.hidden_size
        
        # ========== 多实例聚合层 ==========
        self.aggregation_hidden_size = aggregation_hidden_size
        
        if aggregation_method == "attention":
            # 注意力聚合
            self.attention_query = nn.Linear(text_feature_dim, aggregation_hidden_size)
            self.attention_key = nn.Linear(text_feature_dim, aggregation_hidden_size)
            self.attention_value = nn.Linear(text_feature_dim, aggregation_hidden_size)
            self.attention_scale = 1.0 / math.sqrt(aggregation_hidden_size)
            aggregated_text_feature_dim = aggregation_hidden_size
        elif aggregation_method == "transformer":
            # Transformer 聚合（使用简单的 Transformer Encoder）
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=text_feature_dim,
                nhead=8,
                dim_feedforward=aggregation_hidden_size,
                dropout=dropout,
                batch_first=True
            )
            self.transformer_aggregator = nn.TransformerEncoder(encoder_layer, num_layers=2)
            # 使用 [CLS] token 或平均池化
            aggregated_text_feature_dim = text_feature_dim
        else:
            # mean 或 max 聚合，不需要额外参数
            aggregated_text_feature_dim = text_feature_dim
        
        # ========== 元数据编码器（根据use_metadata决定是否创建）==========
        if use_metadata:
            self.metadata_encoder = MetadataEncoder(
                gender_embed_dim=gender_embed_dim,
                education_embed_dim=education_embed_dim,
                race_embed_dim=race_embed_dim,
                age_norm_dim=age_norm_dim,
                income_norm_dim=income_norm_dim,
                dropout=dropout
            )
            metadata_dim = self.metadata_encoder.feature_dim
        else:
            self.metadata_encoder = None
            metadata_dim = 0
        
        # ========== 融合层 ==========
        # 聚合后的文本特征 + 元数据特征的总维度
        fused_feature_dim = aggregated_text_feature_dim + metadata_dim
        
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
        """初始化权重（仅回归头和聚合层，元数据编码器有自己的初始化）"""
        for module in self.modules():
            # 跳过元数据编码器的模块（它有自己的初始化）
            if hasattr(module, '__class__') and 'MetadataEncoder' in str(module.__class__):
                continue
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
    
    def _aggregate_comments(
        self,
        comment_features: torch.Tensor,
        comment_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        聚合同一个 speaker 的多个评论特征
        
        Args:
            comment_features: [num_comments, text_feature_dim] 或 [batch_size, num_comments, text_feature_dim]
            comment_mask: [num_comments] 或 [batch_size, num_comments]，1表示有效，0表示padding
        
        Returns:
            aggregated_features: [text_feature_dim] 或 [batch_size, aggregated_text_feature_dim]
        """
        if comment_features.dim() == 2:
            # 单个 speaker: [num_comments, text_feature_dim]
            num_comments, text_feature_dim = comment_features.shape
            comment_features = comment_features.unsqueeze(0)  # [1, num_comments, text_feature_dim]
            if comment_mask is not None:
                comment_mask = comment_mask.unsqueeze(0)  # [1, num_comments]
            batch_mode = False
        else:
            # batch mode: [batch_size, num_comments, text_feature_dim]
            batch_mode = True
        
        batch_size, num_comments, text_feature_dim = comment_features.shape
        
        if self.aggregation_method == "attention":
            # 注意力聚合
            query = self.attention_query(comment_features)  # [batch_size, num_comments, hidden_size]
            key = self.attention_key(comment_features)  # [batch_size, num_comments, hidden_size]
            value = self.attention_value(comment_features)  # [batch_size, num_comments, hidden_size]
            
            # 计算注意力分数
            scores = torch.bmm(query, key.transpose(1, 2)) * self.attention_scale  # [batch_size, num_comments, num_comments]
            
            # 应用 mask（如果有）
            if comment_mask is not None:
                mask_expanded = comment_mask.unsqueeze(1).expand(-1, num_comments, -1)  # [batch_size, num_comments, num_comments]
                scores = scores.masked_fill(mask_expanded == 0, float('-inf'))
            
            # Softmax
            attn_weights = torch.softmax(scores, dim=-1)  # [batch_size, num_comments, num_comments]
            
            # 加权聚合
            aggregated = torch.bmm(attn_weights, value)  # [batch_size, num_comments, hidden_size]
            
            # 对每个 comment 的聚合结果再次聚合（使用平均）
            aggregated = aggregated.mean(dim=1)  # [batch_size, hidden_size]
            
        elif self.aggregation_method == "transformer":
            # Transformer 聚合
            # comment_features: [batch_size, num_comments, text_feature_dim]
            aggregated = self.transformer_aggregator(comment_features)  # [batch_size, num_comments, text_feature_dim]
            
            # 使用平均池化或第一个位置
            if comment_mask is not None:
                mask_expanded = comment_mask.unsqueeze(-1).expand_as(aggregated).float()
                aggregated = (aggregated * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1e-9)
            else:
                aggregated = aggregated.mean(dim=1)  # [batch_size, text_feature_dim]
                
        elif self.aggregation_method == "mean":
            # 平均聚合
            if comment_mask is not None:
                mask_expanded = comment_mask.unsqueeze(-1).expand_as(comment_features).float()
                aggregated = (comment_features * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1e-9)
            else:
                aggregated = comment_features.mean(dim=1)  # [batch_size, text_feature_dim]
                
        elif self.aggregation_method == "max":
            # 最大池化聚合
            if comment_mask is not None:
                mask_expanded = comment_mask.unsqueeze(-1).expand_as(comment_features).float()
                comment_features_masked = comment_features.clone()
                comment_features_masked[mask_expanded == 0] = float('-inf')
                aggregated = torch.max(comment_features_masked, dim=1)[0]
            else:
                aggregated = torch.max(comment_features, dim=1)[0]  # [batch_size, text_feature_dim]
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation_method}")
        
        if not batch_mode:
            aggregated = aggregated.squeeze(0)  # [text_feature_dim] 或 [aggregated_text_feature_dim]
        
        return aggregated
    
    def forward(
        self,
        # 多实例输入：每个 speaker 的多个评论
        input_ids_list: List[torch.Tensor],  # List of [num_comments_i, max_length]
        attention_mask_list: List[torch.Tensor],  # List of [num_comments_i, max_length]
        comment_mask: Optional[torch.Tensor] = None,  # [batch_size, max_num_comments]，1表示有效，0表示padding
        labels: Optional[torch.Tensor] = None,
        # 元数据（每个 speaker 一份，不是每个 comment 一份）
        gender: Optional[torch.Tensor] = None,
        education: Optional[torch.Tensor] = None,
        race: Optional[torch.Tensor] = None,
        age: Optional[torch.Tensor] = None,
        income: Optional[torch.Tensor] = None
    ):
        """
        Args:
            input_ids_list: List of [num_comments_i, max_length]，每个元素是一个 speaker 的所有评论
            attention_mask_list: List of [num_comments_i, max_length]
            comment_mask: [batch_size, max_num_comments]，用于标识哪些评论是有效的（用于 batch 模式）
            labels: [batch_size, num_labels] 真实标签（训练时使用）
            gender: [batch_size] 或 None
            education: [batch_size] 或 None
            race: [batch_size] 或 None
            age: [batch_size] 或 None (已归一化)
            income: [batch_size] 或 None (已归一化)
        
        Returns:
            outputs: 包含logits和loss的字典
        """
        batch_size = len(input_ids_list)
        all_comment_features = []
        
        # 对每个 speaker 的所有评论分别编码
        for i in range(batch_size):
            input_ids = input_ids_list[i]  # [num_comments_i, max_length]
            attention_mask = attention_mask_list[i]  # [num_comments_i, max_length]
            
            # 通过基座模型获取每个评论的特征
            outputs = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            last_hidden_state = outputs.last_hidden_state  # [num_comments_i, max_length, hidden_size]
            
            # 对每个评论进行特征池化
            if self.use_improved_pooling:
                # [num_comments_i, text_feature_dim]
                comment_features = self._pool_features_batch(last_hidden_state, attention_mask)
            else:
                # [num_comments_i, text_feature_dim]
                comment_features = last_hidden_state[:, 0, :]  # [CLS] token
            
            all_comment_features.append(comment_features)
        
        # 将所有 speaker 的特征 pad 到相同长度（用于 batch 处理）
        max_num_comments = max([feat.shape[0] for feat in all_comment_features])
        text_feature_dim = all_comment_features[0].shape[1]
        
        padded_features = []
        # 如果 comment_mask 为 None，创建一个新的
        if comment_mask is None:
            comment_mask = torch.zeros(batch_size, max_num_comments, dtype=torch.float32, 
                                      device=all_comment_features[0].device)
            # 填充 mask
            for i, feat in enumerate(all_comment_features):
                num_comments = feat.shape[0]
                comment_mask[i, :num_comments] = 1.0
        
        for i, feat in enumerate(all_comment_features):
            num_comments = feat.shape[0]
            if num_comments < max_num_comments:
                # Padding
                padding = torch.zeros(max_num_comments - num_comments, text_feature_dim, 
                                    device=feat.device)
                padded_feat = torch.cat([feat, padding], dim=0)  # [max_num_comments, text_feature_dim]
            else:
                padded_feat = feat
            
            padded_features.append(padded_feat)
        
        # Stack: [batch_size, max_num_comments, text_feature_dim]
        stacked_features = torch.stack(padded_features, dim=0)
        
        # Dropout
        stacked_features = self.dropout(stacked_features)
        
        # 聚合每个 speaker 的所有评论特征
        aggregated_text_features = []
        for i in range(batch_size):
            speaker_features = stacked_features[i]  # [max_num_comments, text_feature_dim]
            speaker_mask = comment_mask[i]  # [max_num_comments]
            aggregated = self._aggregate_comments(speaker_features, speaker_mask)
            aggregated_text_features.append(aggregated)
        
        # Stack: [batch_size, aggregated_text_feature_dim]
        aggregated_text_features = torch.stack(aggregated_text_features, dim=0)
        
        # 编码元数据特征（根据use_metadata决定是否使用）
        if self.use_metadata and self.metadata_encoder is not None:
            metadata_features = self.metadata_encoder(
                gender=gender,
                education=education,
                race=race,
                age=age,
                income=income
            )
            # Late Fusion: 拼接聚合后的文本特征和元数据特征
            if metadata_features.size(1) > 0:
                fused_features = torch.cat([aggregated_text_features, metadata_features], dim=1)
            else:
                fused_features = aggregated_text_features
        else:
            # 不使用metadata，只使用聚合后的文本特征
            fused_features = aggregated_text_features
        
        # 回归预测
        logits = self.regressor(fused_features)
        
        result = {'logits': logits}
        
        # 计算损失
        if labels is not None:
            loss_fn = nn.SmoothL1Loss()
            loss = loss_fn(logits, labels)
            result['loss'] = loss
        
        return result
    
    def _pool_features_batch(self, last_hidden_state, attention_mask):
        """
        批量池化特征（用于处理多个评论）
        
        Args:
            last_hidden_state: [num_comments, max_length, hidden_size]
            attention_mask: [num_comments, max_length]
        
        Returns:
            pooled_output: [num_comments, text_feature_dim]
        """
        num_comments = last_hidden_state.shape[0]
        pooled_list = []
        
        for i in range(num_comments):
            hidden = last_hidden_state[i:i+1]  # [1, max_length, hidden_size]
            mask = attention_mask[i:i+1]  # [1, max_length]
            pooled = self._pool_features(hidden, mask)  # [1, text_feature_dim]
            pooled_list.append(pooled.squeeze(0))  # [text_feature_dim]
        
        return torch.stack(pooled_list, dim=0)  # [num_comments, text_feature_dim]

