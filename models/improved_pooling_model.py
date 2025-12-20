"""
改进的模型架构
包含多种改进方案
"""
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from typing import Optional


class ImprovedPersonalityPredictor(nn.Module):
    """
    改进的性格预测模型
    改进点：
    1. 更好的特征池化（mean + max + cls）
    2. 多层MLP回归头
    3. 可选的分别编码新闻和评论
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
        separate_encoding: bool = False
    ):
        """
        Args:
            base_model_name: 基座模型名称
            num_labels: 输出标签数量（5个维度）
            dropout: dropout率
            freeze_base: 是否冻结基座模型参数
            use_improved_pooling: 是否使用改进的池化（mean + max + cls）
            use_mlp_head: 是否使用MLP回归头（否则使用单层线性）
            mlp_hidden_size: MLP隐藏层大小
            separate_encoding: 是否分别编码新闻和评论（需要修改数据加载器）
        """
        super(ImprovedPersonalityPredictor, self).__init__()
        
        self.base_model_name = base_model_name
        self.num_labels = num_labels
        self.use_improved_pooling = use_improved_pooling
        self.use_mlp_head = use_mlp_head
        self.separate_encoding = separate_encoding
        
        # 加载基座模型
        # 某些模型（如gte-multilingual）需要trust_remote_code=True
        self.base_model = AutoModel.from_pretrained(
            base_model_name,
            trust_remote_code=True
        )
        config = AutoConfig.from_pretrained(
            base_model_name,
            trust_remote_code=True
        )
        self.hidden_size = config.hidden_size
        
        if freeze_base:
            for param in self.base_model.parameters():
                param.requires_grad = False
        
        self.dropout = nn.Dropout(dropout)
        
        # 确定输入特征维度
        if use_improved_pooling:
            # mean + max + cls = 3 * hidden_size
            input_size = self.hidden_size * 3
        else:
            input_size = self.hidden_size
        
        # 回归头
        if use_mlp_head:
            self.regressor = nn.Sequential(
                nn.Linear(input_size, mlp_hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(mlp_hidden_size, mlp_hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(mlp_hidden_size // 2, num_labels)
            )
        else:
            self.regressor = nn.Linear(input_size, num_labels)
        
        # 初始化
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
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
        last_hidden_state[mask_expanded == 0] = float('-inf')
        max_output = torch.max(last_hidden_state, dim=1)[0]  # [batch_size, hidden_size]
        
        # 拼接三种池化方式
        pooled_output = torch.cat([cls_output, mean_output, max_output], dim=1)
        
        return pooled_output
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ):
        """
        Args:
            input_ids: token ids
            attention_mask: attention mask
            labels: 真实标签（训练时使用）
        
        Returns:
            outputs: 包含logits和loss的字典
        """
        # 通过基座模型获取特征
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        last_hidden_state = outputs.last_hidden_state
        
        # 特征池化
        if self.use_improved_pooling:
            pooled_output = self._pool_features(last_hidden_state, attention_mask)
        else:
            # 只使用[CLS] token
            pooled_output = last_hidden_state[:, 0, :]
        
        # Dropout
        pooled_output = self.dropout(pooled_output)
        
        # 回归预测
        logits = self.regressor(pooled_output)
        
        result = {'logits': logits}
        
        # 计算损失
        if labels is not None:
            # 使用Smooth L1 Loss，对异常值更鲁棒
            loss_fn = nn.SmoothL1Loss()
            loss = loss_fn(logits, labels)
            result['loss'] = loss
        
        return result

