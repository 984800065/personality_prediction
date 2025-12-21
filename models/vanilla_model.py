"""
模块化的模型架构，支持即插即用的基座模型
"""
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from typing import Optional


class PersonalityPredictor(nn.Module):
    """
    性格预测模型
    支持不同的预训练模型作为基座（BERT, RoBERTa, DeBERTa等）
    """
    
    def __init__(
        self,
        base_model_name: str = "roberta-base",
        num_labels: int = 5,
        dropout: float = 0.1,
        freeze_base: bool = False,
        local_files_only: bool = False
    ):
        """
        Args:
            base_model_name: 基座模型名称，可以是：
                - "bert-base-uncased"
                - "roberta-base"
                - "microsoft/deberta-base"
                - 或其他transformers支持的模型
            num_labels: 输出标签数量（5个维度）
            dropout: dropout率
            freeze_base: 是否冻结基座模型参数
            local_files_only: 是否只使用本地模型文件，不连接 Hugging Face
        """
        super(PersonalityPredictor, self).__init__()
        
        self.base_model_name = base_model_name
        self.num_labels = num_labels
        
        # 加载基座模型
        # 某些模型（如gte-multilingual）需要trust_remote_code=True
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
        
        # 如果冻结基座模型
        if freeze_base:
            for param in self.base_model.parameters():
                param.requires_grad = False
        
        # Dropout层
        self.dropout = nn.Dropout(dropout)
        
        # 回归头：预测5个维度
        self.regressor = nn.Linear(self.hidden_size, num_labels)
        
        # 初始化回归头
        nn.init.xavier_uniform_(self.regressor.weight)
        nn.init.zeros_(self.regressor.bias)
    
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
        
        # 使用[CLS] token的表示
        pooled_output = outputs.last_hidden_state[:, 0, :]
        
        # Dropout
        pooled_output = self.dropout(pooled_output)
        
        # 回归预测
        logits = self.regressor(pooled_output)
        
        result = {'logits': logits}
        
        # 计算损失（如果提供了标签）
        if labels is not None:
            loss_fn = nn.MSELoss()
            loss = loss_fn(logits, labels)
            result['loss'] = loss
        
        return result

