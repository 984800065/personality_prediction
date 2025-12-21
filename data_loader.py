"""
数据加载和预处理模块
"""
import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Dict, List, Optional
import os
from loguru import logger


class PersonalityDataset(Dataset):
    """性格预测数据集"""
    
    def __init__(
        self,
        train_tsv_path: str,
        articles_csv_path: str,
        tokenizer,
        max_length: int = 512,
        is_training: bool = True,
        normalizer = None  # LabelNormalizer实例，用于归一化标签
    ):
        """
        Args:
            train_tsv_path: 训练/测试TSV文件路径
            articles_csv_path: 新闻文章CSV文件路径
            tokenizer: tokenizer对象
            max_length: 最大序列长度
            is_training: 是否为训练模式（训练模式包含标签）
            normalizer: LabelNormalizer实例，用于归一化标签（训练时使用）
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_training = is_training
        self.normalizer = normalizer
        
        # 加载数据
        self.data = pd.read_csv(train_tsv_path, sep='\t')
        self.articles = pd.read_csv(articles_csv_path)
        
        # 创建article_id到text的映射
        self.article_dict = dict(zip(
            self.articles['article_id'],
            self.articles['text']
        ))
        
        # 如果是训练模式，提取标签
        if self.is_training:
            # 提取标签列
            label_columns = [
                'personality_conscientiousness',
                'personality_openess',
                'personality_extraversion',
                'personality_agreeableness',
                'personality_stability'
            ]
            label_df = self.data[label_columns].copy()
            
            # 将 'unknown' 替换为 NaN，然后转换为数字
            label_df = label_df.replace('unknown', pd.NA)
            
            # 尝试转换为float，无法转换的会变成NaN
            for col in label_columns:
                label_df[col] = pd.to_numeric(label_df[col], errors='coerce')
            
            # 删除包含NaN的行（即标签为unknown的行）
            valid_mask = ~label_df.isna().any(axis=1)
            self.data = self.data[valid_mask].reset_index(drop=True)
            label_df = label_df[valid_mask]
            
            # 转换为numpy数组
            self.labels = label_df.values.astype(float)
            
            # 如果提供了normalizer，对标签进行归一化
            if self.normalizer is not None:
                self.labels = self.normalizer.normalize(self.labels)
                logger.info("标签已归一化到[0, 1]范围")
            
            removed_count = (~valid_mask).sum()
            if removed_count > 0:
                logger.warning(f"过滤掉了 {removed_count} 个包含 'unknown' 标签的样本，剩余 {len(self.data)} 个有效样本")
        else:
            self.labels = None
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        article_id = row['article_id']
        comment = str(row['comment'])
        
        # 获取对应的新闻文本
        article_text = self.article_dict.get(article_id, "")
        
        # 使用tokenizer的encode_plus方法处理文本对
        # 这样会自动添加特殊token（如[SEP]或</s>）
        encoding = self.tokenizer(
            article_text,
            comment,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # 移除batch维度（DataLoader会添加）
        encoding = {k: v.squeeze(0) for k, v in encoding.items()}
        
        result = {
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask'],
            "article_text": article_text,
            "comment": comment,
        }
        
        # 添加标签（如果是训练模式）
        if self.is_training:
            result['labels'] = torch.tensor(self.labels[idx], dtype=torch.float32)
        
        result['article_id'] = article_id
        result['comment_id'] = row.get('comment_id', idx)
        
        return result


def collate_fn(batch):
    """自定义collate函数"""
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    
    result = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'article_ids': [item['article_id'] for item in batch],
        'comment_ids': [item['comment_id'] for item in batch],
    }
    
    # 如果有标签，也stack起来
    if 'labels' in batch[0]:
        labels = torch.stack([item['labels'] for item in batch])
        result['labels'] = labels
    
    return result

