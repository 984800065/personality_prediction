"""
多实例学习数据加载器
按 speaker_id 组织数据，每个样本包含一个 speaker 的所有评论
"""
import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Tuple
import os
from loguru import logger
from data_loader import MetadataNormalizer


class MultiInstancePersonalityDataset(Dataset):
    """
    多实例性格预测数据集
    每个样本对应一个 speaker，包含该 speaker 的所有评论
    """
    
    def __init__(
        self,
        train_tsv_path: str,
        articles_csv_path: str,
        tokenizer,
        max_length: int = 512,
        is_training: bool = True,
        normalizer = None,  # LabelNormalizer实例，用于归一化标签
        # 元数据相关参数
        use_gender: bool = True,
        use_education: bool = True,
        use_race: bool = True,
        use_age: bool = True,
        use_income: bool = True,
        metadata_normalizer: Optional = None  # 用于归一化 age 和 income
    ):
        """
        Args:
            train_tsv_path: 训练/测试TSV文件路径
            articles_csv_path: 新闻文章CSV文件路径
            tokenizer: tokenizer对象
            max_length: 最大序列长度
            is_training: 是否为训练模式（训练模式包含标签）
            normalizer: LabelNormalizer实例，用于归一化标签（训练时使用）
            use_gender: 是否使用性别特征
            use_education: 是否使用教育程度特征
            use_race: 是否使用种族特征
            use_age: 是否使用年龄特征
            use_income: 是否使用收入特征
            metadata_normalizer: MetadataNormalizer实例，用于归一化 age 和 income
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_training = is_training
        self.normalizer = normalizer
        
        # 保存元数据字段启用状态
        self.use_gender = use_gender
        self.use_education = use_education
        self.use_race = use_race
        self.use_age = use_age
        self.use_income = use_income
        self.metadata_normalizer = metadata_normalizer
        
        # 加载数据
        self.data = pd.read_csv(train_tsv_path, sep='\t')
        self.articles = pd.read_csv(articles_csv_path)
        
        # 创建article_id到text的映射
        self.article_dict = dict(zip(
            self.articles['article_id'],
            self.articles['text']
        ))
        
        # 检查是否有 speaker_id 列
        if 'speaker_id' not in self.data.columns:
            raise ValueError("数据文件中必须包含 'speaker_id' 列")
        
        # 按 speaker_id 分组
        logger.info("按 speaker_id 组织数据...")
        self.speaker_groups = self.data.groupby('speaker_id')
        self.speaker_ids = sorted(self.speaker_groups.groups.keys())
        logger.info(f"找到 {len(self.speaker_ids)} 个不同的 speaker_id")
        
        # 处理每个 speaker 的数据
        self.speaker_data = {}
        self.speaker_labels = {}
        self.speaker_metadata = {}
        
        for speaker_id in self.speaker_ids:
            speaker_df = self.speaker_groups.get_group(speaker_id)
            
            # 提取标签（如果是训练模式）
            if self.is_training:
                label_columns = [
                    'personality_conscientiousness',
                    'personality_openess',
                    'personality_extraversion',
                    'personality_agreeableness',
                    'personality_stability'
                ]
                label_df = speaker_df[label_columns].copy()
                
                # 处理 'unknown' 值
                label_df = label_df.replace('unknown', pd.NA)
                for col in label_columns:
                    label_df[col] = pd.to_numeric(label_df[col], errors='coerce')
                
                # 检查是否有有效标签（至少有一行所有标签都有效）
                valid_mask = ~label_df.isna().any(axis=1)
                if valid_mask.sum() == 0:
                    # 如果该 speaker 的所有标签都无效，跳过
                    logger.warning(f"Speaker {speaker_id} 的所有标签都无效，跳过")
                    continue
                
                # 使用第一个有效标签（因为同一 speaker 的标签应该相同）
                valid_labels = label_df[valid_mask].iloc[0].values.astype(float)
                
                # 如果提供了normalizer，对标签进行归一化
                if self.normalizer is not None:
                    valid_labels = self.normalizer.normalize(valid_labels.reshape(1, -1)).flatten()
                
                self.speaker_labels[speaker_id] = valid_labels
            else:
                self.speaker_labels[speaker_id] = None
            
            # 提取元数据（使用第一个样本的元数据，因为同一 speaker 的元数据应该相同）
            first_row = speaker_df.iloc[0]
            
            metadata = {}
            
            # 性别
            if self.use_gender:
                gender_val = first_row['gender']
                if pd.isna(gender_val) or gender_val == 'unknown':
                    gender_val = 0
                else:
                    gender_val = int(gender_val)
                metadata['gender'] = float(gender_val)
            
            # 教育程度
            if self.use_education:
                education_val = first_row['education']
                if pd.isna(education_val) or education_val == 'unknown':
                    education_val = 0
                else:
                    education_val = int(education_val)
                    if education_val == 1:
                        education_val = 0
                metadata['education'] = float(education_val)
            
            # 种族
            if self.use_race:
                race_val = first_row['race']
                if pd.isna(race_val) or race_val == 'unknown':
                    race_val = 0
                else:
                    race_val = int(race_val)
                    valid_races = {1, 2, 3, 5, 6}
                    if race_val not in valid_races:
                        race_val = 0
                metadata['race'] = float(race_val)
            
            # 年龄
            if self.use_age:
                age_val = first_row['age']
                if pd.isna(age_val) or age_val == 'unknown':
                    age_val = np.nan
                else:
                    age_val = float(age_val)
                
                # 归一化
                if self.metadata_normalizer is not None:
                    age_array = np.array([age_val])
                    age_val = self.metadata_normalizer.normalize_age(age_array)[0]
                else:
                    # 如果没有 normalizer，先计算（仅用于训练）
                    if self.is_training:
                        valid_ages = speaker_df['age'].replace('unknown', pd.NA)
                        valid_ages = pd.to_numeric(valid_ages, errors='coerce')
                        valid_ages = valid_ages.dropna()
                        if len(valid_ages) > 0:
                            age_min, age_max = valid_ages.min(), valid_ages.max()
                            if age_max > age_min:
                                age_val = (age_val - age_min) / (age_max - age_min) if not np.isnan(age_val) else 0.0
                            else:
                                age_val = 0.5 if not np.isnan(age_val) else 0.0
                        age_val = np.clip(age_val, 0.0, 1.0) if not np.isnan(age_val) else 0.0
                
                metadata['age'] = float(age_val) if not np.isnan(age_val) else 0.0
            
            # 收入
            if self.use_income:
                income_val = first_row['income']
                if pd.isna(income_val) or income_val == 'unknown':
                    income_val = np.nan
                else:
                    income_val = float(income_val)
                
                # 归一化
                if self.metadata_normalizer is not None:
                    income_array = np.array([income_val])
                    income_val = self.metadata_normalizer.normalize_income(income_array)[0]
                else:
                    # 如果没有 normalizer，先计算（仅用于训练）
                    if self.is_training:
                        valid_incomes = speaker_df['income'].replace('unknown', pd.NA)
                        valid_incomes = pd.to_numeric(valid_incomes, errors='coerce')
                        valid_incomes = valid_incomes.dropna()
                        if len(valid_incomes) > 0:
                            income_min, income_max = valid_incomes.min(), valid_incomes.max()
                            if income_max > income_min:
                                income_val = (income_val - income_min) / (income_max - income_min) if not np.isnan(income_val) else 0.0
                            else:
                                income_val = 0.5 if not np.isnan(income_val) else 0.0
                        income_val = np.clip(income_val, 0.0, 1.0) if not np.isnan(income_val) else 0.0
                
                metadata['income'] = float(income_val) if not np.isnan(income_val) else 0.0
            
            self.speaker_metadata[speaker_id] = metadata
            
            # 保存该 speaker 的所有评论数据
            self.speaker_data[speaker_id] = speaker_df.reset_index(drop=True)
        
        # 更新 speaker_ids 列表（移除被跳过的）
        self.speaker_ids = [sid for sid in self.speaker_ids if sid in self.speaker_data]
        logger.info(f"有效 speaker 数量: {len(self.speaker_ids)}")
        
        # 统计每个 speaker 的评论数量
        comment_counts = [len(self.speaker_data[sid]) for sid in self.speaker_ids]
        logger.info(f"每个 speaker 的评论数: 最小={min(comment_counts)}, 最大={max(comment_counts)}, "
                   f"平均={np.mean(comment_counts):.2f}, 中位数={np.median(comment_counts):.2f}")
    
    def __len__(self):
        return len(self.speaker_ids)
    
    def __getitem__(self, idx):
        speaker_id = self.speaker_ids[idx]
        speaker_df = self.speaker_data[speaker_id]
        metadata = self.speaker_metadata[speaker_id]
        
        # 处理该 speaker 的所有评论
        input_ids_list = []
        attention_mask_list = []
        
        for _, row in speaker_df.iterrows():
            article_id = row['article_id']
            comment = str(row['comment'])
            
            # 获取对应的新闻文本
            article_text = self.article_dict.get(article_id, "")
            
            # 使用tokenizer处理文本对
            encoding = self.tokenizer(
                article_text,
                comment,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            # 移除batch维度
            input_ids_list.append(encoding['input_ids'].squeeze(0))
            attention_mask_list.append(encoding['attention_mask'].squeeze(0))
        
        result = {
            'input_ids_list': input_ids_list,
            'attention_mask_list': attention_mask_list,
            'speaker_id': speaker_id,
            'num_comments': len(input_ids_list),
        }
        
        # 添加元数据
        if self.use_gender:
            result['gender'] = torch.tensor(metadata['gender'], dtype=torch.float32)
        if self.use_education:
            result['education'] = torch.tensor(metadata['education'], dtype=torch.float32)
        if self.use_race:
            result['race'] = torch.tensor(metadata['race'], dtype=torch.float32)
        if self.use_age:
            result['age'] = torch.tensor(metadata['age'], dtype=torch.float32)
        if self.use_income:
            result['income'] = torch.tensor(metadata['income'], dtype=torch.float32)
        
        # 添加标签（如果是训练模式）
        if self.is_training and self.speaker_labels[speaker_id] is not None:
            result['labels'] = torch.tensor(self.speaker_labels[speaker_id], dtype=torch.float32)
        
        return result


def collate_fn_multi_instance(batch):
    """
    多实例学习的 collate 函数
    每个样本包含一个 speaker 的所有评论
    """
    # 提取所有 speaker 的评论列表
    input_ids_list = [item['input_ids_list'] for item in batch]
    attention_mask_list = [item['attention_mask_list'] for item in batch]
    
    result = {
        'input_ids_list': input_ids_list,
        'attention_mask_list': attention_mask_list,
        'speaker_ids': [item['speaker_id'] for item in batch],
        'num_comments': [item['num_comments'] for item in batch],
    }
    
    # 计算 comment_mask（用于标识哪些评论是有效的）
    max_num_comments = max([len(ids_list) for ids_list in input_ids_list])
    batch_size = len(batch)
    comment_mask = torch.zeros(batch_size, max_num_comments, dtype=torch.float32)
    
    for i, num_comments in enumerate(result['num_comments']):
        comment_mask[i, :num_comments] = 1.0
    
    result['comment_mask'] = comment_mask
    
    # 添加元数据（每个 speaker 一份）
    if 'gender' in batch[0]:
        result['gender'] = torch.stack([item['gender'] for item in batch])
    if 'education' in batch[0]:
        result['education'] = torch.stack([item['education'] for item in batch])
    if 'race' in batch[0]:
        result['race'] = torch.stack([item['race'] for item in batch])
    if 'age' in batch[0]:
        result['age'] = torch.stack([item['age'] for item in batch])
    if 'income' in batch[0]:
        result['income'] = torch.stack([item['income'] for item in batch])
    
    # 添加标签（如果是训练模式）
    if 'labels' in batch[0]:
        labels = torch.stack([item['labels'] for item in batch])
        result['labels'] = labels
    
    return result

