"""
数据加载和预处理模块
支持加载人口统计学元数据（gender, education, race, age, income）
"""
import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Dict, List, Optional
import os
from loguru import logger


class MetadataNormalizer:
    """元数据归一化器（用于 age 和 income）"""
    
    def __init__(self):
        self.age_min = None
        self.age_max = None
        self.income_min = None
        self.income_max = None
    
    def fit(self, age_data: np.ndarray, income_data: np.ndarray):
        """从数据中计算归一化参数"""
        # 过滤掉 NaN 和 unknown
        valid_age = age_data[~np.isnan(age_data)]
        valid_income = income_data[~np.isnan(income_data)]
        
        if len(valid_age) > 0:
            self.age_min = float(valid_age.min())
            self.age_max = float(valid_age.max())
        
        if len(valid_income) > 0:
            self.income_min = float(valid_income.min())
            self.income_max = float(valid_income.max())
        
        return self
    
    def normalize_age(self, age: np.ndarray) -> np.ndarray:
        """归一化年龄到[0, 1]"""
        if self.age_min is None or self.age_max is None:
            return age
        age = np.asarray(age, dtype=np.float32)
        # 处理 NaN
        mask = ~np.isnan(age)
        normalized = np.zeros_like(age)
        if self.age_max > self.age_min:
            normalized[mask] = (age[mask] - self.age_min) / (self.age_max - self.age_min)
        else:
            normalized[mask] = 0.5
        return np.clip(normalized, 0.0, 1.0)
    
    def normalize_income(self, income: np.ndarray) -> np.ndarray:
        """归一化收入到[0, 1]"""
        if self.income_min is None or self.income_max is None:
            return income
        income = np.asarray(income, dtype=np.float32)
        # 处理 NaN
        mask = ~np.isnan(income)
        normalized = np.zeros_like(income)
        if self.income_max > self.income_min:
            normalized[mask] = (income[mask] - self.income_min) / (self.income_max - self.income_min)
        else:
            normalized[mask] = 0.5
        return np.clip(normalized, 0.0, 1.0)
    
    def to_dict(self) -> Dict:
        """转换为字典（用于保存到checkpoint）"""
        return {
            'age_min': self.age_min,
            'age_max': self.age_max,
            'income_min': self.income_min,
            'income_max': self.income_max
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'MetadataNormalizer':
        """从字典创建（用于从checkpoint加载）"""
        normalizer = cls()
        normalizer.age_min = data.get('age_min')
        normalizer.age_max = data.get('age_max')
        normalizer.income_min = data.get('income_min')
        normalizer.income_max = data.get('income_max')
        return normalizer


class PersonalityDataset(Dataset):
    """性格预测数据集"""
    
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
        metadata_normalizer: Optional[MetadataNormalizer] = None  # 用于归一化 age 和 income
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
        
        # ========== 加载和编码元数据 ==========
        self.metadata = {}
        
        # 性别 (gender): 1=男性, 2=女性, unknown/NaN=0
        if self.use_gender:
            gender_data = self.data['gender'].copy()
            gender_data = gender_data.replace('unknown', pd.NA)
            gender_data = pd.to_numeric(gender_data, errors='coerce')
            # 将 NaN 设为 0，1->1, 2->2
            gender_data = gender_data.fillna(0).astype(int).clip(0, 2)
            self.metadata['gender'] = gender_data.values
        else:
            self.metadata['gender'] = None
        
        # 教育程度 (education): 2-7, unknown/NaN=0
        if self.use_education:
            education_data = self.data['education'].copy()
            education_data = education_data.replace('unknown', pd.NA)
            education_data = pd.to_numeric(education_data, errors='coerce')
            # 将 NaN 或 <2 的值设为 0，2-7 保持不变
            education_data = education_data.fillna(0)
            education_data = education_data.astype(int).clip(0, 7)
            # 将 1 也设为 0（因为教育程度从2开始）
            education_data[education_data == 1] = 0
            self.metadata['education'] = education_data.values
        else:
            self.metadata['education'] = None
        
        # 种族 (race): 1,2,3,5,6, unknown/NaN=0
        if self.use_race:
            race_data = self.data['race'].copy()
            race_data = race_data.replace('unknown', pd.NA)
            race_data = pd.to_numeric(race_data, errors='coerce')
            # 将 NaN 设为 0，1,2,3,5,6 保持不变，其他值设为0
            race_data = race_data.fillna(0).astype(int)
            # 只保留有效的种族值：1,2,3,5,6，其他设为0
            valid_races = {1, 2, 3, 5, 6}
            race_data = race_data.apply(lambda x: x if x in valid_races else 0)
            race_data = race_data.clip(0, 6)
            self.metadata['race'] = race_data.values
        else:
            self.metadata['race'] = None
        
        # 年龄 (age): 数值，需要归一化
        if self.use_age:
            age_data = self.data['age'].copy()
            age_data = age_data.replace('unknown', pd.NA)
            age_data = pd.to_numeric(age_data, errors='coerce')
            age_array = age_data.values.astype(float)
            
            # 如果提供了 normalizer，进行归一化
            if self.metadata_normalizer is not None:
                age_array = self.metadata_normalizer.normalize_age(age_array)
            else:
                # 如果没有 normalizer，先计算（仅用于训练）
                if self.is_training:
                    valid_age = age_array[~np.isnan(age_array)]
                    if len(valid_age) > 0:
                        age_min, age_max = valid_age.min(), valid_age.max()
                        if age_max > age_min:
                            age_array = np.where(
                                ~np.isnan(age_array),
                                (age_array - age_min) / (age_max - age_min),
                                0.0
                            )
                        else:
                            age_array = np.where(~np.isnan(age_array), 0.5, 0.0)
                    age_array = np.clip(age_array, 0.0, 1.0)
            
            self.metadata['age'] = age_array
        else:
            self.metadata['age'] = None
        
        # 收入 (income): 数值，需要归一化
        if self.use_income:
            income_data = self.data['income'].copy()
            income_data = income_data.replace('unknown', pd.NA)
            income_data = pd.to_numeric(income_data, errors='coerce')
            income_array = income_data.values.astype(float)
            
            # 如果提供了 normalizer，进行归一化
            if self.metadata_normalizer is not None:
                income_array = self.metadata_normalizer.normalize_income(income_array)
            else:
                # 如果没有 normalizer，先计算（仅用于训练）
                if self.is_training:
                    valid_income = income_array[~np.isnan(income_array)]
                    if len(valid_income) > 0:
                        income_min, income_max = valid_income.min(), valid_income.max()
                        if income_max > income_min:
                            income_array = np.where(
                                ~np.isnan(income_array),
                                (income_array - income_min) / (income_max - income_min),
                                0.0
                            )
                        else:
                            income_array = np.where(~np.isnan(income_array), 0.5, 0.0)
                    income_array = np.clip(income_array, 0.0, 1.0)
            
            self.metadata['income'] = income_array
        else:
            self.metadata['income'] = None
    
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
        
        # 添加元数据（如果启用）
        if self.use_gender and self.metadata['gender'] is not None:
            result['gender'] = torch.tensor(self.metadata['gender'][idx], dtype=torch.float32)
        if self.use_education and self.metadata['education'] is not None:
            result['education'] = torch.tensor(self.metadata['education'][idx], dtype=torch.float32)
        if self.use_race and self.metadata['race'] is not None:
            result['race'] = torch.tensor(self.metadata['race'][idx], dtype=torch.float32)
        if self.use_age and self.metadata['age'] is not None:
            result['age'] = torch.tensor(self.metadata['age'][idx], dtype=torch.float32)
        if self.use_income and self.metadata['income'] is not None:
            result['income'] = torch.tensor(self.metadata['income'][idx], dtype=torch.float32)
        
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
    
    # 添加元数据（如果存在）
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
    
    # 如果有标签，也stack起来
    if 'labels' in batch[0]:
        labels = torch.stack([item['labels'] for item in batch])
        result['labels'] = labels
    
    return result

