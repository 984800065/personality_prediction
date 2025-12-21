"""
测试和预测脚本
"""
import os
import argparse
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from tqdm.auto import tqdm

from data_loader import PersonalityDataset, collate_fn
from utils import set_seed
from config import config
from logger_config import setup_logger


# 设置logger
logger = setup_logger(
    log_level=config.log_level,
    log_file=config.log_file,
    log_dir=config.log_dir
)


def personality_detection(tokenizer: BertTokenizer, model: BertForSequenceClassification, model_input: str, device: torch.device) -> dict:
    '''
    Performs personality prediction on the given input text

    Args: 
        model_input (str): The text conversation 

    Returns:
        dict: A dictionary where keys are speaker labels and values are their personality predictions
    '''

    if len(model_input) == 0:
        ret = {
            "personality_extraversion": float(0),
            "personality_neuroticism": float(0),
            "personality_agreeableness": float(0),
            "personality_conscientiousness": float(0),
            "personality_openness": float(0),
        }
        return ret
    else:
        dict_custom = {}
        preprocess_part1 = model_input[:len(model_input)]
        dict1 = tokenizer.encode_plus(preprocess_part1, max_length=1024, padding=True, truncation=True)
        dict_custom['input_ids'] = [dict1['input_ids'], dict1['input_ids']]
        dict_custom['token_type_ids'] = [dict1['token_type_ids'], dict1['token_type_ids']]
        dict_custom['attention_mask'] = [dict1['attention_mask'], dict1['attention_mask']]
        input_ids: torch.Tensor = torch.tensor(dict_custom['input_ids']).to(device)
        token_type_ids: torch.Tensor = None
        attention_mask: torch.Tensor = torch.tensor(dict_custom['attention_mask']).to(device)
        outs = model(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        b_logit_pred = outs[0]
        pred_label = torch.sigmoid(b_logit_pred)
        ret = {
            "personality_extraversion": float(pred_label[0][0]),
            "personality_neuroticism": float(pred_label[0][1]),
            "personality_agreeableness": float(pred_label[0][2]),
            "personality_conscientiousness": float(pred_label[0][3]),
            "personality_openness": float(pred_label[0][4]),
        }

        return ret


def predict(tokenizer: BertTokenizer, model: BertForSequenceClassification, dataset: PersonalityDataset, device: torch.device):
    """进行预测"""
    model.eval()

    result: dict[str, list[float]] = {
        "personality_extraversion": [],
        "personality_neuroticism": [],
        "personality_agreeableness": [],
        "personality_conscientiousness": [],
        "personality_openness": [],
    }

    with torch.no_grad():
        for item in tqdm(dataset, desc="预测中"):
            texts: list[str] = item['comment']
            personality_prediction: dict[str, float] = personality_detection(tokenizer, model, texts, device)
            for key, value in personality_prediction.items():
                result[key].append(value)
    
    for key in result.keys():
        result[key] = np.array(result[key], dtype=np.float32)
    result["personality_stability"] = 1 - result["personality_neuroticism"]
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_file', type=str,
                       default='datasets/news_personality/test.tsv')
    parser.add_argument('--articles_file', type=str,
                       default='datasets/news_personality/articles.csv')
    parser.add_argument('--output_file', type=str,
                       default='predictions.tsv')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 设备（使用全局config模块）
    device = torch.device(f'cuda:{config.gpu_id}')
    logger.info(f"使用设备: {device}")
    
    # 加载tokenizer
    logger.info("加载tokenizer...")
    tokenizer = BertTokenizer.from_pretrained("Minej/bert-base-personality", local_files_only=True)
    # 加载模型
    logger.info("加载模型...")
    model = BertForSequenceClassification.from_pretrained("Minej/bert-base-personality", num_labels=5, local_files_only=True)
    model.config.label2id = {
        "personality_extraversion": 0,
        "personality_neuroticism": 1,
        "personality_agreeableness": 2,
        "personality_conscientiousness": 3,
        "personality_openness": 4,
    }
    model.config.id2label = {
        "0": "personality_extraversion",
        "1": "personality_neuroticism",
        "2": "personality_agreeableness",
        "3": "personality_conscientiousness",
        "4": "personality_openness",
    }
    model.to(device)

    # 加载测试集
    logger.info("加载测试集...")
    test_dataset = PersonalityDataset(
        train_tsv_path=args.test_file,
        articles_csv_path=args.articles_file,
        tokenizer=tokenizer,
        max_length=args.max_length,
        is_training=False
    )
    
    logger.info(f"测试集大小: {len(test_dataset)}")
    
    # 进行预测
    logger.info("开始预测...")
    result: dict[str, np.ndarray] = predict(tokenizer, model, test_dataset, device)
    
    # 保存结果
    logger.info(f"保存预测结果到: {args.output_file}")
    
    # 指定列顺序
    columns_order = [
        'personality_conscientiousness',
        'personality_openness', 
        'personality_extraversion',
        'personality_agreeableness',
        'personality_stability'
    ]

    # 按顺序从字典中提取数据
    df = pd.DataFrame({
        col: result[col] for col in columns_order
    })
    
    # 保存为TSV文件（使用制表符分隔）
    # 第一行是列名称，第二行开始是预测值
    df.to_csv(
        args.output_file,
        sep='\t',
        index=False,
        float_format='%.6f'
    )
    
    logger.success(f"预测完成！共 {len(result['personality_extraversion'])} 个样本")


if __name__ == '__main__':
    main()

