"""
测试和预测脚本
"""
import os
import argparse
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm.auto import tqdm

from data_loader import PersonalityDataset, collate_fn
from model import PersonalityPredictor
from utils import set_seed
from config import config
from logger_config import setup_logger

# 条件导入改进模型
USE_IMPROVED_MODEL = False
ImprovedPersonalityPredictor = None
if config.use_improved_model:
    try:
        from model_improved import ImprovedPersonalityPredictor
        USE_IMPROVED_MODEL = True
    except ImportError:
        USE_IMPROVED_MODEL = False

# 设置logger
logger = setup_logger(
    log_level=config.log_level,
    log_file=config.log_file,
    log_dir=config.log_dir
)


def predict(model, dataloader, device):
    """进行预测"""
    model.eval()
    all_predictions = []
    all_comment_ids = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="预测中"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            logits = outputs['logits']
            all_predictions.append(logits.cpu().numpy())
            all_comment_ids.extend(batch['comment_ids'])
    
    all_predictions = np.concatenate(all_predictions, axis=0)
    
    return all_predictions, all_comment_ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='模型检查点路径')
    parser.add_argument('--test_file', type=str,
                       default='datasets/news_personality/test.tsv')
    parser.add_argument('--articles_file', type=str,
                       default='datasets/news_personality/articles.csv')
    parser.add_argument('--output_file', type=str,
                       default='predictions.tsv')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda:2' if torch.cuda.is_available() else 'cpu')
    
    args = parser.parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 设备
    device = torch.device(args.device)
    logger.info(f"使用设备: {device}")
    
    # 加载检查点
    logger.info(f"加载模型: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    
    # 获取配置
    if 'config' in checkpoint:
        config = checkpoint['config']
        base_model = config.get('base_model', 'Alibaba-NLP/gte-multilingual-base')
    else:
        # 如果没有配置，使用默认模型
        base_model = 'Alibaba-NLP/gte-multilingual-base'
        logger.warning("检查点中没有配置信息，使用默认模型: Alibaba-NLP/gte-multilingual-base")
    
    # 加载tokenizer
    logger.info(f"加载tokenizer: {base_model}")
    # 某些模型（如gte-multilingual）需要trust_remote_code=True
    try:
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    except Exception:
        # 如果失败，尝试不使用trust_remote_code
        tokenizer = AutoTokenizer.from_pretrained(base_model)
    
    # 创建模型
    logger.info("创建模型...")
    # 检查checkpoint中是否包含改进模型的配置
    is_improved_model = False
    if 'config' in checkpoint:
        checkpoint_config = checkpoint.get('config', {})
        is_improved_model = checkpoint_config.get('use_improved_model', False)
    
    if is_improved_model and USE_IMPROVED_MODEL and ImprovedPersonalityPredictor is not None:
        logger.info("使用改进的模型架构")
        model = ImprovedPersonalityPredictor(
            base_model_name=base_model,
            num_labels=5,
            use_improved_pooling=checkpoint_config.get('use_improved_pooling', True),
            use_mlp_head=checkpoint_config.get('use_mlp_head', True),
            mlp_hidden_size=checkpoint_config.get('mlp_hidden_size', 256)
        )
    else:
        model = PersonalityPredictor(
            base_model_name=base_model,
            num_labels=5
        )
    
    # 加载权重
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # 加载测试集
    logger.info("加载测试集...")
    test_dataset = PersonalityDataset(
        train_tsv_path=args.test_file,
        articles_csv_path=args.articles_file,
        tokenizer=tokenizer,
        max_length=args.max_length,
        is_training=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    logger.info(f"测试集大小: {len(test_dataset)}")
    
    # 进行预测
    logger.info("开始预测...")
    predictions, comment_ids = predict(model, test_loader, device)
    
    # 保存结果
    logger.info(f"保存预测结果到: {args.output_file}")
    
    # 创建DataFrame（只包含五列，不包含comment_id）
    df = pd.DataFrame(
        predictions,
        columns=[
            'personality_conscientiousness',
            'personality_openess',
            'personality_extraversion',
            'personality_agreeableness',
            'personality_stability'
        ]
    )
    
    # 保存为TSV文件（使用制表符分隔）
    # 第一行是列名称，第二行开始是预测值
    df.to_csv(
        args.output_file,
        sep='\t',
        index=False,
        float_format='%.6f'
    )
    
    logger.success(f"预测完成！共 {len(predictions)} 个样本")
    logger.info(f"预测值范围: [{predictions.min():.2f}, {predictions.max():.2f}]")
    logger.info(f"预测值均值: {predictions.mean(axis=0)}")


if __name__ == '__main__':
    main()

