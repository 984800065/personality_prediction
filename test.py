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

from data_loader_multi_instance import MultiInstancePersonalityDataset, collate_fn_multi_instance
from data_loader import MetadataNormalizer
from utils import set_seed
from config import config
from logger_config import setup_logger
from label_normalizer import LabelNormalizer
from train import validate

# 设置logger
logger = setup_logger(
    log_level=config.log_level,
    log_file=config.log_file,
    log_dir=config.log_dir
)

# 导入多实例学习模型
try:
    from models.multi_instance_late_fusion_model import MultiInstanceLateFusionPersonalityPredictor
except ImportError as e:
    logger.error(f"无法导入 Multi-Instance Late Fusion 模型: {e}")
    raise


def predict(model, dataloader, device):
    """进行预测"""
    model.eval()
    all_predictions = []
    all_speaker_ids = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="预测中"):
            # 多实例模式
            input_ids_list = batch['input_ids_list']
            attention_mask_list = batch['attention_mask_list']
            comment_mask = batch.get('comment_mask', None)
            
            # 将每个 speaker 的评论列表移动到设备并堆叠成张量
            processed_input_ids_list = []
            processed_attention_mask_list = []
            for ids_list, mask_list in zip(input_ids_list, attention_mask_list):
                # 堆叠每个 speaker 的所有评论
                ids_tensor = torch.stack([ids.to(device) for ids in ids_list])  # [num_comments, max_length]
                mask_tensor = torch.stack([mask.to(device) for mask in mask_list])  # [num_comments, max_length]
                processed_input_ids_list.append(ids_tensor)
                processed_attention_mask_list.append(mask_tensor)
            
            if comment_mask is not None:
                comment_mask = comment_mask.to(device)
            
            model_kwargs = {
                'input_ids_list': processed_input_ids_list,
                'attention_mask_list': processed_attention_mask_list,
                'comment_mask': comment_mask
            }
            
            # 准备元数据（如果存在）
            if 'gender' in batch:
                model_kwargs['gender'] = batch['gender'].to(device)
            if 'education' in batch:
                model_kwargs['education'] = batch['education'].to(device)
            if 'race' in batch:
                model_kwargs['race'] = batch['race'].to(device)
            if 'age' in batch:
                model_kwargs['age'] = batch['age'].to(device)
            if 'income' in batch:
                model_kwargs['income'] = batch['income'].to(device)
            
            outputs = model(**model_kwargs)
            
            logits = outputs['logits']
            all_predictions.append(logits.cpu().numpy())
            all_speaker_ids.extend(batch.get('speaker_ids', []))
    
    all_predictions = np.concatenate(all_predictions, axis=0)
    
    return all_predictions, all_speaker_ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='checkpoints/roberta_base/best_model.pt', 
                       help='模型检查点路径', )
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
    
    # 设备（使用全局config模块）
    device = torch.device(f'cuda:{config.gpu_id}')
    logger.info(f"使用设备: {device}")
    
    # 加载检查点
    logger.info(f"加载模型: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    
    # 获取配置（使用checkpoint_config避免覆盖全局config）
    checkpoint_config = None
    normalizer = None
    metadata_normalizer = None
    if 'config' in checkpoint:
        checkpoint_config = checkpoint['config']
        base_model = checkpoint_config.get('base_model', 'roberta-base')
        
        # 加载normalizer（如果存在）
        if 'normalizer' in checkpoint_config:
            logger.info("从checkpoint加载标签归一化参数...")
            normalizer = LabelNormalizer.from_dict(checkpoint_config['normalizer'])
        else:
            logger.warning("Checkpoint中没有标签归一化参数，预测结果将不会被反归一化")
        
        # 加载metadata_normalizer（如果存在）
        if 'metadata_normalizer' in checkpoint_config:
            logger.info("从checkpoint加载元数据归一化参数...")
            metadata_normalizer = MetadataNormalizer.from_dict(checkpoint_config['metadata_normalizer'])
    else:
        # 如果没有配置，使用默认模型
        base_model = 'roberta-base'
        logger.warning("检查点中没有配置信息，使用默认模型: roberta-base")
    
    # 使用checkpoint中的seed（如果存在），确保与训练时一致
    # 必须在数据集划分之前设置seed，确保验证集划分一致
    train_seed = checkpoint_config.get('seed', config.seed) if checkpoint_config else config.seed
    set_seed(train_seed)
    logger.info(f"使用训练时的随机种子: {train_seed} (确保验证集划分一致)")
    
    # 加载tokenizer
    logger.info(f"加载tokenizer: {base_model}")
    # 某些模型（如gte-multilingual）需要trust_remote_code=True
    local_files_only = config.local_files_only
    if local_files_only:
        logger.info("✓ 仅使用本地模型文件，不连接 Hugging Face")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            base_model, 
            trust_remote_code=True,
            local_files_only=local_files_only
        )
    except Exception:
        # 如果失败，尝试不使用trust_remote_code
        tokenizer = AutoTokenizer.from_pretrained(
            base_model,
            local_files_only=local_files_only
        )
    
    # 创建模型（统一使用多实例模型）
    logger.info("创建模型...")
    
    if MultiInstanceLateFusionPersonalityPredictor is None:
        logger.error("Multi-Instance Late Fusion 模型未导入，无法创建模型")
        raise RuntimeError("Multi-Instance Late Fusion 模型未导入")
    
    logger.info("使用 Multi-Instance Late Fusion 模型架构")
    model = MultiInstanceLateFusionPersonalityPredictor(
        base_model_name=base_model,
        num_labels=5,
        use_improved_pooling=checkpoint_config.get('use_improved_pooling', config.use_improved_pooling) if checkpoint_config else config.use_improved_pooling,
        use_mlp_head=checkpoint_config.get('use_mlp_head', config.use_mlp_head) if checkpoint_config else config.use_mlp_head,
        mlp_hidden_size=checkpoint_config.get('mlp_hidden_size', config.mlp_hidden_size) if checkpoint_config else config.mlp_hidden_size,
        local_files_only=config.local_files_only,
        use_gender=checkpoint_config.get('use_gender', config.use_gender) if checkpoint_config else config.use_gender,
        use_education=checkpoint_config.get('use_education', config.use_education) if checkpoint_config else config.use_education,
        use_race=checkpoint_config.get('use_race', config.use_race) if checkpoint_config else config.use_race,
        use_age=checkpoint_config.get('use_age', config.use_age) if checkpoint_config else config.use_age,
        use_income=checkpoint_config.get('use_income', config.use_income) if checkpoint_config else config.use_income,
        aggregation_method=checkpoint_config.get('aggregation_method', config.aggregation_method) if checkpoint_config else config.aggregation_method,
        aggregation_hidden_size=checkpoint_config.get('aggregation_hidden_size', config.aggregation_hidden_size) if checkpoint_config else config.aggregation_hidden_size
    )
    
    # 加载权重
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # 获取元数据配置（从checkpoint或当前config）
    use_gender = checkpoint_config.get('use_gender', config.use_gender) if checkpoint_config else config.use_gender
    use_education = checkpoint_config.get('use_education', config.use_education) if checkpoint_config else config.use_education
    use_race = checkpoint_config.get('use_race', config.use_race) if checkpoint_config else config.use_race
    use_age = checkpoint_config.get('use_age', config.use_age) if checkpoint_config else config.use_age
    use_income = checkpoint_config.get('use_income', config.use_income) if checkpoint_config else config.use_income
    
    # 使用多实例学习模式
    full_dataset = MultiInstancePersonalityDataset(
        train_tsv_path=config.train_file,
        articles_csv_path=config.articles_file,
        tokenizer=tokenizer,
        max_length=config.max_length,
        is_training=True,
        normalizer=normalizer,
        use_gender=use_gender,
        use_education=use_education,
        use_race=use_race,
        use_age=use_age,
        use_income=use_income,
        metadata_normalizer=metadata_normalizer
    )
    
    # 划分训练集和验证集
    val_size = int(len(full_dataset) * config.val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    # 创建DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn_multi_instance
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn_multi_instance
    )

    train_loss, train_metrics = validate(model, train_loader, device, 0, writer=None)
    logger.info(f"训练 - Loss: {train_loss:.4f}, "f"Pearson: {train_metrics['pearson_mean']:.4f}")

    val_loss, val_metrics = validate(model, val_loader, device, 0, writer=None)
    logger.info(f"验证 - Loss: {val_loss:.4f}, "f"Pearson: {val_metrics['pearson_mean']:.4f}")

    # 加载测试集
    logger.info("加载测试集...")
    test_dataset = MultiInstancePersonalityDataset(
        train_tsv_path=args.test_file,
        articles_csv_path=args.articles_file,
        tokenizer=tokenizer,
        max_length=args.max_length,
        is_training=False,
        use_gender=use_gender,
        use_education=use_education,
        use_race=use_race,
        use_age=use_age,
        use_income=use_income,
        metadata_normalizer=metadata_normalizer
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn_multi_instance
    )
    
    logger.info(f"测试集大小: {len(test_dataset)}")
    
    # 进行预测
    logger.info("开始预测...")
    predictions, speaker_ids = predict(model, test_loader, device)
    
    # 如果normalizer存在，反归一化预测结果
    if normalizer is not None:
        logger.info("反归一化预测结果到原始范围...")
        predictions = normalizer.denormalize(predictions)
    else:
        logger.warning("没有normalizer，预测结果保持原样（可能在[0, 1]范围）")
    
    # 将 speaker_id 的预测结果映射回所有 comment_id
    # 读取测试数据，建立 speaker_id -> comment_ids 的映射
    logger.info("将 speaker 预测结果映射回所有 comment...")
    test_df = pd.read_csv(args.test_file, sep='\t')
    
    # 创建 speaker_id -> prediction 的映射
    speaker_to_prediction = {}
    for speaker_id, pred in zip(speaker_ids, predictions):
        speaker_to_prediction[speaker_id] = pred
    
    # 为每个 comment 分配对应的预测结果
    comment_predictions = []
    for _, row in test_df.iterrows():
        speaker_id = row['speaker_id']
        if speaker_id in speaker_to_prediction:
            comment_predictions.append(speaker_to_prediction[speaker_id])
        else:
            # 如果找不到对应的 speaker_id，使用零向量（不应该发生）
            logger.warning(f"找不到 speaker_id {speaker_id} 的预测结果，使用零向量")
            comment_predictions.append(np.zeros(5))
    
    comment_predictions = np.array(comment_predictions)
    
    # 保存结果
    logger.info(f"保存预测结果到: {args.output_file}")
    
    # 创建DataFrame（只包含五列，不包含comment_id）
    df = pd.DataFrame(
        comment_predictions,
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
    
    logger.success(f"预测完成！共 {len(comment_predictions)} 个样本")
    logger.info(f"预测值范围: [{comment_predictions.min():.2f}, {comment_predictions.max():.2f}]")
    logger.info(f"预测值均值: {comment_predictions.mean(axis=0)}")


if __name__ == '__main__':
    main()

