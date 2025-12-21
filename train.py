"""
训练脚本
支持TensorBoard、Checkpoint管理等
"""
import os
import json
import argparse
from typing import Optional, Dict, Tuple, Any
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from tqdm.auto import tqdm
import numpy as np
from pathlib import Path
import shutil

from data_loader import PersonalityDataset, collate_fn
from models.vanilla_model import PersonalityPredictor
from utils import set_seed, compute_metrics
from config import config
from logger_config import setup_logger
from label_normalizer import LabelNormalizer

# 设置logger
logger = setup_logger(
    log_level=config.log_level,
    log_file=config.log_file,
    log_dir=config.log_dir
)


class CheckpointManager:
    """Checkpoint管理器"""
    
    def __init__(self, output_dir: str, max_checkpoints: int = 0):
        """
        Args:
            output_dir: 输出目录
            max_checkpoints: 最多保留的checkpoint数量（0表示不限制）
        """
        self.output_dir = Path(output_dir)
        self.max_checkpoints = max_checkpoints
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_history = []
    
    def save(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        epoch: int,
        val_score: float,
        is_best: bool = False,
        config_dict: Optional[Dict[str, Any]] = None,
        global_step: int = 0,
        normalizer: Optional[LabelNormalizer] = None
    ) -> None:
        """
        保存checkpoint
        
        Args:
            model: 模型
            optimizer: 优化器
            scheduler: 调度器
            epoch: 当前epoch
            val_score: 验证分数
            is_best: 是否为最佳模型
            config_dict: 配置字典
            global_step: 全局步数
        """
        # 准备config字典
        final_config = config_dict.copy() if config_dict else {}
        
        # 添加normalizer信息到config
        if normalizer is not None:
            final_config['normalizer'] = normalizer.to_dict()
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_score': val_score,
            'config': final_config,
            'global_step': global_step
        }
        
        # 保存常规checkpoint
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        self.checkpoint_history.append((epoch, checkpoint_path, val_score))
        
        # 如果是最佳模型，保存为best_model.pt
        if is_best:
            best_path = self.output_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            logger.success(f"保存最佳模型: {best_path} (Pearson: {val_score:.4f})")
        
        # 清理旧checkpoint
        if self.max_checkpoints > 0:
            self._cleanup_old_checkpoints()
    
    def _cleanup_old_checkpoints(self) -> None:
        """清理旧的checkpoint，只保留最新的N个"""
        if len(self.checkpoint_history) <= self.max_checkpoints:
            return
        
        # 按epoch排序，保留最新的
        self.checkpoint_history.sort(key=lambda x: x[0])
        to_remove = self.checkpoint_history[:-self.max_checkpoints]
        
        for epoch, path, _ in to_remove:
            if path.exists():
                path.unlink()
                logger.debug(f"删除旧checkpoint: {path}")
        
        self.checkpoint_history = self.checkpoint_history[-self.max_checkpoints:]


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    device: torch.device,
    epoch: int,
    writer: Optional[SummaryWriter] = None,
    log_interval: int = 10,
    global_step: int = 0,
    scaler: Optional[torch.cuda.amp.GradScaler] = None
) -> Tuple[float, Dict[str, Any], int]:
    """训练一个epoch"""
    model.train()
    total_loss = 0
    all_predictions = []
    all_labels = []
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, batch in enumerate(progress_bar):
        # 移动到设备
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # 清零梯度
        optimizer.zero_grad()
        
        # 前向传播
        if scaler is not None:
            # 混合精度训练
            with torch.amp.autocast(device_type=device.type):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs['loss']
        else:
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs['loss']
        
        logits = outputs['logits']
        
        # 反向传播
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.max_grad_norm)
            optimizer.step()
        
        # 先optimizer.step()，再scheduler.step()（PyTorch 1.1.0+要求）
        scheduler.step()
        
        # 记录
        total_loss += loss.item()
        all_predictions.append(logits.detach().cpu().numpy())
        all_labels.append(labels.detach().cpu().numpy())
        
        # 更新进度条
        progress_bar.set_postfix({'loss': loss.item()})
        
        # 记录到TensorBoard
        if writer and (batch_idx + 1) % log_interval == 0:
            writer.add_scalar('Train/Loss', loss.item(), global_step)
            writer.add_scalar('Train/LearningRate', scheduler.get_last_lr()[0], global_step)
            global_step += 1
    
    # 计算指标
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    metrics = compute_metrics(all_predictions, all_labels)
    
    avg_loss = total_loss / len(dataloader)
    
    # 记录epoch级别的指标到TensorBoard
    if writer:
        writer.add_scalar('Train/EpochLoss', avg_loss, epoch)
        writer.add_scalar('Train/EpochPearson', metrics['pearson_mean'], epoch)
        for i, dim_name in enumerate(['conscientiousness', 'openness', 'extraversion', 'agreeableness', 'stability']):
            writer.add_scalar(f'Train/Pearson_{dim_name}', metrics['pearson_scores'][i], epoch)
    
    return avg_loss, metrics, global_step


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    epoch: int,
    writer: Optional[SummaryWriter] = None
) -> Tuple[float, Dict[str, Any]]:
    """验证"""
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs['loss']
            logits = outputs['logits']
            
            total_loss += loss.item()
            all_predictions.append(logits.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    # 计算指标
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    metrics = compute_metrics(all_predictions, all_labels)
    
    avg_loss = total_loss / len(dataloader)
    
    # 记录到TensorBoard
    if writer:
        writer.add_scalar('Val/EpochLoss', avg_loss, epoch)
        writer.add_scalar('Val/EpochPearson', metrics['pearson_mean'], epoch)
        for i, dim_name in enumerate(['conscientiousness', 'openness', 'extraversion', 'agreeableness', 'stability']):
            writer.add_scalar(f'Val/Pearson_{dim_name}', metrics['pearson_scores'][i], epoch)
    
    return avg_loss, metrics


def main() -> None:
    global config, logger
    
    parser = argparse.ArgumentParser(description='训练性格预测模型')
    parser.add_argument('--config', type=str, default=None,
                       help='.env文件路径（可选，默认使用项目根目录的.env）')
    
    # 允许通过命令行参数覆盖配置
    parser.add_argument('--base_model', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--learning_rate', type=float, default=None)
    parser.add_argument('--num_epochs', type=int, default=None)
    parser.add_argument('--resume_from', type=str, default=None,
                       help='从checkpoint恢复训练，指定checkpoint路径')
    
    args = parser.parse_args()
    
    # 如果指定了配置文件，重新加载
    if args.config:
        from config import Config
        config = Config(args.config)
        logger = setup_logger(
            log_level=config.log_level,
            log_file=config.log_file,
            log_dir=config.log_dir
        )
    
    # 条件导入改进模型（在config加载后）
    USE_IMPROVED_MODEL = False
    ImprovedPersonalityPredictor = None
    if config.use_improved_model:
        try:
            from models.improved_pooling_model import ImprovedPersonalityPredictor
            USE_IMPROVED_MODEL = True
            logger.info("✓ 已启用改进模型架构")
        except ImportError as e:
            logger.warning(f"无法导入改进模型: {e}，使用原始模型")
            USE_IMPROVED_MODEL = False
            
    logger.info(f"使用的模型架构为: {'改进模型' if USE_IMPROVED_MODEL else '原始模型'}")
    
    # 命令行参数覆盖配置
    base_model = args.base_model or config.base_model
    batch_size = args.batch_size or config.batch_size
    learning_rate = args.learning_rate or config.learning_rate
    num_epochs = args.num_epochs or config.num_epochs
    
    # 设置随机种子
    set_seed(config.seed)
    
    # 设备选择
    if torch.cuda.is_available():
        if config.gpu_id is not None:
            device = torch.device(f'cuda:{config.gpu_id}')
            logger.info(f"使用指定GPU: {device}")
        else:
            device = torch.device('cuda')
            logger.info(f"使用默认GPU: {device}")
        # 显示GPU信息
        if config.gpu_id is not None:
            gpu_id = config.gpu_id
        else:
            gpu_id = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(gpu_id)
        gpu_memory = torch.cuda.get_device_properties(gpu_id).total_memory / 1024**3
        logger.info(f"GPU: {gpu_name}, 显存: {gpu_memory:.2f} GB")
    else:
        device = torch.device('cpu')
        logger.info(f"使用设备: {device} (CPU模式)")
    
    # 创建输出目录
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存配置
    config_dict = config.to_dict()
    config_dict.update({
        'base_model': base_model,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'num_epochs': num_epochs,
        'use_improved_model': USE_IMPROVED_MODEL,  # 保存模型类型信息
        'use_improved_pooling': config.use_improved_pooling if USE_IMPROVED_MODEL else False,
        'use_mlp_head': config.use_mlp_head if USE_IMPROVED_MODEL else False,
        'mlp_hidden_size': config.mlp_hidden_size if USE_IMPROVED_MODEL else 256,
    })
    config_path = output_dir / 'config.json'
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)
    logger.info(f"配置已保存到: {config_path}")
    
    # 初始化TensorBoard
    writer = None
    if config.use_tensorboard:
        tensorboard_dir = Path(config.tensorboard_dir)
        tensorboard_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=tensorboard_dir)
        logger.info(f"TensorBoard日志目录: {tensorboard_dir}")
        logger.info(f"启动TensorBoard: tensorboard --logdir {tensorboard_dir}")
    
    # 初始化CheckpointManager
    checkpoint_manager = CheckpointManager(
        output_dir=output_dir,
        max_checkpoints=config.max_checkpoints
    )
    
    # 如果从checkpoint恢复，先检查checkpoint以获取正确的base_model
    checkpoint_config = None
    checkpoint = None
    if args.resume_from:
        checkpoint_path = Path(args.resume_from)
        if not checkpoint_path.exists():
            logger.error(f"Checkpoint文件不存在: {checkpoint_path}")
            raise FileNotFoundError(f"Checkpoint文件不存在: {checkpoint_path}")
        
        logger.info(f"从checkpoint恢复训练: {checkpoint_path}")
        # 先加载checkpoint元数据（不加载到GPU）
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        # 从checkpoint中读取配置
        if 'config' in checkpoint:
            checkpoint_config = checkpoint['config']
            # 如果checkpoint中有base_model，使用checkpoint的base_model
            if 'base_model' in checkpoint_config:
                checkpoint_base_model = checkpoint_config['base_model']
                if checkpoint_base_model != base_model:
                    logger.warning(f"Checkpoint中的base_model ({checkpoint_base_model}) 与当前配置 ({base_model}) 不同，将使用checkpoint的base_model")
                    base_model = checkpoint_base_model
    
    # 加载tokenizer（使用正确的base_model）
    logger.info(f"加载tokenizer: {base_model}")
    # 某些模型（如gte-multilingual）需要trust_remote_code=True
    try:
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    except Exception:
        # 如果失败，尝试不使用trust_remote_code
        tokenizer = AutoTokenizer.from_pretrained(base_model)
    
    # 加载数据集并计算归一化参数
    logger.info("加载数据集...")
    
    # 如果从checkpoint恢复，尝试加载normalizer
    normalizer = None
    if checkpoint_config and 'normalizer' in checkpoint_config:
        logger.info("从checkpoint加载归一化参数...")
        normalizer = LabelNormalizer.from_dict(checkpoint_config['normalizer'])
    else:
        # 先创建一个临时数据集来计算normalizer（不归一化）
        temp_dataset = PersonalityDataset(
            train_tsv_path=config.train_file,
            articles_csv_path=config.articles_file,
            tokenizer=tokenizer,
            max_length=config.max_length,
            is_training=True,
            normalizer=None  # 不归一化，用于计算min/max
        )
        
        # 从训练数据计算normalizer
        logger.info("计算标签归一化参数...")
        normalizer = LabelNormalizer()
        normalizer.fit(temp_dataset.labels)
    
    # 使用normalizer创建实际的数据集（标签会被归一化）
    full_dataset = PersonalityDataset(
        train_tsv_path=config.train_file,
        articles_csv_path=config.articles_file,
        tokenizer=tokenizer,
        max_length=config.max_length,
        is_training=True,
        normalizer=normalizer
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
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    logger.info(f"训练集大小: {len(train_dataset)}, 验证集大小: {len(val_dataset)}")
    
    # 从checkpoint恢复训练状态（checkpoint已在前面加载）
    start_epoch = 1
    best_val_score = -1.0
    best_epoch = -1
    global_step = 0
    
    if checkpoint is not None:
        # checkpoint已在前面加载，这里只需要恢复训练状态
        # checkpoint_config也已在前面设置
        
        # 如果checkpoint中有模型类型信息，使用checkpoint的配置
        if checkpoint_config and 'use_improved_model' in checkpoint_config:
            logger.info("检测到checkpoint中的模型配置，将使用checkpoint的模型类型")
            # 临时更新USE_IMPROVED_MODEL以匹配checkpoint
            if checkpoint_config['use_improved_model'] and not USE_IMPROVED_MODEL:
                try:
                    from models.improved_pooling_model import ImprovedPersonalityPredictor
                    USE_IMPROVED_MODEL = True
                    logger.warning("当前配置使用原始模型，但checkpoint使用改进模型，已切换到改进模型")
                except ImportError:
                    logger.error("checkpoint使用改进模型，但无法导入改进模型类，请确保improved_pooling_model.py存在")
                    raise
            elif not checkpoint_config['use_improved_model'] and USE_IMPROVED_MODEL:
                logger.warning("当前配置使用改进模型，但checkpoint使用原始模型，将使用原始模型")
                USE_IMPROVED_MODEL = False
        
        # 恢复训练状态
        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch'] + 1  # 从下一个epoch开始
            logger.info(f"✓ 从Epoch {start_epoch} 继续训练")
        
        if 'val_score' in checkpoint:
            best_val_score = checkpoint['val_score']
            logger.info(f"✓ 最佳验证分数: {best_val_score:.4f}")
        
        if checkpoint_config and 'best_epoch' in checkpoint_config:
            best_epoch = checkpoint_config.get('best_epoch', -1)
        
        # 恢复global_step（如果存在）
        if 'global_step' in checkpoint:
            global_step = checkpoint['global_step']
            logger.info(f"✓ Global step: {global_step}")
        
        # 重新加载checkpoint到正确的device（用于后续加载模型权重等）
        checkpoint = torch.load(args.resume_from, map_location=device, weights_only=False)
    
    # 创建模型（根据checkpoint配置或当前配置）
    logger.info(f"创建模型: {base_model}")
    if checkpoint_config and 'use_improved_model' in checkpoint_config:
        # 使用checkpoint中的配置创建模型
        if checkpoint_config['use_improved_model'] and USE_IMPROVED_MODEL and ImprovedPersonalityPredictor is not None:
            logger.info("使用改进的模型架构（从checkpoint恢复）")
            model = ImprovedPersonalityPredictor(
                base_model_name=checkpoint_config.get('base_model', base_model),
                num_labels=checkpoint_config.get('num_labels', config.num_labels),
                freeze_base=checkpoint_config.get('freeze_base', config.freeze_base),
                use_improved_pooling=checkpoint_config.get('use_improved_pooling', config.use_improved_pooling),
                use_mlp_head=checkpoint_config.get('use_mlp_head', config.use_mlp_head),
                mlp_hidden_size=checkpoint_config.get('mlp_hidden_size', config.mlp_hidden_size)
            )
        else:
            logger.info("使用原始模型架构（从checkpoint恢复）")
            model = PersonalityPredictor(
                base_model_name=checkpoint_config.get('base_model', base_model),
                num_labels=checkpoint_config.get('num_labels', config.num_labels),
                freeze_base=checkpoint_config.get('freeze_base', config.freeze_base)
            )
    elif USE_IMPROVED_MODEL and ImprovedPersonalityPredictor is not None:
        logger.info("使用改进的模型架构")
        model = ImprovedPersonalityPredictor(
            base_model_name=base_model,
            num_labels=config.num_labels,
            freeze_base=config.freeze_base,
            use_improved_pooling=config.use_improved_pooling,
            use_mlp_head=config.use_mlp_head,
            mlp_hidden_size=config.mlp_hidden_size
        )
    else:
        model = PersonalityPredictor(
            base_model_name=base_model,
            num_labels=config.num_labels,
            freeze_base=config.freeze_base
        )
    model.to(device)
    
    # 如果从checkpoint恢复，加载模型权重
    if checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info("✓ 模型权重已加载")
    
    # 启用梯度检查点（节省显存）
    if config.gradient_checkpointing and hasattr(model.base_model, 'gradient_checkpointing_enable'):
        model.base_model.gradient_checkpointing_enable()
        logger.info("✓ 已启用梯度检查点（节省显存）")
    
    # 优化器和调度器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=config.weight_decay
    )
    
    # 计算总步数
    total_steps = len(train_loader) * num_epochs
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=total_steps
    )
    
    # 混合精度训练（FP16）
    scaler = None
    if config.fp16 and device.type == 'cuda':
        scaler = torch.amp.GradScaler(device=device)
        logger.info("✓ 已启用混合精度训练（FP16，节省显存）")
    
    # 如果从checkpoint恢复，加载优化器和调度器状态
    if checkpoint:
        # 加载优化器状态
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            logger.info("✓ 优化器状态已加载")
        
        # 加载调度器状态
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            logger.info("✓ 调度器状态已加载")
        
        logger.success(f"成功从checkpoint恢复，将从Epoch {start_epoch} 继续训练")
    
    # 训练循环
    logger.info("开始训练...")
    for epoch in range(start_epoch, num_epochs + 1):
        logger.info(f"{'='*50}")
        logger.info(f"Epoch {epoch}/{num_epochs}")
        logger.info(f"{'='*50}")
        
        # 训练
        train_loss, train_metrics, global_step = train_epoch(
            model, train_loader, optimizer, scheduler, device, epoch,
            writer=writer, log_interval=config.log_interval, global_step=global_step,
            scaler=scaler
        )
        
        logger.info(f"训练 - Loss: {train_loss:.4f}, "
                    f"Pearson: {train_metrics['pearson_mean']:.4f}")
        
        # 验证
        val_loss, val_metrics = validate(model, val_loader, device, epoch, writer=writer)
        
        logger.info(f"验证 - Loss: {val_loss:.4f}, "
                    f"Pearson: {val_metrics['pearson_mean']:.4f}")
        
        # 判断是否为最佳模型
        is_best = val_metrics['pearson_mean'] > best_val_score
        if is_best:
            best_val_score = val_metrics['pearson_mean']
            best_epoch = epoch
        
        # 保存checkpoint
        should_save = (config.save_every_n_epochs == 0) or (epoch % config.save_every_n_epochs == 0)
        if should_save and (not config.save_best_only or is_best):
            checkpoint_manager.save(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                val_score=val_metrics['pearson_mean'],
                is_best=is_best,
                config_dict=config_dict,
                global_step=global_step,
                normalizer=normalizer
            )
    
    # 关闭TensorBoard writer
    if writer:
        writer.close()
    
    logger.info(f"{'='*50}")
    logger.success(f"训练完成！")
    logger.info(f"最佳验证Pearson相关系数: {best_val_score:.4f} (Epoch {best_epoch})")
    logger.info(f"{'='*50}")


if __name__ == '__main__':
    main()
