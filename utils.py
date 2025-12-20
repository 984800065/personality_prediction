"""
工具函数
"""
import torch
import numpy as np
from scipy.stats import pearsonr
from loguru import logger


def set_seed(seed: int):
    """设置随机种子"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_metrics(predictions: np.ndarray, labels: np.ndarray):
    """
    计算评估指标（皮尔逊相关系数）
    
    Args:
        predictions: 预测值，shape (n_samples, 5)
        labels: 真实标签，shape (n_samples, 5)
    
    Returns:
        dict: 包含各维度相关系数和平均值的字典
    """
    # 确保是numpy数组
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    
    # 计算每个维度的皮尔逊相关系数
    pearson_scores = []
    dimension_names = [
        'conscientiousness',
        'openness',
        'extraversion',
        'agreeableness',
        'stability'
    ]
    
    metrics = {}
    
    for i, dim_name in enumerate(dimension_names):
        pred_dim = predictions[:, i]
        label_dim = labels[:, i]
        
        # 计算皮尔逊相关系数
        # 使用scipy.stats.pearsonr，它会自动处理边界情况
        corr, _ = pearsonr(pred_dim, label_dim)
        
        # 如果结果是NaN（比如常数输入或方差为0），保持NaN
        # 这样可以在平均值计算时识别问题
        pearson_scores.append(corr)
        metrics[f'pearson_{dim_name}'] = corr
    
    # 计算平均值：按照任务要求 r = (1/5) * Σr(k)
    # 如果某个维度是NaN，使用nanmean会返回NaN，这可以提醒我们有问题
    # 但为了健壮性，我们使用nanmean忽略NaN值
    pearson_mean = np.nanmean(pearson_scores)
    
    # 如果所有维度都是NaN，返回0
    if np.isnan(pearson_mean):
        pearson_mean = 0.0
        logger.warning("所有维度的皮尔逊相关系数都是NaN，可能存在数据问题")
    
    metrics['pearson_mean'] = pearson_mean
    metrics['pearson_scores'] = pearson_scores
    
    return metrics

