"""
配置管理模块
从.env文件加载配置，并提供类型安全的配置访问
"""
import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv


class Config:
    """配置类，管理所有配置参数"""
    
    def __init__(self, env_file: str = ".env"):
        """
        初始化配置
        
        Args:
            env_file: .env文件路径
        """
        # 加载.env文件
        env_path = Path(env_file)
        if env_path.exists():
            load_dotenv(env_path, override=True)
        else:
            print(f"警告: {env_file} 文件不存在，使用默认值或环境变量")
    
    # ========== 数据路径配置 ==========
    @property
    def train_file(self) -> str:
        return os.getenv("TRAIN_FILE", "datasets/news_personality/train.tsv")
    
    @property
    def test_file(self) -> str:
        return os.getenv("TEST_FILE", "datasets/news_personality/test.tsv")
    
    @property
    def articles_file(self) -> str:
        return os.getenv("ARTICLES_FILE", "datasets/news_personality/articles.csv")
    
    @property
    def include_article(self) -> bool:
        """是否在tokenizer输入中包含新闻文本（默认False，只使用评论）"""
        return os.getenv("INCLUDE_ARTICLE", "False").lower() == "true"
    
    # ========== 模型配置 ==========
    @property
    def base_model(self) -> str:
        return os.getenv("BASE_MODEL", "roberta-base")
    
    @property
    def num_labels(self) -> int:
        return int(os.getenv("NUM_LABELS", "5"))
    
    @property
    def freeze_base(self) -> bool:
        return os.getenv("FREEZE_BASE", "False").lower() == "true"
    
    @property
    def local_files_only(self) -> bool:
        """是否只使用本地模型文件，不连接 Hugging Face"""
        return os.getenv("LOCAL_FILES_ONLY", "False").lower() == "true"
    
    @property
    def use_metadata(self) -> bool:
        """是否使用metadata（控制模型分类头上是否有metadata的维度拼接），默认True"""
        return os.getenv("USE_METADATA", "True").lower() == "true"
    
    # ========== 模型模式配置 ==========
    @property
    def use_multi_instance(self) -> bool:
        """是否使用多实例学习模式（聚合模式），False表示使用原始模型模式"""
        return os.getenv("USE_MULTI_INSTANCE", "False").lower() == "true"
    
    @property
    def use_mlp_head(self) -> bool:
        """是否使用MLP回归头"""
        return os.getenv("USE_MLP_HEAD", "True").lower() == "true"
    
    @property
    def mlp_hidden_size(self) -> int:
        """MLP隐藏层大小"""
        return int(os.getenv("MLP_HIDDEN_SIZE", "256"))
    
    # ========== 多实例学习配置 ==========
    @property
    def aggregation_method(self) -> str:
        """多实例聚合方法: 'attention', 'mean', 'max', 'transformer'"""
        return os.getenv("AGGREGATION_METHOD", "mean")
    
    @property
    def aggregation_hidden_size(self) -> int:
        """聚合层的隐藏层大小"""
        return int(os.getenv("AGGREGATION_HIDDEN_SIZE", "256"))
    
    # ========== Late Fusion 元数据配置 ==========
    # 注意：元数据默认全部使用，不再提供单独的控制选项
    
    # ========== GPU和显存配置 ==========
    @property
    def gpu_id(self) -> Optional[int]:
        """指定使用的GPU ID（None表示自动选择）"""
        gpu_str = os.getenv("GPU_ID", "")
        if gpu_str == "" or gpu_str.lower() == "none":
            return None
        return int(gpu_str)
    
    @property
    def gradient_checkpointing(self) -> bool:
        """是否使用梯度检查点（节省显存但增加计算时间）"""
        return os.getenv("GRADIENT_CHECKPOINTING", "False").lower() == "true"
    
    @property
    def fp16(self) -> bool:
        """是否使用混合精度训练（FP16，节省显存）"""
        return os.getenv("FP16", "False").lower() == "true"
    
    # ========== 训练配置 ==========
    @property
    def batch_size(self) -> int:
        return int(os.getenv("BATCH_SIZE", "16"))
    
    @property
    def learning_rate(self) -> float:
        return float(os.getenv("LEARNING_RATE", "2e-5"))
    
    @property
    def num_epochs(self) -> int:
        return int(os.getenv("NUM_EPOCHS", "5"))
    
    @property
    def max_length(self) -> int:
        return int(os.getenv("MAX_LENGTH", "512"))
    
    @property
    def warmup_steps(self) -> int:
        return int(os.getenv("WARMUP_STEPS", "500"))
    
    @property
    def val_split(self) -> float:
        return float(os.getenv("VAL_SPLIT", "0.1"))
    
    @property
    def seed(self) -> int:
        """获取随机种子，如果 SEED == -1 则返回 -1（由调用方处理）"""
        return int(os.getenv("SEED", "42"))
    
    @property
    def weight_decay(self) -> float:
        return float(os.getenv("WEIGHT_DECAY", "0.01"))
    
    @property
    def max_grad_norm(self) -> float:
        return float(os.getenv("MAX_GRAD_NORM", "1.0"))
    
    # ========== 输出配置 ==========
    @property
    def output_dir(self) -> str:
        return os.getenv("OUTPUT_DIR", "./checkpoints")
    
    @property
    def log_dir(self) -> str:
        return os.getenv("LOG_DIR", "logs")
    
    @property
    def tensorboard_dir(self) -> str:
        return os.getenv("TENSORBOARD_DIR", "./runs")
    
    @property
    def experiment_name(self) -> str:
        """实验名称，用于区分不同实验的 TensorBoard 日志"""
        return os.getenv("EXPERIMENT_NAME", "default")
    
    # ========== Checkpoint配置 ==========
    @property
    def save_every_n_epochs(self) -> int:
        """每N个epoch保存一次checkpoint"""
        return int(os.getenv("SAVE_EVERY_N_EPOCHS", "1"))
    
    @property
    def save_best_only(self) -> bool:
        """是否只保存最佳模型"""
        return os.getenv("SAVE_BEST_ONLY", "False").lower() == "true"
    
    @property
    def max_checkpoints(self) -> int:
        """最多保留的checkpoint数量（0表示不限制）"""
        return int(os.getenv("MAX_CHECKPOINTS", "0"))
    
    @property
    def save_checkpoint_from_epoch(self) -> int:
        """
        从第几个epoch开始保存checkpoint（从末尾计算）
        例如：如果总epochs=100，save_checkpoint_from_epoch=10，则从第91个epoch开始保存
        如果为-1，则全程都保存checkpoint（原有逻辑）
        默认值：10
        """
        return int(os.getenv("SAVE_CHECKPOINT_FROM_EPOCH", "10"))
    
    # ========== TensorBoard配置 ==========
    @property
    def use_tensorboard(self) -> bool:
        return os.getenv("USE_TENSORBOARD", "True").lower() == "true"
    
    @property
    def log_interval(self) -> int:
        """每N个batch记录一次到tensorboard"""
        return int(os.getenv("LOG_INTERVAL", "10"))
    
    # ========== 日志配置 ==========
    @property
    def log_level(self) -> str:
        return os.getenv("LOG_LEVEL", "INFO")
    
    @property
    def log_file(self) -> Optional[str]:
        return os.getenv("LOG_FILE", None)
    
    def to_dict(self) -> dict:
        """将配置转换为字典"""
        return {
            "train_file": self.train_file,
            "test_file": self.test_file,
            "articles_file": self.articles_file,
            "base_model": self.base_model,
            "num_labels": self.num_labels,
            "freeze_base": self.freeze_base,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "num_epochs": self.num_epochs,
            "max_length": self.max_length,
            "warmup_steps": self.warmup_steps,
            "val_split": self.val_split,
            "seed": self.seed,
            "weight_decay": self.weight_decay,
            "max_grad_norm": self.max_grad_norm,
            "output_dir": self.output_dir,
            "log_dir": self.log_dir,
            "tensorboard_dir": self.tensorboard_dir,
            "experiment_name": self.experiment_name,
            "save_every_n_epochs": self.save_every_n_epochs,
            "save_best_only": self.save_best_only,
            "max_checkpoints": self.max_checkpoints,
            "save_checkpoint_from_epoch": self.save_checkpoint_from_epoch,
            "use_tensorboard": self.use_tensorboard,
            "log_interval": self.log_interval,
            "log_level": self.log_level,
            "log_file": self.log_file,
            "use_multi_instance": self.use_multi_instance,
            "use_mlp_head": self.use_mlp_head,
            "mlp_hidden_size": self.mlp_hidden_size,
            "aggregation_method": self.aggregation_method if self.use_multi_instance else None,
            "aggregation_hidden_size": self.aggregation_hidden_size if self.use_multi_instance else None,
            "use_metadata": self.use_metadata,
            "include_article": self.include_article,
        }


# 全局配置实例
config = Config()

