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
    
    # ========== 模型改进配置 ==========
    @property
    def use_improved_model(self) -> bool:
        """是否使用改进的模型架构"""
        return os.getenv("USE_IMPROVED_MODEL", "False").lower() == "true"
    
    @property
    def use_improved_pooling(self) -> bool:
        """是否使用改进的池化（mean + max + cls）"""
        return os.getenv("USE_IMPROVED_POOLING", "True").lower() == "true"
    
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
        return os.getenv("AGGREGATION_METHOD", "attention")
    
    @property
    def aggregation_hidden_size(self) -> int:
        """聚合层的隐藏层大小"""
        return int(os.getenv("AGGREGATION_HIDDEN_SIZE", "256"))
    
    # ========== Late Fusion 元数据配置 ==========
    @property
    def use_gender(self) -> bool:
        """是否使用性别特征"""
        return os.getenv("USE_GENDER", "True").lower() == "true"
    
    @property
    def use_education(self) -> bool:
        """是否使用教育程度特征"""
        return os.getenv("USE_EDUCATION", "True").lower() == "true"
    
    @property
    def use_race(self) -> bool:
        """是否使用种族特征"""
        return os.getenv("USE_RACE", "True").lower() == "true"
    
    @property
    def use_age(self) -> bool:
        """是否使用年龄特征"""
        return os.getenv("USE_AGE", "True").lower() == "true"
    
    @property
    def use_income(self) -> bool:
        """是否使用收入特征"""
        return os.getenv("USE_INCOME", "True").lower() == "true"
    
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
            "use_tensorboard": self.use_tensorboard,
            "log_interval": self.log_interval,
            "log_level": self.log_level,
            "log_file": self.log_file,
            "use_gender": self.use_gender,
            "use_education": self.use_education,
            "use_race": self.use_race,
            "use_age": self.use_age,
            "use_income": self.use_income,
            "use_multi_instance": True,  # 统一使用多实例学习
            "aggregation_method": self.aggregation_method,
            "aggregation_hidden_size": self.aggregation_hidden_size,
        }


# 全局配置实例
config = Config()

