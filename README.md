# 性格预测 Pipeline

基于Transformer的模块化性格预测系统，支持即插即用的基座模型。

## 功能特点

- **模块化设计**: 基座模型可以轻松切换（BERT, RoBERTa, DeBERTa等）
- **多任务学习**: 同时预测五个大五人格维度
- **特征融合**: 融合新闻文本和用户评论的特征
- **完整流程**: 包含训练、验证、测试和评估
- **TensorBoard支持**: 实时可视化训练过程
- **Checkpoint管理**: 自动保存和清理checkpoint
- **环境配置**: 使用.env文件管理配置参数
- **日志系统**: 统一的日志管理，支持文件和控制台输出

## 安装依赖

```bash
pip install -r requirements.txt
```

## 配置管理

项目使用`.env`文件管理配置参数，提供了工程化的配置管理方式。

### 初始化配置

1. 复制环境变量模板：
```bash
cp .env.example .env
```

2. 编辑`.env`文件，修改你需要的参数：
```bash
# 模型配置
BASE_MODEL=roberta-base
BATCH_SIZE=4
LEARNING_RATE=2e-5
NUM_EPOCHS=5

# GPU和显存优化配置（激进优化方案）
GPU_ID=2
FP16=True  # 混合精度训练，节省显存
GRADIENT_CHECKPOINTING=True  # 梯度检查点，节省显存

# TensorBoard
USE_TENSORBOARD=True
LOG_INTERVAL=10

# Checkpoint管理
SAVE_EVERY_N_EPOCHS=1
MAX_CHECKPOINTS=5  # 最多保留5个checkpoint
```

### 主要配置项

- **数据路径**: `TRAIN_FILE`, `TEST_FILE`, `ARTICLES_FILE`
- **模型配置**: `BASE_MODEL`, `NUM_LABELS`, `FREEZE_BASE`, `USE_METADATA`
- **训练配置**: `BATCH_SIZE`, `LEARNING_RATE`, `NUM_EPOCHS`, `MAX_LENGTH`
- **GPU和显存优化**: `GPU_ID`, `FP16`, `GRADIENT_CHECKPOINTING`
- **Checkpoint**: `SAVE_EVERY_N_EPOCHS`, `SAVE_BEST_ONLY`, `MAX_CHECKPOINTS`, `SAVE_CHECKPOINT_FROM_EPOCH`
- **TensorBoard**: `USE_TENSORBOARD`, `LOG_INTERVAL`
- **日志**: `LOG_LEVEL`, `LOG_FILE`

详细配置说明请参考`.env.example`文件。

## 数据准备

确保数据文件位于 `datasets/news_personality/` 目录下：
- `train.tsv`: 训练集
- `test.tsv`: 测试集
- `articles.csv`: 新闻文章

## 使用方法

### 训练模型

#### 使用环境变量配置（推荐）

```bash
# 直接运行，使用.env文件中的配置
python train.py
```

#### 使用命令行参数覆盖配置

```bash
# 最开始的 per_R: 0.5416 的模型


# 使用优化配置（节省显存），但是用（gte），per_R: 0.4071

# 切换基座embedding模型为roberta-base
python train.py --base_model roberta-base --batch_size 32 --learning_rate 2e-5 --num_epochs 30
```

**注意**: 显存优化配置（FP16、梯度检查点等）需要在`.env`文件中设置。

#### 使用自定义配置文件

```bash
python train.py --config /path/to/custom.env
```

#### 从Checkpoint恢复训练

如果训练中断，可以从checkpoint继续训练：

```bash
# 从指定的checkpoint恢复
python train.py --resume_from ./checkpoints/checkpoints/checkpoint_epoch_3.pt

# 从最佳模型恢复
python train.py --resume_from ./checkpoints/best_model.pt

# 恢复训练时可以覆盖配置参数（继续之前训练到2个epoch）
python train.py --resume_from ./checkpoints/best_model.pt --num_epochs 50
```

**注意**：
- Checkpoint会自动保存模型权重、优化器状态、调度器状态、训练配置等
- 恢复训练时会自动从下一个epoch开始
- 会恢复最佳验证分数和global_step
- 确保checkpoint文件完整，否则可能无法正确恢复

### TensorBoard可视化

训练过程中会自动记录指标到TensorBoard。启动TensorBoard：

```bash
tensorboard --logdir ./runs
```

然后在浏览器中打开 `http://localhost:6006` 查看训练曲线。

TensorBoard会记录：
- 训练/验证损失
- 训练/验证Pearson相关系数
- 各维度的Pearson相关系数
- 学习率变化

### Checkpoint管理

训练过程中会自动保存checkpoint：

- **最佳模型**: 自动保存为 `checkpoints/best_model.pt`
- **定期保存**: 根据`SAVE_EVERY_N_EPOCHS`配置定期保存
- **自动清理**: 如果设置了`MAX_CHECKPOINTS`，会自动清理旧checkpoint
- **保存范围**: 通过`SAVE_CHECKPOINT_FROM_EPOCH`控制从最后几个epoch开始保存（默认10，-1表示全程保存）

Checkpoint包含：
- 模型权重
- 优化器状态
- 调度器状态
- 训练配置
- 验证分数

### 测试/预测

```bash
python test.py

python test.py     --checkpoint ./checkpoints/checkpoints/checkpoint_epoch_30.pt     --test_file datasets/news_personality/test.tsv     --articles_file datasets/news_personality/articles.csv     --output_file predictions.tsv

python test_for_bert_base_personality.py --test_file datasets/news_personality/test.tsv     --articles_file datasets/news_personality/articles.csv     --output_file predictions.tsv
```

## 支持的基座模型

- `roberta-base`（默认）
- `Minej/bert-base-personality`（专门用于大五人格预测的BERT模型，会自动提取encoder部分）
- `Alibaba-NLP/gte-multilingual-base`
- `bert-base-uncased`
- `microsoft/deberta-base`
- `distilbert-base-uncased`
- 或其他transformers支持的模型

**注意**：`Minej/bert-base-personality` 是一个带分类头的模型，系统会自动提取其BERT encoder部分作为基座模型使用。

## 模型架构

模型采用以下架构：
1. **基座模型**: 预训练的Transformer模型（可切换）
2. **特征提取**: 使用[CLS] token的表示
3. **回归头**: 线性层输出5个维度的预测值

## 评估指标

使用皮尔逊相关系数（Pearson Correlation）作为评估指标：
- 对每个维度分别计算相关系数
- 取五个维度的平均值作为最终指标

## 文件结构

```
.
├── config.py            # 配置管理模块
├── logger_config.py     # Logger配置模块
├── data_loader.py        # 数据加载模块
├── model.py             # 模型定义
├── train.py             # 训练脚本（支持TensorBoard和Checkpoint管理）
├── test.py              # 测试脚本
├── utils.py             # 工具函数
├── .env                 # 环境变量配置（需要自己创建）
├── .env.example         # 环境变量模板
├── .gitignore           # Git忽略文件
├── requirements.txt     # 依赖文件
└── README.md            # 说明文档
```

## 日志系统

项目使用统一的日志系统：

- **控制台输出**: 彩色日志，包含时间、级别、文件路径和行号
- **文件输出**: 可选的日志文件，支持自动轮转和压缩
- **日志级别**: 可通过`LOG_LEVEL`配置（DEBUG, INFO, WARNING, ERROR, CRITICAL）

日志格式示例：
```
2024-01-01 12:00:00 | INFO     | /path/to/file.py:123 | 消息内容
```

## 示例

### 使用默认模型训练（roberta-base）

```bash
python train.py
```

### 使用GTE-Multilingual训练

修改`.env`文件：
```bash
BASE_MODEL=Alibaba-NLP/gte-multilingual-base
```

然后运行：
```bash
python train.py
```

### 使用DeBERTa训练

修改`.env`文件：
```bash
BASE_MODEL=microsoft/deberta-base
```

然后运行：
```bash
python train.py
```

### 使用BERT训练（冻结基座）

修改`.env`文件：
```bash
BASE_MODEL=bert-base-uncased
FREEZE_BASE=True
```

### 使用人格预测专用BERT模型

`Minej/bert-base-personality` 是一个专门用于大五人格预测的BERT模型，系统会自动提取其encoder部分：

修改`.env`文件：
```bash
BASE_MODEL=Minej/bert-base-personality
```

然后运行：
```bash
python train.py
```

**注意**：该模型基于 `bert-base-uncased` 在大五人格数据集上微调，系统会自动提取BERT encoder部分，忽略其分类头。

### 禁用metadata

如果不想使用metadata（只使用文本特征），可以设置：

修改`.env`文件：
```bash
USE_METADATA=False
```

**注意**：当`USE_METADATA=False`时，模型的分类头只使用文本特征，不会拼接metadata的维度。默认值为`True`。

### 只保存最佳模型

修改`.env`文件：
```bash
SAVE_BEST_ONLY=True
SAVE_EVERY_N_EPOCHS=0
```

### 限制checkpoint数量

修改`.env`文件：
```bash
MAX_CHECKPOINTS=5  # 只保留最新的5个checkpoint
```

### 只在最后几个epoch保存checkpoint

如果训练很多epoch，可以只在最后几个epoch保存checkpoint以节省存储空间：

修改`.env`文件：
```bash
# 只在最后10个epoch保存checkpoint（如果总epochs=100，则从第91个epoch开始保存）
SAVE_CHECKPOINT_FROM_EPOCH=10

# 如果想全程都保存checkpoint，设置为-1
SAVE_CHECKPOINT_FROM_EPOCH=-1
```

**注意**：`SAVE_CHECKPOINT_FROM_EPOCH` 与 `SAVE_EVERY_N_EPOCHS` 配合使用：
- 如果 `SAVE_CHECKPOINT_FROM_EPOCH=10`，`SAVE_EVERY_N_EPOCHS=2`，总epochs=100
- 则从第91个epoch开始，每2个epoch保存一次checkpoint（即91, 93, 95, 97, 99, 100）

### 显存优化配置

如果遇到显存不足的问题，可以使用以下优化配置：

```bash
# 在.env文件中
GPU_ID=2  # 指定使用的GPU
BATCH_SIZE=4  # 减小batch size
FP16=True  # 混合精度训练，节省40-50%显存
GRADIENT_CHECKPOINTING=True  # 梯度检查点，节省30-40%显存
```

**预期效果**:
- 显存需求: 从 ~2-2.5GB 降低到 ~0.6-0.8GB
- 训练速度: 可能稍慢（梯度检查点），但FP16会加速

详细说明请参考 `MEMORY_OPTIMIZATION.md` 文件。

## 注意事项

1. 首次运行时会自动下载预训练模型，需要网络连接
2. 建议使用GPU训练，CPU训练速度较慢
3. 可以根据显存大小调整`BATCH_SIZE`
4. 输出文件格式为TSV，使用制表符分隔
5. `.env`文件包含敏感配置，已添加到`.gitignore`，不会提交到版本控制
6. 使用`.env.example`作为配置模板
7. 如果显存不足，建议启用`FP16`和`GRADIENT_CHECKPOINTING`

## 故障排除

### TensorBoard无法启动

确保已安装tensorboard：
```bash
pip install tensorboard
```

### Checkpoint加载失败

确保checkpoint文件完整，包含所有必要的状态字典。

### 内存不足

减小`BATCH_SIZE`或`MAX_LENGTH`。
