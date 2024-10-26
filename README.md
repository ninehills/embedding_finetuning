# Embedding Model Fine-Tuning 案例


## 1. 准备环境

测试环境：WSL2 + CUDA 12.4

```bash
conda create -n embedding python=3.10 -y
conda activate embedding

# install pytorch with cuda 12.4, see https://pytorch.org/get-started/locally/
# because this bug: https://github.com/huggingface/diffusers/issues/9704, we need to install pytorch-nightly or torch 2.4.
conda install pytorch==2.4.1 torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
#pip3 install torch==2.4.1 torchvision torchaudio

# install other dependencies
pip install -r requirements.txt
```

## 2. 准备数据集


使用标准的 RAG 评估数据集格式作为 Retrieval 评估数据集格式，其中 `reference_answers` 字段在评估检索效果时可以省略。

评估数据集由如下列组成：

```json
{
    "queries": {
        "<query_id>": "<query>",
        ...
    },
    "corpus": {
        "<corpus_id>": "<corpus>",
        ...
    },
    "relevant_docs": {
        "<query_id>": ["<corpus_id>", ...], //每个 Query 可能对应多个 corpus，但是在本案例中，只包含一个。
        ...
    },
    "negative_docs": {
        "<query_id>": ["<corpus_id>", ...],
        ...
    },
    "reference_answers": { // 如果只检查检索效果，可以不需要 refrerence_answer 字段。
        "<query_id>": "<reference_answer>",
    }
}
```


```bash
# https://github.com/AIR-Bench/AIR-Bench/blob/main/docs/available_tasks.md#air-bench_2405
PYTHONPATH="." python utils/convert_airbench_to_ragdataset.py \
    --dataset "qa_healthcare_zh" \
    --train_val_split 0.5 \
    --output_path ./data/airbench_qa_healthcare_zh.json
corpus: 360218, train_queries: 187, val_queries: 187, save to ./data/airbench_qa_healthcare_zh.json

# https://huggingface.co/datasets/infgrad/retrieval_data_llm
PYTHONPATH="." python utils/convert_infgrad_retrieval_data_llm_to_ragdataset.py \
    --dataset "infgrad/retrieval_data_llm" \
    --train_val_split 0.01 \
    --output_path ./data/infgrad_retrieval_data_llm.json
corpus: 369307, train_queries: 182979, val_queries: 1848, save to ./data/infgrad_retrieval_data_llm.json
```

## 3. 进行基线评估

使用 [BAAI/bge-small-zh-v1.5](https://huggingface.co/BAAI/bge-small-zh-v1.5) 作为基线模型进行评估。该模型足够小，且容易进行微调。

评估分为2步：

1. 将所有的 corpus 转换为 embedding 向量，并构建 Faiss 索引
2. 进行 eval，在 val 数据上进行评估。

为避免多次操作的时候重复加载模型，制定统一的 embedding cache 机制。

```bash
PYTHONPATH="." python eval/evaluate_basic.py \
    --dataset_path "./data/airbench_qa_healthcare_zh.json" \
    --encoder "BAAI/bge-small-zh-v1.5" \
    --query_instruction "为这个句子生成表示以用于检索相关文章：" \
    --split "val" \
    --search_top_k 10
{
    "ndcg_at_1": 0.38503,
    "ndcg_at_3": 0.32342,
    "ndcg_at_5": 0.29477,
    "ndcg_at_10": 0.3141,
    "map_at_1": 0.07793,
    "map_at_3": 0.14495,
    "map_at_5": 0.1733,
    "map_at_10": 0.20662,
    "recall_at_1": 0.07793,
    "recall_at_3": 0.175,
    "recall_at_5": 0.2281,
    "recall_at_10": 0.32806
}

PYTHONPATH="." python eval/evaluate_basic.py \
    --dataset_path "./data/infgrad_retrieval_data_llm.json" \
    --encoder "BAAI/bge-small-zh-v1.5" \
    --query_instruction "为这个句子生成表示以用于检索相关文章：" \
    --split "val" \
    --search_top_k 10

{
    "ndcg_at_1": 0.5395,
    "ndcg_at_3": 0.63031,
    "ndcg_at_5": 0.65451,
    "ndcg_at_10": 0.67399,
    "map_at_1": 0.5395,
    "map_at_3": 0.60868,
    "map_at_5": 0.62218,
    "map_at_10": 0.63025,
    "recall_at_1": 0.5395,
    "recall_at_3": 0.69264,
    "recall_at_5": 0.75108,
    "recall_at_10": 0.81115
}
```

## 4. 微调准备

准备微调数据集和 Loss 函数。根据 <https://sbert.net/docs/sentence_transformer/loss_overview.html> 数据集基本提供如下三种格式：

1. 仅正样本： (anchor, positive) pairs，一般使用 MultipleNegativesRankingLoss 损失函数
2. 正负样本：(anchor, positive, negative_1, ..., negative_n)，一般使用 [MultipleNegativesRankingLoss](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#multiplenegativesrankingloss) 损失函数
3. 分数样本：(sentence_A, sentence_B, score)，一般使用 CoSENTLoss 损失函数。

第 3 种比较适合做知识蒸馏，也就是用比较强的模型的相似度分数来训练小模型。目前基本使用第一种或第二种。

TODO: 增加负样本挖掘工具。
TODO：做三组数据，分别是 train 中的正负样本，仅正样本，自动挖掘的负样本。从而分析微调的效果。


## 5. 全参数微调

使用 `finetune/sft.ipynb` 进行全参数微调。微调结果保存在 `checkpoint/bge-small-zh-v1.5-sft`，使用正负例数据。

```bash
# 检查在训练的 val 数据上的效果
PYTHONPATH="." python eval/evaluate_basic.py \
    --dataset_path "./data/infgrad_retrieval_data_llm.json" \
    --encoder "checkpoint/bge-small-zh-v1.5-sft" \
    --query_instruction "为这个句子生成表示以用于检索相关文章：" \
    --split "val" \
    --search_top_k 10
{
    "ndcg_at_1": 0.61472,
    "ndcg_at_3": 0.69546,
    "ndcg_at_5": 0.71485,
    "ndcg_at_10": 0.73301,
    "map_at_1": 0.61472,
    "map_at_3": 0.67551,
    "map_at_5": 0.68614,
    "map_at_10": 0.69371,
    "recall_at_1": 0.61472,
    "recall_at_3": 0.75325,
    "recall_at_5": 0.80087,
    "recall_at_10": 0.8566
}
# 可以看到 ndcg@10 从 0.67399 提升到 0.73301，提升 6pp

# 检查在非训练数据上的效果
PYTHONPATH="." python eval/evaluate_basic.py \
    --dataset_path "./data/airbench_qa_healthcare_zh.json" \
    --encoder "checkpoint/bge-small-zh-v1.5-sft" \
    --query_instruction "为这个句子生成表示以用于检索相关文章：" \
    --split "val" \
    --search_top_k 10
{
    "ndcg_at_1": 0.39037,
    "ndcg_at_3": 0.31856,
    "ndcg_at_5": 0.28343,
    "ndcg_at_10": 0.30265,
    "map_at_1": 0.07986,
    "map_at_3": 0.14648,
    "map_at_5": 0.16715,
    "map_at_10": 0.19863,
    "recall_at_1": 0.07986,
    "recall_at_3": 0.16695,
    "recall_at_5": 0.21002,
    "recall_at_10": 0.30356
}

# 可以看到 ndcg@10 从 0.3141 下降到 0.30265，下降 1.1pp
# 这就是灾难性遗忘的特点。
```

同时我们全参数微调只能适应数据量较多的训练集（1w+），如果数据量太小，模型基本学习不到。

## 6. LoRA 微调

TODO

## 7. NUDGE 微调

参考 <https://github.com/szeighami/nudge> 进行 NUDGE 微调。

```bash
PYTHONPATH="." python eval/evaluate_nudge.py \
    --dataset_path "./data/infgrad_retrieval_data_llm.json" \
    --encoder "checkpoint/bge-small-zh-v1.5-sft" \
    --query_instruction "为这个句子生成表示以用于检索相关文章：" \
    --split "val" \
    --search_top_k 10 \
    --use_nudge_n True
{
    "ndcg_at_1": 0.61147,
    "ndcg_at_3": 0.69472,
    "ndcg_at_5": 0.71365,
    "ndcg_at_10": 0.73091,
    "map_at_1": 0.61147,
    "map_at_3": 0.67397,
    "map_at_5": 0.68447,
    "map_at_10": 0.69166,
    "recall_at_1": 0.61147,
    "recall_at_3": 0.75487,
    "recall_at_5": 0.80087,
    "recall_at_10": 0.8539,
    "precision_at_1": 0.61147,
    "precision_at_3": 0.25162,
    "precision_at_5": 0.16017,
    "precision_at_10": 0.08539,
    "mrr_at_1": 0.61147,
    "mrr_at_3": 0.67397,
    "mrr_at_5": 0.68447,
    "mrr_at_10": 0.69166
}
```

ndcg@10 从 0.67399 提升到 0.73091，提升 5.7pp。和全参数微调的效果差不多。

NUDGE和全参数微调的对比：

- 训练时间： NUDGE 的训练其实是训练 Embedding 变换参数。
- NUDGE 针对 Embedding 后的数据，新增数据需要重复训练。且只影响 corpus embedding，query embedding 不变。
- SFT 需要重新部署模型。
- NUDGE 不需要 Negative samples，SFT 如果挖掘的难负样本不好，效果不是特别好。


