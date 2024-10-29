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

使用 `infgrad/retrieval_data_llm` 作为第一个数据集用于训练和验证，其特点是具备挖掘良好的正例和负例，比较适合微调。

```bash
# https://huggingface.co/datasets/infgrad/retrieval_data_llm
PYTHONPATH="." python utils/convert_infgrad_retrieval_data_llm_to_ragdataset.py \
    --dataset "infgrad/retrieval_data_llm" \
    --train_val_split 0.01 \
    --output_path ./data/infgrad_retrieval_data_llm.json
corpus: 369307, train_queries: 182979, val_queries: 1848, save to ./data/infgrad_retrieval_data_llm.json
```

第二个数据集使用 AirBench-QA-Healthcare-zh，他有 374 条验证集，但是没有训练集，正好可以实践数据合成。
```bash
# https://github.com/AIR-Bench/AIR-Bench/blob/main/docs/available_tasks.md#air-bench_2405
PYTHONPATH="." python utils/convert_airbench_to_ragdataset.py \
    --dataset "qa_healthcare_zh" \
    --train_val_split 1.0 \
    --output_path ./data/airbench_qa_healthcare_zh.json
corpus: 360218, train_queries: 0, val_queries: 374, save to ./data/airbench_qa_healthcare_zh.json


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
    "ndcg_at_1": 0.40107,
    "ndcg_at_3": 0.32827,
    "ndcg_at_5": 0.30947,
    "ndcg_at_10": 0.32522,
    "map_at_1": 0.0833,
    "map_at_3": 0.14634,
    "map_at_5": 0.18023,
    "map_at_10": 0.21772,
    "recall_at_1": 0.0833,
    "recall_at_3": 0.17145,
    "recall_at_5": 0.23619,
    "recall_at_10": 0.33766,
    "precision_at_1": 0.40107,
    "precision_at_3": 0.29679,
    "precision_at_5": 0.2508,
    "precision_at_10": 0.18503,
    "mrr_at_1": 0.40107,
    "mrr_at_3": 0.47014,
    "mrr_at_5": 0.49073,
    "mrr_at_10": 0.50432
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

### 4.1 数据合成

对 airbench_qa_healthcare_zh 数据集实践数据挖掘，从 corpus 中挖掘出正样本（Query和Corpus 的关联关系）。实际生产环境下，corpus 可以通过对文档的识别和切割得到，本文从略。

使用 `finetune/data_synthesis.ipynb` 进行数据合成。合成后的数据集在 `data/airbench_qa_healthcare_zh_synthesis.json`，共有 9810 条QA对。 

### 4.2 难负例挖掘

针对 airbench_qa_healthcare_zh 数据集，使用 `finetune/hard_negative_mining.ipynb` 进行难负例挖掘。可以控制挖掘的负例的数量，暂定每个 Query 挖掘 15 个负例。合成的数据集在 `data/airbench_qa_healthcare_zh_synthesis_hard_negative.json`。


## 5. 全参数微调

使用 `finetune/sft_infgrad.ipynb` 进行全参数微调。微调结果保存在 `checkpoint/bge-small-zh-v1.5-sft`，使用 infgrad_retrieval_data_llm 数据集的正负例数据进行微调，并在 airbench_qa_healthcare_zh 数据集上验证 Held-out 效果。

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
    "ndcg_at_1": 0.37701,
    "ndcg_at_3": 0.31708,
    "ndcg_at_5": 0.29099,
    "ndcg_at_10": 0.30451,
    "map_at_1": 0.07493,
    "map_at_3": 0.14166,
    "map_at_5": 0.16724,
    "map_at_10": 0.19973,
    "recall_at_1": 0.07493,
    "recall_at_3": 0.16802,
    "recall_at_5": 0.22041,
    "recall_at_10": 0.31009,
    "precision_at_1": 0.37701,
    "precision_at_3": 0.28699,
    "precision_at_5": 0.23529,
    "precision_at_10": 0.17219,
    "mrr_at_1": 0.37701,
    "mrr_at_3": 0.44563,
    "mrr_at_5": 0.46582,
    "mrr_at_10": 0.47957
}

# 可以看到 ndcg@10 从 0.32522 下降到 0.30451，下降 2.1pp
# 这就是灾难性遗忘的特点。
```

也测试下 `sft_airbench.ipynb`，使用 airbench_qa_healthcare_zh 我们自己合成的数据集进行微调，并在 infgrad_retrieval_data_llm 数据集上验证 Held-out 效果。 微调结果保存在 `checkpoint/bge-small-zh-v1.5-sft-airbench`

```bash
# 检查在训练的 val 数据上的效果
PYTHONPATH="." python eval/evaluate_basic.py \
    --dataset_path "./data/airbench_qa_healthcare_zh.json" \
    --encoder "checkpoint/bge-small-zh-v1.5-sft-airbench" \
    --query_instruction "为这个句子生成表示以用于检索相关文章：" \
    --split "val" \
    --search_top_k 10
{
    "ndcg_at_1": 0.32888,
    "ndcg_at_3": 0.27589,
    "ndcg_at_5": 0.25202,
    "ndcg_at_10": 0.26363,
    "map_at_1": 0.07424,
    "map_at_3": 0.12941,
    "map_at_5": 0.1511,
    "map_at_10": 0.17766,
    "recall_at_1": 0.07424,
    "recall_at_3": 0.14725,
    "recall_at_5": 0.18944,
    "recall_at_10": 0.26303,
    "precision_at_1": 0.32888,
    "precision_at_3": 0.24688,
    "precision_at_5": 0.19893,
    "precision_at_10": 0.14652,
    "mrr_at_1": 0.32888,
    "mrr_at_3": 0.3926,
    "mrr_at_5": 0.40477,
    "mrr_at_10": 0.41658
}
# 可以看到 ndcg@10 从 0.32522 下降到 0.26363，这是因为模型训练集和验证集的来源不同，而且训练过程没有充分收敛，后续可以优化。

# 检查在非训练数据上的效果
PYTHONPATH="." python eval/evaluate_basic.py \
    --dataset_path "./data/infgrad_retrieval_data_llm.json" \
    --encoder "checkpoint/bge-small-zh-v1.5-sft-airbench" \
    --query_instruction "为这个句子生成表示以用于检索相关文章：" \
    --split "val" \
    --search_top_k 10
{
    "ndcg_at_1": 0.50866,
    "ndcg_at_3": 0.5959,
    "ndcg_at_5": 0.62093,
    "ndcg_at_10": 0.64245,
    "map_at_1": 0.50866,
    "map_at_3": 0.57477,
    "map_at_5": 0.58856,
    "map_at_10": 0.59754,
    "recall_at_1": 0.50866,
    "recall_at_3": 0.65693,
    "recall_at_5": 0.71807,
    "recall_at_10": 0.78409,
    "precision_at_1": 0.50866,
    "precision_at_3": 0.21898,
    "precision_at_5": 0.14361,
    "precision_at_10": 0.07841,
    "mrr_at_1": 0.50866,
    "mrr_at_3": 0.57477,
    "mrr_at_5": 0.58856,
    "mrr_at_10": 0.59754
}
# 可以看到 ndcg@10 从 0.67399 下降到 0.64245，这就是灾难性遗忘的特点。
```


## 6. LoRA 微调

TODO

## 7. NUDGE 微调

TODO: nudge-n/-m 得到的best-gamma = 0.0 导致embedding 没有变化


其他 todo：

1. support load model as fp16
2. support dump as onnx-int8