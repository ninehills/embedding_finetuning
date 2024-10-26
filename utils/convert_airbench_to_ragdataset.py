"""
# 将 airbench_qa_healthcare_zh_dev 转换为 RAGDataset 格式并保存
python utils/convert_airbench_to_ragdataset.py \
    # 数据集名称
    --dataset "qa_healthcare_zh" \
    # 训练集和验证集的划分比例
    --train_val_split 0.2 \
    # 保存路径
    --output_path ./data/airbench_qa_healthcare_zh.json
"""
import argparse
import datasets
from collections import defaultdict
from tqdm import tqdm
from eval.dataset import RAGDataset


AIR_BENCH_VERSION = "AIR-Bench_24.05"
AIR_BENCH_PATH = "AIR-Bench"

def main(args: argparse.Namespace):
    # 1. 从 Huggingface 上下载数据集
    corpus_ds = datasets.load_dataset(path=f"{AIR_BENCH_PATH}/{args.dataset}", data_files=[f"{AIR_BENCH_VERSION}/default/corpus.jsonl"], split="train")
    query_ds = datasets.load_dataset(path=f"{AIR_BENCH_PATH}/{args.dataset}", data_files=[f"{AIR_BENCH_VERSION}/default/dev_queries.jsonl"], split="train")
    qrels_ds = datasets.load_dataset(path=f"{AIR_BENCH_PATH}/qrels-{args.dataset}-dev", split="qrels_default_dev")

    # 2. 将数据集转换为 RAGDataset 格式
    rag_dataset = RAGDataset()
    for corpus in tqdm(corpus_ds):
        rag_dataset.corpus[corpus["id"]] = corpus["text"]

    for query in tqdm(query_ds):
        rag_dataset.queries[query["id"]] = query["text"]
    
    # Dataset({features: ['qid', 'docid', 'relevance'],
    relevance_dict = defaultdict(set)
    negative_dict = defaultdict(set)
    for qrel in tqdm(qrels_ds):
        if qrel["relevance"] == 1:
            relevance_dict[qrel["qid"]].add(qrel["docid"])
        elif qrel["relevance"] == 0:
            negative_dict[qrel["qid"]].add(qrel["docid"])
        else:
            raise ValueError(f"Invalid relevance value: {qrel['relevance']}")
    
    rag_dataset.relevant_docs = {qid: list(docs) for qid, docs in relevance_dict.items()}
    rag_dataset.negative_docs = {qid: list(docs) for qid, docs in negative_dict.items()}

    # 3. 进行拆分
    rag_dataset.split(args.train_val_split, seed=42)

    # 4. 保存数据集
    print(f"corpus: {len(rag_dataset.corpus)}, train_queries: {len(rag_dataset.queries_split['train'])}, val_queries: {len(rag_dataset.queries_split['val'])}, save to {args.output_path}")
    rag_dataset.save(args.output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str, required=True,
        help="The name of the dataset to convert."
    )
    parser.add_argument(
        "--train_val_split",
        type=float, required=True, default=0.2,
        help="The split ratio between the training set and the validation set. If set to 0, only the training set will be saved. (default: 0.2)"
    )
    parser.add_argument(
        "--output_path",
        type=str, required=True,
        help="The path to save the dataset."
    )
    args = parser.parse_args()


    main(args)
