"""
# 将 infgrad/retrieval_data_llm转换为 RAGDataset 格式并保存
python utils/convert_infgrad_retrieval_data_llm_to_ragdataset.py \
    # 数据集名称
    --dataset "infgrad/retrieval_data_llm" \
    # 训练集和验证集的划分比例
    --train_val_split 0.2 \
    # 保存路径
    --output_path ./data/infgrad_retrieval_data_llm.json
"""
import argparse
import datasets
from collections import defaultdict
from tqdm import tqdm
from eval.dataset import RAGDataset


def main(args: argparse.Namespace):
    # 1. 从 Huggingface 上下载数据集
    ds = datasets.load_dataset(path=args.dataset, split="train")
    # 2. 将数据集转换为 RAGDataset 格式
    rag_dataset = RAGDataset()
    query_id_map = {}
    corpus_id_map = {}
    for line in tqdm(ds):
        query, positive, negative = line["Query"], line["Positive Document"], line["Hard Negative Document"]
        if query in query_id_map:
            continue
        query_id = f"q-{len(query_id_map)}"
        query_id_map[query] = query_id
        rag_dataset.queries[query_id] = query
        rag_dataset.relevant_docs[query_id] = []
        rag_dataset.negative_docs[query_id] = []

        if positive not in corpus_id_map:
            corpus_id = f"c-{len(corpus_id_map)}"
            corpus_id_map[positive] = corpus_id
            rag_dataset.corpus[corpus_id] = positive
        else:
            corpus_id = corpus_id_map[positive]
        rag_dataset.relevant_docs[query_id].append(corpus_id)

        if negative not in corpus_id_map:
            corpus_id = f"c-{len(corpus_id_map)}"
            corpus_id_map[negative] = corpus_id
            rag_dataset.corpus[corpus_id] = negative
        else:
            corpus_id = corpus_id_map[negative]
        rag_dataset.negative_docs[query_id].append(corpus_id)

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
