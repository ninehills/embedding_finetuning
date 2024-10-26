import os
import logging
from typing import List, Optional, Union, Dict

from eval.dataset import RAGDataset
from eval.embedding_model import Preparer, Retriever
import pandas as pd

logger = logging.getLogger(__name__)

from eval.metrics import evaluate_metrics, evaluate_mrr

def compute_metrics(
    qrels: Dict[str, Dict[str, int]],
    search_results: Dict[str, Dict[str, float]],
    k_values: List[int],
):
    ndcg, _map, recall, precision = evaluate_metrics(
        qrels=qrels,
        results=search_results,
        k_values=k_values,
    )
    mrr = evaluate_mrr(
        qrels=qrels,
        results=search_results,
        k_values=k_values,
    )
    scores = {
        **{f"ndcg_at_{k.split('@')[1]}": v for (k, v) in ndcg.items()},
        **{f"map_at_{k.split('@')[1]}": v for (k, v) in _map.items()},
        **{f"recall_at_{k.split('@')[1]}": v for (k, v) in recall.items()},
        **{f"precision_at_{k.split('@')[1]}": v for (k, v) in precision.items()},
        **{f"mrr_at_{k.split('@')[1]}": v for (k, v) in mrr.items()},
    }
    return scores


class Benchmark():
    def __init__(
        self,
        dataset_path: str,
        cache_dir: str,
        split: Optional[str] = None,
    ):
        self.dataset = RAGDataset.from_file(dataset_path)
        os.makedirs(cache_dir, exist_ok=True)
        self.cache_dir = cache_dir
        self.split = split
        

    def run(
        self,
        preparer: Preparer,
        retriever: Retriever,
        k_values: List[int] = [1, 3, 5, 10],
    ):
        to_eval_queries = self.dataset.get_queries_split(self.split)
        logger.info(">>> Preparing corpus embeddings and faiss index...")
        #corpus_sample = {}
        #for query_id, query in to_eval_queries.items():
        #    corpus_ids = self.dataset.relevant_docs[query_id] + self.dataset.negative_docs[query_id]
        #    corpus_sample.update({i: self.dataset.corpus[i] for i in corpus_ids})

        vectorstore = preparer(self.dataset.corpus)
        #vectorstore = preparer(corpus_sample)

        logger.info(">>> Retrieving...")
        to_eval_queries = self.dataset.get_queries_split(self.split)
        retriever_result = retriever(vectorstore=vectorstore,
                  queries=to_eval_queries)
        
        logger.info(">>> Evaluation result:")
        qrels: Dict[str, Dict[str, int]] = {}
        for query_id, doc_ids in self.dataset.relevant_docs.items():
            qrels[query_id] = {doc_id: 1 for doc_id in doc_ids}
        search_results: Dict[str, Dict[str, float]] = {}
        for query_id, docs in retriever_result.items():
            search_results[query_id] = {doc_id: score for doc_id, score in docs}
        return compute_metrics(qrels, search_results, k_values)

