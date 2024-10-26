"""https://github.com/AIR-Bench/AIR-Bench/blob/0.1.0/air_benchmark/evaluation_utils/utils.py"""
import pytrec_eval
from collections import defaultdict
from typing import Dict, List, Tuple


# Modified by me
def evaluate_mrr(
    qrels: Dict[str, Dict[str, int]],
    results: Dict[str, Dict[str, float]],
    k_values: List[int],
) -> Dict[str, float]:
    mrr = defaultdict(list)
    k_max = max(k_values)
    
    for query_id, doc_scores in results.items():
        # 对文档得分进行排序
        sorted_docs = sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)[:k_max]
        
        for k in k_values:
            rr = 0
            for rank, (doc_id, _) in enumerate(sorted_docs[:k], 1):
                # 检查文档是否在 qrels 中且相关度大于 0
                if query_id in qrels and qrels[query_id].get(doc_id, 0) > 0:
                    rr = 1.0 / rank
                    break
            mrr[f"MRR@{k}"].append(rr)
    
    # 计算每个 k 值的平均 MRR
    avg_mrr = {}
    for k in k_values:
        avg_mrr[f"MRR@{k}"] = round(sum(mrr[f"MRR@{k}"]) / len(mrr[f"MRR@{k}"]), 5)
    
    return avg_mrr


# Modified from https://github.com/embeddings-benchmark/mteb/blob/18f730696451a5aaa026494cecf288fd5cde9fd0/mteb/evaluation/evaluators/RetrievalEvaluator.py#L501
def evaluate_metrics(
    qrels: Dict[str, Dict[str, int]],
    results: Dict[str, Dict[str, float]],
    k_values: List[int],
) -> Tuple[
    Dict[str, float],
    Dict[str, float],
    Dict[str, float],
    Dict[str, float],
]:
    all_ndcgs, all_aps, all_recalls, all_precisions = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)

    map_string = "map_cut." + ",".join([str(k) for k in k_values])
    ndcg_string = "ndcg_cut." + ",".join([str(k) for k in k_values])
    recall_string = "recall." + ",".join([str(k) for k in k_values])
    precision_string = "P." + ",".join([str(k) for k in k_values])
    evaluator = pytrec_eval.RelevanceEvaluator(
        qrels, {map_string, ndcg_string, recall_string, precision_string}
    )
    scores = evaluator.evaluate(results)

    for query_id in scores.keys():
        for k in k_values:
            all_ndcgs[f"NDCG@{k}"].append(scores[query_id]["ndcg_cut_" + str(k)])
            all_aps[f"MAP@{k}"].append(scores[query_id]["map_cut_" + str(k)])
            all_recalls[f"Recall@{k}"].append(scores[query_id]["recall_" + str(k)])
            all_precisions[f"P@{k}"].append(scores[query_id]["P_" + str(k)])

    ndcg, _map, recall, precision = (
        all_ndcgs.copy(),
        all_aps.copy(),
        all_recalls.copy(),
        all_precisions.copy(),
    )

    for k in k_values:
        ndcg[f"NDCG@{k}"] = round(sum(ndcg[f"NDCG@{k}"]) / len(scores), 5)
        _map[f"MAP@{k}"] = round(sum(_map[f"MAP@{k}"]) / len(scores), 5)
        recall[f"Recall@{k}"] = round(sum(recall[f"Recall@{k}"]) / len(scores), 5)
        precision[f"P@{k}"] = round(sum(precision[f"P@{k}"]) / len(scores), 5)

    return ndcg, _map, recall, precision
