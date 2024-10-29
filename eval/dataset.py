# 定义评估数据集结构
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Tuple, Optional
from collections import OrderedDict
import json
import random
import math
import logging

logger = logging.getLogger(__name__)

@dataclass
class RAGDataset:
    """RAGDataset
    """

    queries: OrderedDict[str, str] = field(default_factory=OrderedDict)  # query_id -> query
    corpus: OrderedDict[str, str] = field(default_factory=OrderedDict)  # corpus_id -> corpus
    relevant_docs: Dict[str, List[str]] = field(default_factory=dict)  # query_id -> list of corpus_id, no order.
    negative_docs: Dict[str, List[str]] = field(default_factory=dict)  # query_id -> list of corpus_id, no order. for hard negative training.
    reference_answers: Dict[str, str] = field(default_factory=dict)  # query_id -> reference_answer
    queries_split: Dict[str, List[str]] = field(default_factory=dict)  # 将 queries 分割为训练集和验证集

    def dict(self):
        return asdict(self)

    def save(self, path: str) -> None:
        """Save json."""
        with open(path, "w") as f:
            json.dump(self.dict(), f, indent=4, ensure_ascii=False)

    @classmethod
    def from_file(cls, path: str) -> "RAGDataset":
        """Load json."""
        with open(path) as f:
            data = json.load(f, object_pairs_hook=OrderedDict)
        return cls(**data)
    
    def get_queries_split(self, split: Optional[str] = None) -> Dict[str, str]:
        """获得特定split 的 queries"""
        if split is None:
            return self.queries
        return {i: self.queries[i] for i in self.queries_split[split]}

    def split(self, ratio: float, seed: int = 0):
        """将数据集分割为训练集和验证集。

        Args:
            ratio (float): 验证集占整个数据集的比例，范围应在0到1之间。
            seed (int): 随机种子，默认为0。

        Returns:
            None
        """
        if not 0 <= ratio <= 1:
            raise ValueError("ratio must be between 0 and 1.")
        
        if seed:
            random.seed(seed)

        # 获取所有查询ID并保持原有顺序
        query_ids = list(self.queries.keys())
        
        # 计算验证集的大小
        val_size = int(len(query_ids) * ratio)
        
        # 随机选择索引而不是直接打乱ID列表
        total_indices = list(range(len(query_ids)))
        val_indices = sorted(random.sample(total_indices, val_size))
        train_indices = sorted(set(total_indices) - set(val_indices))
        
        # 根据选择的索引获取ID
        val_ids = [query_ids[i] for i in val_indices]
        train_ids = [query_ids[i] for i in train_indices]

        self.queries_split["train"] = train_ids
        self.queries_split["val"] = val_ids

    def get_train_dataset(self,
            split: Optional[str] = None,
            negative_num: int = 0,
            query_instruction_for_retrieval: Optional[str] = None,
            passage_instruction_for_retrieval: Optional[str] = None,
        ):
        res = []
        queries = self.get_queries_split(split)
        for query_id, query in queries.items():
            item = {}
            if query_instruction_for_retrieval:
                item["query"] = query_instruction_for_retrieval + query
            
            relevant_docs_ids = self.relevant_docs.get(query_id, [])
            if not relevant_docs_ids:
                continue
            # random choice one positive doc
            item["pos"] = self.corpus[random.choice(relevant_docs_ids)]
            if passage_instruction_for_retrieval:
                item["pos"] = passage_instruction_for_retrieval + item["pos"]

            if negative_num > 0:
                negative_docs_ids = self.negative_docs.get(query_id, [])
                if len(negative_docs_ids) == 0:
                    logger.warning("No negative docs found for query_id: {}, skip this query".format(query_id))
                    continue
                if len(negative_docs_ids) < negative_num:
                    # 负面样本不足，则多采样
                    num = math.ceil(negative_num / len(negative_docs_ids))
                    negs_ids = random.sample(negative_docs_ids * num, negative_num)
                else:
                    negs_ids = random.sample(negative_docs_ids, negative_num)
                negs = [self.corpus[i] for i in negs_ids]

                if passage_instruction_for_retrieval:
                    negs = [passage_instruction_for_retrieval + doc for doc in negs]
                
                for i in range(negative_num):
                    item["neg_{}".format(i)] = negs[i]
            
            res.append(item)
        return res, ["query", "pos"] + ["neg_{}".format(i) for i in range(negative_num)]
