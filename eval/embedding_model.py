import numpy as np
import hashlib
import os
from tqdm import tqdm
from typing import Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import logging

from langchain_community.vectorstores.faiss import FAISS
from eval.base import Retriever, Preparer, Embedding
from eval.nudge_model import NudgeModel
from eval.utils import embed_cached

logger = logging.getLogger(__name__)

class EmbeddingModelPreparer(Preparer):
    def __init__(self, embedding_model: Embedding, cache_dir: str, **kwargs):
        self.embedding_model = embedding_model
        self.cache_dir = cache_dir
    
    def __str__(self) -> str:
        return f"EmbeddingModelPreparer({self.embedding_model.name})"
    
    def __call__(
        self,
        corpus: Dict[str, str]
    ):
        """
        1. Encode the entire corpus into dense embeddings; 
        2. Create faiss index; 
        3. Optionally save embeddings.
        """
        # 快速计算所有 corpus 的 hash
        corpus_keys, corpus_values, corpus_embeddings = embed_cached(self.embedding_model, corpus, self.cache_dir, type="corpus")
        
        vectorstore = FAISS.from_embeddings(
            text_embeddings=zip(corpus_values, corpus_embeddings),
            metadatas=[{"id": id} for id in corpus_keys],
            embedding=self.embedding_model,
            distance_strategy="cosine",
        )

        return vectorstore

class NudgeModelPreparer(Preparer):
    def __init__(self, nudge_model: NudgeModel, cache_dir: str, **kwargs):
        self.nudge_model = nudge_model
        self.cache_dir = cache_dir
        self.dataset = kwargs.get("dataset", None)
        assert self.dataset is not None, "RAGDataset is required for NudgeModelPreparer"
    
    def __str__(self) -> str:
        return f"NudgeModelPreparer({self.nudge_model.name})"
    
    def __call__(self, corpus: Dict[str, str]) -> FAISS:
        self.nudge_model.finetune(self.dataset)
        vectorstore = FAISS.from_embeddings(
            text_embeddings=zip(self.nudge_model.corpus_values, self.nudge_model.corpus_embeddings),
            metadatas=[{"id": id} for id in self.nudge_model.corpus_keys],
            embedding=self.nudge_model,
            distance_strategy="cosine",
        )
        return vectorstore

def retrieve_docs(args):
    vectorstore, query_embedding, search_top_k = args
    docs_with_scores = vectorstore.similarity_search_with_score_by_vector(
        embedding=query_embedding,
        k=search_top_k,
    )
    return [(doc.metadata["id"], 1 / (1 + score)) for doc, score in docs_with_scores]


class EmbeddingModelRetriever(Retriever):
    def __init__(self, embedding_model: Embedding, search_top_k: int, **kwargs):
        self.search_top_k = search_top_k
        self.embedding_model = embedding_model

    def __str__(self) -> str:
        return f"EmbeddingModelRetriever({self.embedding_model.name})"
    
    def __call__(
        self,
        vectorstore: FAISS,
        queries: Dict[str, str],
        workers: int = 10,
    ):
        """
        1. Encode queries into dense embeddings;
        2. Search through faiss index
        """
        query_items = list(queries.items())
        query_keys = [i[0] for i in query_items]
        query_values = [i[1] for i in query_items]
        query_embeddings = self.embedding_model.embed_query(query_values)

        result = {}
        
        # 创建参数列表
        args_list = [(vectorstore, emb, self.search_top_k) for emb in query_embeddings]
        
        # 使用ThreadPoolExecutor来并行处理
        with ThreadPoolExecutor(max_workers=workers) as executor:  # 可以根据需要调整max_workers
            future_to_key = {executor.submit(retrieve_docs, args): key for args, key in zip(args_list, query_keys)}
            
            for future in tqdm(as_completed(future_to_key), total=len(query_keys), desc="Retrieving"):
                key = future_to_key[future]
                try:
                    docs_with_relevance = future.result()
                    assert any(0 <= score <= 1 for _, score in docs_with_relevance), "Score should be in [0, 1]"
                    result[key] = docs_with_relevance
                except Exception as exc:
                    print(f'{key} generated an exception: {exc}')
        
        return result

    