import logging
import os
import hashlib
from typing import List, Optional, Dict
import numpy as np
from tqdm import tqdm

from eval.base import Embedding

logger = logging.getLogger(__name__)

def embed_cached(embedding_model: Embedding, corpus: Dict[str, str], cache_dir: str, type: str = "corpus"):
    """Embed a query/corpus and cache the embeddings.
    
    Args:
        embedding_model: Embedding: The embedding model to use.
        corpus: Dict[str, str]: The corpus to embed.
        cache_dir: str: The directory to cache the embeddings.
        type: str: The type of the corpus, can be "corpus" or "query" or "none".
    
    Returns:
        Tuple[List[str], List[str], np.ndarray]: The corpus keys, corpus values, and corpus embeddings.
    """
    # 快速计算所有 corpus 的 hash
    corpus_items = list(corpus.items())
    corpus_keys = [i[0] for i in corpus_items]
    corpus_values = [i[1] for i in corpus_items]
    logger.info(f"embedding model name: {embedding_model.name}")
    cache_name_prefix = embedding_model.name + "/" + type
    logger.info(f"cache name prefix: {cache_name_prefix}")
    corpus_hash = hashlib.md5((embedding_model.name + "/" + type + "/" + "".join(corpus_keys)).encode("utf-8")).hexdigest()
    embedding_save_path = os.path.join(cache_dir, f"embeddings_cache.{corpus_hash}.memmap")

    if os.path.exists(embedding_save_path):
        logger.info(f"loading embeddings from {embedding_save_path}...")
        test = embedding_model.embed("test")[0]
        dtype = test.dtype
        dim = len(test)

        corpus_embeddings = np.memmap(
            embedding_save_path,
            mode="r",
            dtype=dtype
        ).reshape(-1, dim)
    else:
        logger.info(f"encoding corpus...")
        if type == "corpus":
            corpus_embeddings = embedding_model.embed_documents(corpus_values)
        elif type == "query":
            corpus_embeddings = embedding_model.embed_query(corpus_values)
        else:
            corpus_embeddings = embedding_model.embed(corpus_values)
        dim = corpus_embeddings.shape[-1]
        
        if len(corpus_embeddings) == len(corpus_keys) and cache_dir:
            logger.info(f"saving embeddings at {embedding_save_path}...")
            memmap = np.memmap(
                embedding_save_path,
                shape=corpus_embeddings.shape,
                mode="w+",
                dtype=corpus_embeddings.dtype
            )

            length = corpus_embeddings.shape[0]
            # add in batch
            save_batch_size = 10000
            if length > save_batch_size:
                for i in tqdm(range(0, length, save_batch_size), leave=False, desc="Saving Embeddings"):
                    j = min(i + save_batch_size, length)
                    memmap[i: j] = corpus_embeddings[i: j]
            else:
                memmap[:] = corpus_embeddings
    
    return corpus_keys, corpus_values, corpus_embeddings
    