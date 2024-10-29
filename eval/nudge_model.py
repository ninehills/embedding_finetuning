import logging 
import torch
from tqdm import tqdm
import numpy as np
from typing import Dict, Optional, List
from eval.base import Embedding
from eval.dataset import RAGDataset
from eval.utils import embed_cached

class NudgeModel(Embedding):
    """The algorithm implemented here and the current state of the art is called [NUDGE](https://www.arxiv.org/abs/2409.02343).
    If a validation dataset is provided, the best model is evaluated and saved based on the validation loss at the end of every epoch.
    """

    def __init__(
        self,
        embed_model: Embedding,
        nudge_type: str = "nudge-n",
        device: Optional[str] = None,
        cache_dir: str = "./cache",
    ) -> None:
        """Init params."""
        try:
            from nudge import NUDGEN, NUDGEM
        except ImportError:
            raise ImportError("Please install nudge: pip install nudge-ft")
        if device is None:
            self._target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._target_device = torch.device(device)

        self.embed_model = embed_model
        self.nudge = (
            NUDGEN(device=self._target_device)
            if nudge_type == "nudge-n"
            else NUDGEM(device=self._target_device)
        )

        self.cache_dir = cache_dir

    def _format_dataset(
        self,
        dataset: RAGDataset,
        split: str,
        corpus_keys: List[str],
    ):
        query_ids = dataset.queries_split[split]
        q_embs = self.embed_model.embed_query(query_values)
        
        # 创建一个字典,将corpus_keys映射到它们的索引
        corpus_key_to_index = {key: index for index, key in enumerate(corpus_keys)}
        
        q_ans_indx = []
        for query_id in tqdm(query_ids, desc="Processing queries"):
            relevant_docs = dataset.relevant_docs[query_id]
            # 使用映射字典来获取索引,而不是每次都调用index方法
            relevant_doc_indices = [corpus_key_to_index[doc] for doc in relevant_docs]
            q_ans_indx.append(relevant_doc_indices)

        return {
            "q_embs": np.array(q_embs),
            "q_ans_indx": q_ans_indx
        }


    def finetune(self, dataset: RAGDataset):
        self.corpus_keys, self.corpus_values, self.corpus_embeddings = embed_cached(self.embed_model, dataset.corpus, self.cache_dir, type="corpus")
        self.train_dataset = self._format_dataset(dataset, split="train", corpus_keys=self.corpus_keys)
        self.val_dataset = self._format_dataset(dataset, split="val", corpus_keys=self.corpus_keys)

        print(f"self.corpus_embeddings[0]: {self.corpus_embeddings[0][:10]}")
        self.corpus_embeddings = self.nudge.finetune_embeddings(
            embeddings=self.corpus_embeddings.copy(),
            train_set=self.train_dataset,
            val_set=self.val_dataset,
            nontrain_embeddings=None,
            val_batch_size=256,
            gamma=None,
        )
        print(f"self.corpus_embeddings[0]: {self.corpus_embeddings[0][:10]}")
    def insert_data_and_finetune(
        self,
        new_dataset: RAGDataset,
    ):
        """
        Insert data and finetune. This should only be done if the new data you are inserting does not conflict with the already existing data. It's important to not finetune multiple times as this can cause the embeddings to lose semantic meaning since they will become further from the original embeddings.
        """
        new_corpus_batch = new_dataset.corpus
        # if any of the new ids are already in the existing corpus, raise an error
        if any(id in self.corpus for id in new_corpus_batch):
            raise ValueError(
                f"ID {id} already exists in the existing corpus. New IDs must be unique."
            )

        # get the embeddings for the new corpus
        new_corpus_initial_embeddings_batch = self._get_corpus_embeddings(
            new_corpus_batch
        )
        new_corpus_keys, new_corpus_values, new_corpus_embeddings = embed_cached(self.embed_model, new_corpus_batch, self.cache_dir, type="corpus")

        existing_corpus_embeddings = self.corpus_embeddings

        new_train_dataset = self._format_dataset(
            new_dataset, split="train", corpus_keys=new_corpus_keys
        )
        new_val_dataset = self._format_dataset(
            new_dataset, split="val", corpus_keys=new_corpus_keys
        )

        new_corpus_embeddings_batch = self.nudge.finetune_embeddings(
            embeddings=new_corpus_initial_embeddings_batch,
            train_set=new_train_dataset,
            val_set=new_val_dataset,
            # runs faster by filtering the embeddings which will not have any queries
            nontrain_embeddings=existing_corpus_embeddings,
            val_batch_size=256,
            gamma=None,
        )

        self.corpus_embeddings = np.concatenate(
            [existing_corpus_embeddings, new_corpus_embeddings_batch]
        )

    def get_finetuned_corpus_embeddings(self):
        return self.corpus_embeddings


    def embed_documents(self, corpus: List[str]):
        assert self.corpus_values == corpus
        return self.corpus_embeddings

    def embed_query(self, queries: List[str]):
        return self.embed_model.embed_query(queries)
    
    def embed(self, texts: List[str]):
        return self.embed_model.embed(texts)

    def __str__(self):
        return f"NudgeModel({self.embed_model.name})"
