"""
PYTHONPATH="." python eval/evaluate_basic.py \
    --dataset_path "./data/airbench_qa_healthcare_zh.json" \
    --encoder "BAAI/bge-small-zh-v1.5" \
    --query_instruction "为这个句子生成表示以用于检索相关文章：" \
    --split "val" \
    --search_top_k 10
"""
import json
from dataclasses import dataclass, field
from typing import Dict, Any, List
from transformers import HfArgumentParser
from eval.sentence_transformer_model import SentenceTransformerEncoder

from eval.benchmark import Benchmark
import logging
logging.basicConfig(level=logging.INFO)

from eval.embedding_model import EmbeddingModelRetriever, NudgeModelPreparer
from eval.arguments import ModelArgs, EvalArgs
from eval.nudge_model import NudgeModel
@dataclass
class NudgeArgs:
    nudge_type: str = field(default="nudge-n", metadata={"help": "Use NUDGE-N or NUDGE-M."})
    device: str = field(default="cuda", metadata={"help": "Device to use. If no GPU, set to 'cpu'."})

def get_models(model_args: ModelArgs):
    embedding_model = SentenceTransformerEncoder(
        model_args.encoder,
        normalize_embeddings=model_args.normalize_embeddings,
        query_instruction_for_retrieval=model_args.query_instruction,
        passage_instruction_for_retrieval=model_args.passage_instruction,
        max_query_length=model_args.max_query_length,
        max_passage_length=model_args.max_passage_length,
        batch_size=model_args.batch_size,
        corpus_batch_size=model_args.corpus_batch_size,
        trust_remote_code=model_args.trust_remote_code,
    )
    return embedding_model


if __name__ == "__main__":
    parser = HfArgumentParser([ModelArgs, EvalArgs, NudgeArgs])
    model_args, eval_args, nudge_args = parser.parse_args_into_dataclasses()
    model_args: ModelArgs
    eval_args: EvalArgs
    nudge_args: NudgeArgs


    embedding_model = get_models(model_args)
    nudge_model = NudgeModel(
        embed_model=embedding_model,
        cache_dir=eval_args.cache_dir,
        nudge_type=nudge_args.nudge_type,
        device=nudge_args.device
    )

    evaluation = Benchmark(
        dataset_path=eval_args.dataset_path,
        split=eval_args.split,
        cache_dir=eval_args.cache_dir,
    )

    preparer = NudgeModelPreparer(
        nudge_model,
        cache_dir=eval_args.cache_dir,
        dataset=evaluation.dataset,
    )
    
    retriever = EmbeddingModelRetriever(
        nudge_model, 
        search_top_k=eval_args.search_top_k,
    )
    
    
    results = evaluation.run(
        preparer,
        retriever,
    )

    print(json.dumps(results, indent=4, ensure_ascii=False))
