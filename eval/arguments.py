
from dataclasses import dataclass, field

@dataclass
class ModelArgs:
    encoder: str = field(
        metadata={
            "help": 'Encoder name or path. For example, "BAAI/bge-m3" or "path/to/encoder".',
            "required": True,
        }
    )
    normalize_embeddings: bool = field(
        default=True, metadata={"help": "Normalize embeddings or not"}
    )
    query_instruction: str = field(
        default=None, metadata={"help": "query instruction for retrieval"}
    )
    passage_instruction: str = field(
        default=None, metadata={"help": "passage instruction for retrieval"}
    )
    max_query_length: int = field(default=512, metadata={"help": "Max query length for retrieval."})
    max_passage_length: int = field(
        default=512, metadata={"help": "Max passage length for retrieval."}
    )
    batch_size: int = field(default=64, metadata={"help": "Inference batch size for retrieval."})
    corpus_batch_size: int = field(
        default=0,
        metadata={
            "help": "Inference batch size for corpus. If 0, then use `batch_size`."
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={"help": "Trust remote code or not."},
    )
    corpus_chunk_size: int = field(
        default=512,
        metadata={"help": "Chunk size for encoding corpus. now not used."}
    )

@dataclass
class EvalArgs:
    dataset_path: str = field(
        metadata={"help": "Path to the dataset to evaluate."}
    )
    output_dir: str = field(
        default="./search_results", metadata={"help": "Path to save results."}
    )
    split: str = field(
        default=None, metadata={"help": "Split to evaluate."}
    )
    search_top_k: int = field(
        default=100, metadata={"help": "Top k values for evaluation."}
    )
    cache_dir: str = field(
        default="./.cache", metadata={"help": "Cache directory for embeddings."}
    )
    overwrite: bool = field(
        default=False, metadata={"help": "whether to overwrite evaluation results"}
    )
