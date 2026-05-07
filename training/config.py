# Hyperparameter and path configuration for the full pipeline.
# All modules import from here so there is a single source of truth.


class Config:
    # Pretrained backbone (used for encoding + recurrent loop)
    backbone_name: str = "gpt2"         # HuggingFace model ID or local path
    backbone_dim: int = 768             # Hidden dimension of the backbone

    # Answer decoder model (instruction-tuned; used only for final generation)
    # Kept separate from the backbone so the loop stays lightweight.
    answer_model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"

    # Mem tokens (attention bias)
    num_mem_tokens: int = 32            # Number of bias vectors; kept small by design

    # Recurrent loop
    max_ticks: int = 8                  # Hard upper bound on loop iterations
    stop_confidence_threshold: float = 0.9  # Early exit if "done" score exceeds this

    # Answer decoder sampling
    max_new_tokens: int = 512
    temperature: float = 1.0

    # Knowledge update gate
    novelty_threshold: float = 0.5
    confidence_threshold: float = 0.7

    # File store paths
    store_path: str = "knowledge/store.jsonl"
    queue_path: str = "knowledge/queue.jsonl"

    # RAG
    rag_top_k: int = 5

    # Tool floor — minimum exploration rate for tool/memory branches
    # Formula: final_score = raw_score * (1 - tool_floor_x) + tool_floor_x
    # Must be in [0, 0.05]; prevents routing collapse onto the decoder branch
    tool_floor_x: float = 0.05
