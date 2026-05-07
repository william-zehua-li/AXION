# AXION

A demo system built on a pretrained LM backbone with a recurrent reasoning loop and learnable mem tokens.

## Architecture

Pipeline stages (in order):

1. **GlobalPrePro** (`prepro/global_prepro.py`) — tokenize + encode input via pretrained backbone; encoding lives here, not in a separate encoder module
2. **GlobalRouter** (`router/global_router.py`) — coarse action plan and estimated tick count
3. **LoopController** (`loop/loop_controller.py`) — recurrent ticks until StopCheck exits:
   - LocalPrePro → inject mem bias + recurrent state
   - LocalRouter → pick branch: `tool | rag | mem | decoder`
   - Branch executes (Tool / RAG / MemTokens write / SharedDecoder)
   - SharedDecoder → update RecurrentState
   - StopCheck → exit on max ticks, EOS, or confidence threshold
4. **FinalPrePro** (`prepro/final_prepro.py`) — aggregate loop outputs for the answer decoder
5. **AnswerDecoder** (`decoder/answer_decoder.py`) — autoregressive generation with temperature + top-p
6. **UpdateGate** (`knowledge/update_gate.py`) — write / reject / queue new knowledge to file store

## Key design decisions

- **Mem tokens** are a small set of temporary attention bias vectors (`current_state + bias`). They are NOT a retrieval store. They are small (default 32), reset each inference run, and signal to the model where to focus attention within the current run.
- **Long-term memory** lives in a file (`knowledge/store.jsonl`), managed by `FileStore`. The `UpdateGate` decides what gets written there after each run. RAG reads from this file.
- **Encoder is part of GlobalPrePro** — not a separate module. The pretrained backbone does the encoding as the first step of preprocessing.
- **SharedDecoder** reuses one backbone layer (weight-tied) across all loop ticks to keep the model small.
- **Backbone**: defaults to `gpt2` (set in `training/config.py`). Backbone weights are frozen by default; only routers, mem tokens, stop-check head, and adapter layers are trained.

## File structure

```
core/           pipeline.py              — end-to-end wiring, single run() entry point
prepro/         global_prepro.py         — encode + preprocess
                local_prepro.py          — per-tick state + mem bias injection
                final_prepro.py          — aggregate before answer decoder
router/         global_router.py         — coarse plan
                local_router.py          — per-tick branch decision
memory/         mem_tokens.py            — small temp attention bias (reset each run)
                state.py                 — recurrent hidden state across ticks
loop/           loop_controller.py       — runs ticks
                stop_check.py            — exit condition
decoder/        shared_decoder.py        — weight-tied layer reused each tick
                answer_decoder.py        — final autoregressive generation
tools/          tool_executor.py         — external tool dispatch
                rag.py                   — retrieves from file store (read-only)
knowledge/      file_store.py            — persistent on-disk knowledge (JSONL)
                update_gate.py           — write / reject / queue after each run
training/       config.py               — all hyperparams in one place
                trainer.py              — training loop
eval/           evaluator.py            — benchmarks + metrics
demo/           run_demo.py             — interactive terminal demo
```

## Config defaults (`training/config.py`)

| Key | Default | Notes |
|-----|---------|-------|
| `backbone_name` | `gpt2` | HuggingFace model ID |
| `backbone_dim` | `768` | Must match backbone |
| `num_mem_tokens` | `32` | Keep small by design |
| `max_ticks` | `8` | Hard loop ceiling |
| `stop_confidence_threshold` | `0.9` | Early exit threshold |
| `max_new_tokens` | `512` | Answer decoder limit |
| `store_path` | `knowledge/store.jsonl` | Persistent knowledge file |
| `queue_path` | `knowledge/queue.jsonl` | Pending review queue |

## Entry points

```bash
# Interactive demo
python demo/run_demo.py --checkpoint <path>

# Training
python -m training.trainer

# Evaluation
python -m eval.evaluator
```
