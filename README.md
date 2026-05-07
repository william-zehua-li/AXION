# AXION

A demo AI system built around a recurrent reasoning loop. Instead of generating an answer in one pass, AXION routes each query through a plan of steps — retrieve, compute, remember, reason — before handing the result to an instruction-tuned model for the final answer.

No training is needed to run this. The demo uses pretrained GPT-2 for the reasoning loop and Qwen2.5-0.5B-Instruct for the answer.

---

## How it works

```
Input
  │
  ▼
GlobalPrePro        tokenise + encode the query (GPT-2 backbone)
  │
  ▼
RuleRouter          classify the query, build an action plan
  │                   factual      → [rag, decoder]
  │                   calculation  → [tool, decoder]
  │                   uncertain    → [rag, decoder, rag]
  │                   reasoning    → [decoder, mem, decoder]
  ▼
Loop (per tick)
  ├─ LocalPrePro    condition the hidden state on tick position + mem bias
  ├─ LocalRouter    pick the branch for this tick (follows the plan exactly)
  │
  ├─ tool           run the calculator on the extracted expression
  ├─ rag            retrieve matching chunks from the knowledge store
  ├─ mem            write the current state into the attention bias tokens
  └─ decoder        refine the hidden representation
  │
  ▼
FinalPrePro         aggregate the last hidden state across all ticks
  │
  ▼
AnswerDecoder       build an evidence-grounded prompt, generate with Qwen
  │
  ▼
Answer
```

The loop exits early when confidence reaches the threshold (default 0.9), which in demo mode happens naturally on the last planned step.

---

## Project structure

```
AXION/
├── core/
│   └── pipeline.py          end-to-end orchestration
├── prepro/
│   ├── global_prepro.py     tokenise + backbone encoding
│   ├── local_prepro.py      per-tick conditioning, tick embeddings, ToolFloor
│   ├── final_prepro.py      aggregate loop outputs into one representation
│   └── tool_floor.py        minimum exploration guarantee for tool/mem branches
├── router/
│   ├── rule_router.py       regex classifier → deterministic action plan
│   ├── local_router.py      per-tick branch selector (strict in demo, ML in trained)
│   └── global_router.py     learned global planner (used after training)
├── loop/
│   ├── loop_controller.py   recurrent tick loop
│   └── stop_check.py        early-exit conditions (confidence / norm collapse)
├── decoder/
│   ├── shared_decoder.py    weight-tied GPT-2 layer used inside the loop
│   └── answer_decoder.py    Qwen-based evidence-grounded final generation
├── memory/
│   ├── mem_tokens.py        attention bias vectors (ephemeral, reset each run)
│   └── state.py             recurrent hidden state wrapper
├── tools/
│   ├── calculator.py        safe AST-based arithmetic (no eval, no model)
│   ├── rag.py               TF-IDF retrieval over the knowledge store
│   └── tool_executor.py     tool registry and dispatch
├── knowledge/
│   ├── file_store.py        JSONL store + in-memory TF-IDF index
│   └── update_gate.py       novelty + confidence filter before writing
├── training/
│   ├── config.py            all hyperparameters and paths
│   └── trainer.py           training loop (for future use)
├── demo/
│   └── run_demo.py          entry point
└── requirements.txt
```

---

## Setup

```bash
pip install -r requirements.txt
```

First run downloads GPT-2 and Qwen2.5-0.5B-Instruct and saves them to `.model_cache/`. Every run after that loads from disk.

---

## Running

```bash
python demo/run_demo.py
```

Or from Python:

```python
from core.pipeline import Pipeline
from training.config import Config

pipeline = Pipeline(Config())
result = pipeline.run("What is 17 * 24?")
print(result["answer"])
```

---

## Adding knowledge

The RAG branch retrieves from `knowledge/store.jsonl`. Each line is one fact:

```jsonl
{"text": "The Eiffel Tower is located in Paris, France. It was completed in 1889."}
{"text": "Python's GIL prevents true multi-threaded CPU parallelism for pure Python code."}
{"text": "The speed of light in a vacuum is approximately 299,792,458 metres per second."}
```

Add entries directly to the file — the TF-IDF index rebuilds automatically on the next run. Entries pending review go in `knowledge/queue.jsonl`, same format.

---

## Configuration

All settings live in `training/config.py`:

| Setting | Default | What it does |
|---|---|---|
| `backbone_name` | `gpt2` | Backbone model for encoding and the loop |
| `answer_model_name` | `Qwen/Qwen2.5-0.5B-Instruct` | Model used for final answer generation |
| `max_ticks` | `8` | Hard upper bound on loop iterations |
| `stop_confidence_threshold` | `0.9` | Exit early when confidence exceeds this |
| `num_mem_tokens` | `32` | Attention bias token count (ephemeral per run) |
| `tool_floor_x` | `0.05` | Minimum tool/mem exploration rate |
| `max_new_tokens` | `512` | Max tokens in the generated answer |
| `temperature` | `1.0` | Sampling temperature for Qwen |

---

## Calculator

Math queries are routed to a safe calculator — no model, no `eval()`. It uses Python's AST parser with a strict whitelist of allowed nodes.

Supports: `+ - * / // % **`, `sqrt`, `sin`, `cos`, `tan`, `log`, `log10`, `factorial`, `abs`, `ceil`, `floor`, and constants `pi` and `e`.

Natural language is normalised before parsing:

| Input | Extracted expression | Result |
|---|---|---|
| `"What is 144 / 12?"` | `144 / 12` | `12` |
| `"Calculate the square root of 256"` | `sqrt(256)` | `16` |
| `"What is 2 to the power of 10?"` | `2**10` | `1024` |
| `"Factorial of 7"` | `factorial(7)` | `5040` |

---

## Sample runs
## Sample 1
```text
python demo/run_demo.py --query "What is the capital of France?"

════════════════════════════════════════════════════════════
  Input : 'What is the capital of France?'
  Mode  : demo
════════════════════════════════════════════════════════════
  Query type : factual

  ────────────────────────────────────────────────────────
  Global plan (2 steps): rag → decoder
  ────────────────────────────────────────────────────────
  Tick 0  hint=rag       branch=rag       tool_score=0.539  conf=0.500  │ 3 chunk(s) ✓
  Tick 1  hint=decoder   branch=decoder   tool_score=0.547  conf=1.000  │ refining representation  ← STOP (confident)
  ────────────────────────────────────────────────────────

════════════════════════════════════════════════════════════
  Answer    : The capital of France is Paris.
  Ticks used: 2
  Knowledge         : demo mode — writes disabled
════════════════════════════════════════════════════════════

The capital of France is Paris.
```
## Sample 2
```text
python demo/run_demo.py --query "Who is the president of the US in 2050"

════════════════════════════════════════════════════════════
  Input : 'Who is the president of the US in 2050'
  Mode  : demo
════════════════════════════════════════════════════════════
  Query type : factual

  ────────────────────────────────────────────────────────
  Global plan (2 steps): rag → decoder
  ────────────────────────────────────────────────────────
  Tick 0  hint=rag       branch=rag       tool_score=0.570  conf=0.500  │ 3 chunk(s) ✓
  Tick 1  hint=decoder   branch=decoder   tool_score=0.551  conf=1.000  │ refining representation  ← STOP (confident)
  ────────────────────────────────────────────────────────

════════════════════════════════════════════════════════════
  Answer    : I do not know.
  Ticks used: 2
  Knowledge         : demo mode — writes disabled
════════════════════════════════════════════════════════════

I do not know.
```
## Sample 3
```text
python demo/run_demo.py --query "Calculate 928374 * 17"

════════════════════════════════════════════════════════════
  Input : 'Calculate 928374 * 17'
  Mode  : demo
════════════════════════════════════════════════════════════
  Query type : calculation

  ────────────────────────────────────────────────────────
  Global plan (2 steps): tool → decoder
  ────────────────────────────────────────────────────────
  Tick 0  hint=tool      branch=tool      tool_score=0.507  conf=0.500  │ calculator('928374 * 17') = 15782358 ✓
  Tick 1  hint=decoder   branch=decoder   tool_score=0.539  conf=1.000  │ refining representation  ← STOP (confident)
  ────────────────────────────────────────────────────────

════════════════════════════════════════════════════════════
  Answer    : 15782358
  Ticks used: 2
  Knowledge         : demo mode — writes disabled
════════════════════════════════════════════════════════════

15782358
```
## Sample 4
```text
python demo/run_demo.py --query "Compare RNNs and Transformers for long-context memory."

════════════════════════════════════════════════════════════
  Input : 'Compare RNNs and Transformers for long-context memory.'
  Mode  : demo
════════════════════════════════════════════════════════════
  Query type : reasoning

  ────────────────────────────────────────────────────────
  Global plan (3 steps): decoder → mem → decoder
  ────────────────────────────────────────────────────────
  Tick 0  hint=decoder   branch=decoder   tool_score=0.489  conf=0.333  │ refining representation
  Tick 1  hint=mem       branch=mem       tool_score=0.511  conf=0.667  │ bias updated
  Tick 2  hint=decoder   branch=decoder   tool_score=0.549  conf=1.000  │ refining representation  ← STOP (confident)
  ────────────────────────────────────────────────────────

════════════════════════════════════════════════════════════
  Answer    : Transformer achieves better performance in tasks such as sequence-to-sequence modeling, language translation, and causal inference compared to Recurrent Neural Networks (RNN). These models have been trained on large amounts of data, enabling them to handle contextually sensitive inputs effectively. They also possess a larger number of parameters than simpler models like LSTMs or GRUs, allowing them to capture more intricate relationships between input and output sequences. The detailed discussion on how these models achieve state-of-the-art results across various benchmarks, including those relevant to your questions about comparing RNNs and Transformers for long-context memory, would be too extensive and would require an actual paper or document for a comprehensive comparison. Given that no specific evaluation details were provided for this task, it's clear that you do not have the required evidence at hand. Therefore, based solely on available information, I must conclude that my knowledge is limited to the absence of the requested evidence.
  Ticks used: 3
  Knowledge         : demo mode — writes disabled
════════════════════════════════════════════════════════════

Transformer achieves better performance in tasks such as sequence-to-sequence modeling, language translation, and causal inference compared to Recurrent Neural Networks (RNN). These models have been trained on large amounts of data, enabling them to handle contextually sensitive inputs effectively. They also possess a larger number of parameters than simpler models like LSTMs or GRUs, allowing them to capture more intricate relationships between input and output sequences. The detailed discussion on how these models achieve state-of-the-art results across various benchmarks, including those relevant to your questions about comparing RNNs and Transformers for long-context memory, would be too extensive and would require an actual paper or document for a comprehensive comparison. Given that no specific evaluation details were provided for this task, it's clear that you do not have the required evidence at hand. Therefore, based solely on available information, I must conclude that my knowledge is limited to the absence of the requested evidence.
```

---

## Design notes

**Two models, one pipeline.** GPT-2 is small and fast — it drives the loop cheaply. Qwen handles the final answer where instruction-following and evidence grounding matter. The two models never share weights.

**Mem tokens are not a retrieval store.** They are learnable attention bias vectors that persist only for the duration of one run. They let the model carry soft context between ticks without storing facts. Persistent facts go in the JSONL knowledge store.

**ToolFloor prevents routing collapse.** Without it, the router tends to always pick `decoder` because untrained weights favour it. The formula `score = raw × (1 − x) + x` (x = 0.05) guarantees a minimum push toward tool/rag/mem branches.

**Demo mode never writes knowledge.** All writes to the knowledge store are disabled when running without a trained checkpoint. The knowledge base is human-curated in demo mode.
