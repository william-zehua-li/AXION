# Top-level pipeline — single run() call drives the full forward pass end-to-end.
# Execution order matches structure.txt:
#   GlobalPrePro → GlobalRouter → LoopController → FinalPrePro → AnswerDecoder → UpdateGate
# Also owns the ephemeral objects (MemTokens, RecurrentState) and resets them each run.
#
# Two models are loaded:
#   backbone (GPT-2)            — encodes input, drives the recurrent loop
#   answer_decoder (Qwen-0.5B)  — instruction-tuned; generates the final answer from
#                                  an evidence-grounded prompt built from the loop trace
#
# Two generation modes:
#   demo mode  (no checkpoint) — AnswerDecoder.generate with aggregated=None;
#                                evidence prompt only, no loop embedding injection
#   trained mode (checkpoint)  — aggregated loop state passed for future conditioning

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from prepro import GlobalPrePro, LocalPrePro, FinalPrePro
from router import GlobalRouter, LocalRouter, RuleRouter
from memory import MemTokens, RecurrentState
from tools import ToolExecutor, RAG, calculate
from decoder import SharedDecoder, AnswerDecoder
from loop import LoopController, StopCheck
from knowledge import FileStore, UpdateGate

_ROOT        = os.path.join(os.path.dirname(__file__), "..")
_BACKBONE_CACHE = os.path.join(_ROOT, ".model_cache", "backbone")
_ANSWER_CACHE   = os.path.join(_ROOT, ".model_cache", "answer")


class Pipeline:
    def __init__(self, config):
        # config: instance of training/config.py

        # Shared pretrained backbone + tokenizer (GPT-2 — drives the loop).
        # Saved to .model_cache/backbone/ after first download.
        backbone_cache = os.path.normpath(_BACKBONE_CACHE)
        if os.path.isdir(backbone_cache) and os.listdir(backbone_cache):
            print(f"[Pipeline] Loading backbone from cache: {backbone_cache}")
            self.tokenizer = AutoTokenizer.from_pretrained(backbone_cache)
            self.backbone  = AutoModelForCausalLM.from_pretrained(backbone_cache)
        else:
            print(f"[Pipeline] Downloading backbone '{config.backbone_name}' (first run)…")
            self.tokenizer = AutoTokenizer.from_pretrained(config.backbone_name)
            self.backbone  = AutoModelForCausalLM.from_pretrained(config.backbone_name)
            os.makedirs(backbone_cache, exist_ok=True)
            self.tokenizer.save_pretrained(backbone_cache)
            self.backbone.save_pretrained(backbone_cache)
            print(f"[Pipeline] Backbone saved to {backbone_cache}")

        # GPT-2 has no pad token — set it to eos so generate() doesn't warn
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.backbone.eval()

        # Stage 1 — Global PrePro (encodes the raw input)
        self.global_prepro = GlobalPrePro(self.backbone, self.tokenizer)

        # Stage 2 — Routers
        # RuleRouter: regex-based, zero weights, used in demo mode.
        # GlobalRouter: learned, used only after a checkpoint is loaded.
        self.rule_router = RuleRouter()
        self.global_router = GlobalRouter(
            hidden_dim=config.backbone_dim,
            max_plan_len=config.max_ticks,
        )

        # Ephemeral state objects — reset before every run
        self.mem_tokens      = MemTokens(config.num_mem_tokens, config.backbone_dim)
        self.recurrent_state = RecurrentState(config.backbone_dim)

        # Persistent knowledge store — survives across runs
        self.file_store = FileStore(config.store_path, config.queue_path)

        # Loop internals
        local_prepro  = LocalPrePro(hidden_dim=config.backbone_dim,
                                    max_ticks=config.max_ticks,
                                    tool_floor_x=config.tool_floor_x)
        local_router  = LocalRouter(hidden_dim=config.backbone_dim)
        tool_executor = ToolExecutor(tools={"calculator": calculate})
        rag           = RAG(self.file_store)
        shared_decoder = SharedDecoder(
            backbone_layer=self.backbone.transformer.h[-1],
            hidden_dim=config.backbone_dim,
        )
        stop_check = StopCheck(config.max_ticks, config.stop_confidence_threshold)

        # Stage 3 — Recurrent loop
        self.loop = LoopController(
            local_prepro=local_prepro,
            local_router=local_router,
            tool_executor=tool_executor,
            rag=rag,
            mem_tokens=self.mem_tokens,
            shared_decoder=shared_decoder,
            state=self.recurrent_state,
            stop_check=stop_check,
        )

        # Stage 4 — Final PrePro
        self.final_prepro = FinalPrePro(hidden_dim=config.backbone_dim)

        # Stage 5 — Answer Decoder (Qwen — separate from GPT-2 backbone)
        # Owns its own model + tokenizer; saved to .model_cache/answer/
        self.answer_decoder = AnswerDecoder(
            model_name = config.answer_model_name,
            cache_dir  = os.path.normpath(_ANSWER_CACHE),
        )

        # Stage 6 — Post-run knowledge update
        self.update_gate = UpdateGate(
            self.file_store,
            config.novelty_threshold,
            config.confidence_threshold,
        )

        self.config = config

        # Tracks whether trained weights have been loaded.
        # False → demo mode (backbone.generate on raw input).
        # True  → trained mode (answer_decoder.generate on aggregated rep).
        self._checkpoint_loaded: bool = False

    # ------------------------------------------------------------------

    def run(self, raw_input: str, verbose: bool = True) -> dict:
        # Reset ephemeral state so each run starts clean
        self.mem_tokens.reset()
        self.recurrent_state.reset()

        if verbose:
            mode = "trained" if self._checkpoint_loaded else "demo"
            print(f"\n{'═'*60}")
            print(f"  Input : {raw_input!r}")
            print(f"  Mode  : {mode}")
            print(f"{'═'*60}")

        # 1. Encode + preprocess
        encoded = self.global_prepro.forward(raw_input)

        # 2. Coarse plan
        # Demo mode: rule-based router (no network, deterministic, no ML).
        # Trained mode: learned GlobalRouter uses the encoded representation.
        if not self._checkpoint_loaded:
            action_plan, _, query_type = self.rule_router.forward(raw_input)
            follow_plan = True
        else:
            action_plan, _ = self.global_router.forward(encoded)
            query_type = None
            follow_plan = False

        if verbose and query_type:
            print(f"  Query type : {query_type}")

        # 3. Recurrent loop — always runs; provides the tick trace and structure test
        final_hidden, loop_outputs = self.loop.run(
            encoded_input=encoded,
            action_plan=action_plan,
            max_ticks=self.config.max_ticks,
            verbose=verbose,
            follow_plan=follow_plan,
            raw_input=raw_input,
        )

        # 4. Aggregate loop outputs (runs always; zero-delta in demo mode = pass-through)
        aggregated = self.final_prepro.forward(final_hidden, loop_outputs)

        # 5. Generate answer
        # The prompt is built from loop_outputs (evidence + route trace).
        # Demo mode  : aggregated=None  → backbone.generate from token ids
        # Trained mode: aggregated=rep  → loop state prepended as soft embeddings
        answer = self.answer_decoder.generate(
            raw_input,
            loop_outputs   = loop_outputs,
            aggregated     = aggregated if self._checkpoint_loaded else None,
            max_new_tokens = self.config.max_new_tokens,
            temperature    = self.config.temperature,
        )

        # 6. Post-run knowledge update
        # Demo mode never writes or queues anything — the knowledge base is
        # human-curated.  Only trained mode may write inferred facts.
        if self._checkpoint_loaded:
            knowledge_updates = self.update_gate.evaluate(loop_outputs, answer)
        else:
            knowledge_updates = {"written": [], "rejected": [], "queued": [],
                                 "note": "demo mode — writes disabled"}

        if verbose:
            ku = knowledge_updates
            print(f"\n{'═'*60}")
            print(f"  Answer    : {answer}")
            print(f"  Ticks used: {len(loop_outputs)}")
            if ku.get("written"):
                print(f"  Knowledge written : {len(ku['written'])} item(s)")
            if ku.get("queued"):
                print(f"  Knowledge queued  : {len(ku['queued'])} item(s)")
            if ku.get("note"):
                print(f"  Knowledge         : {ku['note']}")
            print(f"{'═'*60}\n")

        return {
            "answer":            answer,
            "ticks_used":        len(loop_outputs),
            "loop_outputs":      loop_outputs,
            "knowledge_updates": knowledge_updates,
        }

    # ------------------------------------------------------------------

    def load_checkpoint(self, path: str) -> None:
        """
        Load trained weights for the custom modules.
        After this call the pipeline switches to trained mode automatically.
        """
        state = torch.load(path, map_location="cpu")
        trainable_modules = {
            "global_router": self.global_router,
            "local_prepro":  self.loop.local_prepro,
            "local_router":  self.loop.local_router,
            "context_proj":  self.loop.shared_decoder.context_proj,
            "mem_tokens":    self.mem_tokens,
            "final_prepro":  self.final_prepro,
        }
        loaded, missing = 0, []
        for name, param in state.items():
            placed = False
            for mod_name, mod in trainable_modules.items():
                prefix = mod_name + "."
                if name.startswith(prefix):
                    sub_key = name[len(prefix):]
                    mod.load_state_dict({sub_key: param}, strict=False)
                    loaded += 1
                    placed = True
                    break
            if not placed:
                missing.append(name)
        if missing:
            print(f"[load_checkpoint] {len(missing)} key(s) not matched: {missing[:5]}")
        print(f"[load_checkpoint] {loaded} tensors loaded — switching to trained mode.")
        self._checkpoint_loaded = True
