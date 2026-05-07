# Answer decoder — final generation stage, runs once after the loop exits.
#
# Uses a separate instruction-tuned model (Qwen2.5-Instruct by default) that
# is distinct from the GPT-2 backbone driving the recurrent loop.  Qwen is
# better suited to evidence-grounded instruction-following than GPT-2.
#
# The decoder builds a chat-template prompt from three sources extracted out
# of the loop's tick records:
#   • evidence   — RAG chunks + successful tool outputs
#   • route_trace — tick-by-tick branch sequence (what AXION actually did)
#   • query      — the original user question
#
# Model loading:
#   First call downloads Qwen from HuggingFace and saves it to cache_dir.
#   Every subsequent call loads from disk — no network after the first run.

import os
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

_SYSTEM_PROMPT = (
    "You are AXION's output decoder.\n\n"
    "Answer the user question using only the evidence below.\n"
    "If the evidence is insufficient, say you do not know."
)


class AnswerDecoder(nn.Module):
    def __init__(self, model_name: str, cache_dir: str):
        """
        model_name : HuggingFace model ID, e.g. 'Qwen/Qwen2.5-0.5B-Instruct'
        cache_dir  : local directory to save/load the model (avoids re-downloading)
        """
        super().__init__()

        cache_path = os.path.normpath(cache_dir)
        if os.path.isdir(cache_path) and os.listdir(cache_path):
            print(f"[AnswerDecoder] Loading {model_name} from cache: {cache_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                cache_path, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                cache_path, trust_remote_code=True)
        else:
            print(f"[AnswerDecoder] Downloading {model_name} (first run)…")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, trust_remote_code=True)
            os.makedirs(cache_path, exist_ok=True)
            self.tokenizer.save_pretrained(cache_path)
            self.model.save_pretrained(cache_path)
            print(f"[AnswerDecoder] Saved to {cache_path}")

        self.model.eval()

        # Resolve EOS / pad tokens — Qwen2.5 sets these correctly but guard anyway.
        eos = self.tokenizer.eos_token_id
        if eos is None:
            eos = self.tokenizer.convert_tokens_to_ids("<|endoftext|>")
        self.eos_token_id: int = eos if eos is not None else -1

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.eos_token_id

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        raw_input:      str,
        loop_outputs:   list        = None,
        aggregated:     torch.Tensor | None = None,   # reserved for trained mode
        max_new_tokens: int   = 256,
        temperature:    float = 0.7,
        top_p:          float = 0.9,
        top_k:          int   = 50,
        repetition_penalty: float = 1.1,
    ) -> str:
        """
        raw_input   : original query string
        loop_outputs: tick records from LoopController — evidence + route extracted here
        aggregated  : unused in demo; reserved for future trained-mode conditioning
        """
        evidence    = self._extract_evidence(loop_outputs or [])
        route_trace = self._build_route_trace(loop_outputs or [])

        user_content = (
            f"Question:\n{raw_input.strip()}\n\n"
            f"Evidence:\n{evidence}\n\n"
            f"AXION route:\n{route_trace}\n\n"
            f"Final answer:"
        )

        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": user_content},
        ]

        # apply_chat_template adds the model-specific special tokens and the
        # generation-prompt marker so the model knows to start answering.
        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize            = False,
            add_generation_prompt = True,
        )

        device = next(self.model.parameters()).device
        enc = self.tokenizer(
            prompt_text,
            return_tensors = "pt",
            truncation     = True,
            max_length     = 2048,
        )
        input_ids      = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        prompt_len     = input_ids.shape[1]

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                attention_mask     = attention_mask,
                max_new_tokens     = max_new_tokens,
                do_sample          = True,
                temperature        = temperature,
                top_p              = top_p,
                top_k              = top_k,
                repetition_penalty = repetition_penalty,
                pad_token_id       = self.tokenizer.pad_token_id,
                eos_token_id       = self.eos_token_id if self.eos_token_id >= 0 else None,
            )

        new_ids = output_ids[0][prompt_len:]
        return self.tokenizer.decode(new_ids, skip_special_tokens=True).strip()

    # ------------------------------------------------------------------
    # Prompt-building helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_evidence(loop_outputs: list) -> str:
        """
        Collect retrieved / computed evidence from the loop tick records:
          - RAG chunks  (branch == 'rag')
          - Tool output (branch == 'tool', success == True)
        Returns a numbered list, or '(none)' when no evidence was gathered.
        """
        items: list[str] = []

        for record in loop_outputs:
            for chunk in record.get("rag_chunks", []):
                if isinstance(chunk, str) and chunk.strip():
                    items.append(chunk.strip())

            if record.get("tool_success") and record.get("tool_output"):
                out = str(record["tool_output"]).strip()
                if out:
                    items.append(f"[tool] {out}")

        if not items:
            return "(none)"

        # Deduplicate, preserve order
        seen: set[str] = set()
        unique: list[str] = []
        for it in items:
            if it not in seen:
                seen.add(it)
                unique.append(it)

        return "\n".join(f"{i+1}. {txt}" for i, txt in enumerate(unique))

    @staticmethod
    def _build_route_trace(loop_outputs: list) -> str:
        """
        Summarise the tick-by-tick branch sequence, e.g.:
          tick 0 → rag     — 2 chunk(s) ✓
          tick 1 → decoder
        """
        if not loop_outputs:
            return "(no ticks)"

        lines: list[str] = []
        for record in loop_outputs:
            tick   = record.get("tick", "?")
            branch = record.get("branch", "?")
            detail = ""

            if branch == "rag":
                n      = len(record.get("rag_chunks", []))
                hit    = record.get("rag_hit", False)
                detail = f" — {n} chunk(s) {'✓' if hit else '✗ (store empty)'}"
            elif branch == "tool":
                name    = record.get("tool_name", "")
                success = record.get("tool_success", False)
                expr    = (record.get("tool_args") or {}).get("expression", "")
                output  = record.get("tool_output", "")
                if expr and output:
                    detail = f" — {name}({expr}) = {output} {'✓' if success else '✗'}"
                else:
                    detail = f" — {name} {'✓' if success else '✗'}"
            elif branch == "mem":
                detail = " — bias updated"

            lines.append(f"  tick {tick} → {branch}{detail}")

        return "\n".join(lines)
