import torch


class LoopController:
    def __init__(self, local_prepro, local_router, tool_executor, rag,
                 mem_tokens, shared_decoder, state, stop_check):
        self.local_prepro   = local_prepro
        self.local_router   = local_router
        self.tool_executor  = tool_executor
        self.rag            = rag
        self.mem_tokens     = mem_tokens
        self.shared_decoder = shared_decoder
        self.state          = state
        self.stop_check     = stop_check

    def run(self, encoded_input: torch.Tensor, action_plan: list,
            max_ticks: int, verbose: bool = False, follow_plan: bool = False,
            raw_input: str = ""):
        """
        encoded_input : (batch, seq_len, hidden_dim) from GlobalPrePro
        action_plan   : ordered list of step-type hints from GlobalRouter
        max_ticks     : hard upper bound; may exit earlier via StopCheck
        verbose       : if True, print a trace line for every tick

        Returns
        -------
        final_hidden : (batch, seq_len, hidden_dim) — state after the last tick
        loop_outputs : list[dict] — one record per tick for FinalPrePro / Evaluator
        """
        # Seed recurrent state with the encoded input so the first tick has context.
        if self.state.get() is None:
            self.state.update(encoded_input)

        loop_outputs: list[dict] = []
        plan_len  = len(action_plan)
        stop_reason = ""

        if verbose:
            plan_str = " → ".join(action_plan) if action_plan else "(empty)"
            print(f"\n  {'─'*56}")
            print(f"  Global plan ({len(action_plan)} steps): {plan_str}")
            print(f"  {'─'*56}")

        for tick in range(max_ticks):
            plan_hint = action_plan[tick] if tick < plan_len else None

            # ── a. Local preprocessing ─────────────────────────────────────
            mem_bias              = self.mem_tokens.read()
            local_rep, tool_score = self.local_prepro.forward(
                self.state.get(), mem_bias, tick
            )

            # ── b. Route ───────────────────────────────────────────────────
            branch, metadata = self.local_router.forward(
                local_rep, plan_hint,
                tool_score=tool_score,
                follow_plan=follow_plan,
                tick=tick,
                plan_len=plan_len,
                raw_input=raw_input,
            )
            confidence = float(metadata.get("confidence", 0.0))

            # ── c. Execute branch; build context tensor ────────────────────
            tick_record: dict = {"tick": tick, "branch": branch}
            branch_detail = ""

            if branch == "tool":
                tool_name = metadata.get("tool_name", "calculator")
                tool_args = metadata.get("tool_args", {})
                result    = self.tool_executor.execute(tool_name, tool_args)
                tick_record["tool_name"]    = tool_name
                tick_record["tool_args"]    = tool_args
                tick_record["tool_output"]  = result.get("output")
                tick_record["tool_success"] = result.get("success", False)
                tick_record["tool_error"]   = result.get("error")
                context = encoded_input
                status  = "✓" if tick_record["tool_success"] else "✗"
                expr    = tool_args.get("expression", "")
                branch_detail = (
                    f"{tool_name}({expr!r}) = {tick_record['tool_output']} {status}"
                    if expr else f"{tool_name} {status}"
                )

            elif branch == "rag":
                query   = metadata.get("query", "")
                chunks  = self.rag.retrieve(query, top_k=metadata.get("top_k", 5))
                rag_hit = len(chunks) > 0
                tick_record["rag_chunks"] = chunks
                tick_record["rag_hit"]    = rag_hit
                context = encoded_input
                branch_detail = f"{len(chunks)} chunk(s) {'✓' if rag_hit else '✗ (store empty)'}"

            elif branch == "mem":
                self.mem_tokens.write(local_rep)
                bias = self.mem_tokens.read()
                context = bias.unsqueeze(0) if (isinstance(bias, torch.Tensor)
                                                 and bias.dim() == 2) else (
                          bias if isinstance(bias, torch.Tensor) else encoded_input)
                branch_detail = "bias updated"

            else:  # "decoder"
                context = self.state.get() if self.state.get() is not None else encoded_input
                branch_detail = "refining representation"

            # ── d. Shared decoder pass ─────────────────────────────────────
            attn_bias = None
            raw_bias  = self.mem_tokens.read()
            if raw_bias is not None and isinstance(raw_bias, torch.Tensor):
                h      = self.state.get()
                scores = torch.einsum("bih,th->bti", h, raw_bias).sum(dim=1)
                attn_bias = scores.unsqueeze(1).unsqueeze(2).expand(
                    -1, 1, h.shape[1], h.shape[1]
                )
            new_hidden = self.shared_decoder.forward(
                self.state.get(), context, mem_bias=attn_bias
            )

            # ── e. Update recurrent state ──────────────────────────────────
            self.state.update(new_hidden)
            tick_record["hidden"] = new_hidden.detach()

            # ── f. Stop check ──────────────────────────────────────────────
            loop_outputs.append(tick_record)
            should_stop = self.stop_check.check(tick, new_hidden, confidence)

            # ── g. Verbose trace ───────────────────────────────────────────
            if verbose:
                hint_str  = plan_hint if plan_hint else "—"
                stop_tag  = ""
                if should_stop:
                    if tick >= max_ticks - 1:
                        stop_tag = "  ← STOP (max ticks)"
                    elif confidence >= self.stop_check.confidence_threshold:
                        stop_tag = "  ← STOP (confident)"
                    else:
                        stop_tag = "  ← STOP (collapsed)"

                print(
                    f"  Tick {tick}  hint={hint_str:<8}  branch={branch:<8}"
                    f"  tool_score={tool_score:.3f}  conf={confidence:.3f}"
                    + (f"  │ {branch_detail}" if branch_detail else "")
                    + stop_tag
                )

            if should_stop:
                break

        if verbose:
            print(f"  {'─'*56}")

        return self.state.get(), loop_outputs
