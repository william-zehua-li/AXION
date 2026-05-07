# Shared decoder layer — a single transformer decoder block reused every loop tick.
# The same weights are shared across all ticks (weight tying), keeping the model small.
#
# Why context is prepended, not cross-attended:
#   GPT-2 (and most decoder-only backbones) have no cross-attention mechanism.
#   Passing encoder_hidden_states to a GPT-2 block is silently ignored.
#   Instead we prepend the projected context tokens to the sequence so the
#   backbone's self-attention naturally attends over both context and hidden state.
#   After the layer runs, we slice off the context prefix and keep only the
#   hidden_state portion — the backbone has already mixed the two.
#
# Pretrained-friendly initialisation:
#   context_proj  → identity matrix  : context passes through unchanged initially
#   ffn output    → zero weights     : FFN contributes nothing initially (pure residual)
#   Both will diverge from these values as training progresses.
#
# Per-tick flow:
#   context_proj(context)
#   → prepend to hidden_state  →  [ctx | hidden]
#   → backbone layer (self-attn over full seq + mem bias in mask)
#   → slice [ctx_len:]
#   → residual + LayerNorm
#   → FFN  →  residual + LayerNorm
#   → return updated hidden state

import torch
import torch.nn as nn


class SharedDecoder(nn.Module):
    def __init__(self, backbone_layer: nn.Module, hidden_dim: int):
        # backbone_layer : one pretrained transformer block; weights shared across all ticks
        # hidden_dim     : must match the backbone's hidden dimension (e.g. 768 for GPT-2)
        super().__init__()

        self.layer = backbone_layer

        # Projects variable-length context (tool output / RAG chunks / mem signal)
        # into hidden_dim so it can be prepended as soft context tokens.
        # Initialised as identity: on first run context passes through unchanged,
        # preserving the pretrained signal.
        self.context_proj = nn.Linear(hidden_dim, hidden_dim)
        nn.init.eye_(self.context_proj.weight)
        nn.init.zeros_(self.context_proj.bias)

        # Applied after backbone + residual to stabilise the representation
        self.norm = nn.LayerNorm(hidden_dim)

        # Small FFN gives the layer extra capacity to integrate new context
        # after each tick; 4× expansion follows standard transformer convention.
        # Output layer initialised to zeros: FFN is a no-op on first run,
        # growing its influence only as training adjusts its weights.
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        nn.init.zeros_(self.ffn[2].weight)
        nn.init.zeros_(self.ffn[2].bias)

        self.ffn_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        hidden_state: torch.Tensor,         # (batch, seq_len, hidden_dim)
        context: torch.Tensor,              # (batch, context_len, hidden_dim)
        mem_bias: torch.Tensor = None,      # (batch, 1, seq_len, seq_len) or None
    ) -> torch.Tensor:

        # 1. Project context into the same space as hidden_state
        ctx = self.context_proj(context)            # (batch, context_len, hidden_dim)
        ctx_len = ctx.shape[1]

        # 2. Prepend context so self-attention can attend over [context | hidden_state]
        full_seq = torch.cat([ctx, hidden_state], dim=1)

        # 3. Build additive attention mask; inject mem bias into hidden↔hidden quadrant
        attn_mask = self._build_mask(full_seq, ctx_len, mem_bias)

        # 4. Run the shared backbone layer over the full concatenated sequence
        out = self.layer(full_seq, attention_mask=attn_mask)
        if isinstance(out, tuple):
            out = out[0]                            # HF layers return (hidden, present, ...)

        # 5. Discard the context prefix — keep only the updated hidden_state portion
        updated = out[:, ctx_len:, :]               # (batch, seq_len, hidden_dim)

        # 6. Residual connection + layer norm
        updated = self.norm(hidden_state + updated)

        # 7. FFN with its own residual + layer norm
        #    On first run ffn output is zero → pure residual (no corruption of pretrained signal)
        updated = self.ffn_norm(updated + self.ffn(updated))

        return updated                              # (batch, seq_len, hidden_dim)

    def _build_mask(
        self,
        full_seq: torch.Tensor,
        ctx_len: int,
        mem_bias: torch.Tensor,
    ) -> torch.Tensor:
        batch, total_len, _ = full_seq.shape
        mask = torch.zeros(batch, 1, total_len, total_len, device=full_seq.device)
        if mem_bias is not None:
            seq_len = total_len - ctx_len
            mask[:, :, ctx_len:, ctx_len:] += mem_bias[:, :, :seq_len, :seq_len]
        return mask
