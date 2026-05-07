import torch
import torch.nn as nn


class FinalPrePro(nn.Module):
    """
    Stage 4 — aggregates loop outputs before the Answer Decoder.

    Responsibilities:
      1. Take the final recurrent hidden state (after the last tick).
      2. Optionally fuse a summary of intermediate loop outputs
         (mean of all per-tick hidden states, if available).
      3. Apply a learned projection + LayerNorm so the Answer Decoder
         always receives a clean, stable input regardless of tick count.

    Pretrained-friendly initialisation:
      Both projection layers are initialised to zero weights and zero bias.
      Combined with the residual connection (out = final_hidden + delta),
      this means on the first run delta = GELU(0) = 0, so the module is
      a pure identity pass-through: the pretrained hidden states flow to
      the Answer Decoder completely unchanged.
      Training will grow the projections away from zero over time.
    """

    def __init__(self, hidden_dim: int = 768):
        super().__init__()

        # Projects fused representation (final + history mean) back to hidden_dim.
        self.proj_fused  = nn.Linear(hidden_dim * 2, hidden_dim)
        # Projects the final hidden state alone when no tick history is available.
        self.proj_single = nn.Linear(hidden_dim, hidden_dim)

        # Zero-init both projections → delta starts at zero → pure residual on first run
        nn.init.zeros_(self.proj_fused.weight)
        nn.init.zeros_(self.proj_fused.bias)
        nn.init.zeros_(self.proj_single.weight)
        nn.init.zeros_(self.proj_single.bias)

        self.norm = nn.LayerNorm(hidden_dim)
        self.act  = nn.GELU()

    def forward(
        self,
        final_hidden_state: torch.Tensor,
        loop_outputs: list,
    ) -> torch.Tensor:
        """
        final_hidden_state : (batch, seq_len, hidden_dim)
        loop_outputs       : list[dict] — one dict per tick from LoopController

        Returns
        -------
        aggregated : (batch, seq_len, hidden_dim) ready for AnswerDecoder
        """
        # Collect any hidden state tensors stored in per-tick records.
        tick_hiddens = [
            o["hidden"]
            for o in loop_outputs
            if "hidden" in o and isinstance(o["hidden"], torch.Tensor)
        ]

        if tick_hiddens:
            # Mean-pool tick hiddens, fuse with the final state.
            history = torch.stack(tick_hiddens, dim=0).mean(dim=0)     # (B, S, H)
            fused   = torch.cat([final_hidden_state, history], dim=-1) # (B, S, 2H)
            delta   = self.act(self.proj_fused(fused))                 # (B, S, H)
        else:
            delta   = self.act(self.proj_single(final_hidden_state))   # (B, S, H)

        # Residual: on first run delta == 0, so output == norm(final_hidden_state)
        # which is the clean pretrained representation.
        out = self.norm(final_hidden_state + delta)
        return out
