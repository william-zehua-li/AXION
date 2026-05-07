import torch
import torch.nn as nn


class MemTokens(nn.Module):
    """
    Small set of learnable attention-bias vectors, reset each inference run.

    Formula:  bias = base_bias + accumulated_delta
      - base_bias   : learned parameter, updated by the optimizer across runs
      - accumulated_delta : zeroed at the start of every run, updated in-place
                            by write() calls during the recurrent loop
    """

    def __init__(self, num_tokens: int, dim: int):
        super().__init__()
        self.num_tokens = num_tokens
        self.dim        = dim
        # Learned base bias — trained, persists across inference runs.
        self.base_bias  = nn.Parameter(torch.zeros(num_tokens, dim))
        # Projection that maps a signal vector onto the bias space.
        self.write_proj = nn.Linear(dim, dim, bias=False)
        # Per-run accumulated delta — not a Parameter; reset each run.
        self.register_buffer("_delta", torch.zeros(num_tokens, dim), persistent=False)

    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear the per-run delta back to zero before each new inference run."""
        self._delta.zero_()

    def read(self) -> torch.Tensor:
        """Return current bias tensor: (num_tokens, dim)."""
        return self.base_bias + self._delta

    def write(self, signal: torch.Tensor) -> None:
        """
        Update the accumulated delta given a signal from the current loop tick.

        signal: (batch, seq_len, dim) or (seq_len, dim) or (dim,)
          Aggregated to a single (dim,) vector, projected, then added to every
          bias slot via a gated accumulation so later ticks can refine earlier ones.
        """
        # Reduce signal to a single (dim,) update vector.
        s = signal
        while s.dim() > 1:
            s = s.mean(dim=0)               # collapse batch and seq dims

        update = self.write_proj(s)         # (dim,)
        # Broadcast update to all num_tokens slots and accumulate.
        self._delta = self._delta + update.unsqueeze(0)  # (num_tokens, dim)
