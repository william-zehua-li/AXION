import torch


class RecurrentState:
    """
    Mutable container for the hidden state passed between loop ticks.

    Not an nn.Module — it holds no trainable parameters and owns no
    computation; it is purely a typed carrier so the loop controller
    and shared decoder don't access raw tensors by convention.
    """

    def __init__(self, dim: int):
        self.dim    = dim
        self._state: torch.Tensor | None = None

    def reset(self) -> None:
        """Clear state to None; the loop controller seeds it from encoded_input."""
        self._state = None

    def get(self) -> torch.Tensor | None:
        """Return the current hidden state tensor, or None if not yet seeded."""
        return self._state

    def update(self, new_state: torch.Tensor) -> None:
        """Overwrite the hidden state with the shared decoder's output for this tick."""
        self._state = new_state
