import torch


class StopCheck:
    def __init__(self, max_ticks: int, confidence_threshold: float = 0.9):
        self.max_ticks            = max_ticks
        self.confidence_threshold = confidence_threshold

    def check(self, tick: int, decoder_output, confidence_score: float) -> bool:
        # Condition 1: hard ceiling
        if tick >= self.max_ticks - 1:
            return True

        # Condition 2: learned confidence head says "done"
        if confidence_score >= self.confidence_threshold:
            return True

        # Condition 3: hidden-state norm collapse (proxy for EOS / convergence).
        # If the shared decoder produced a near-zero update, the loop has stalled.
        if isinstance(decoder_output, torch.Tensor):
            norm = decoder_output.norm(dim=-1).mean().item()
            if norm < 1e-4:
                return True

        return False
