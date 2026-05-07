# ToolFloor — minimum exploration guarantee for tool and memory usage.
#
# Problem it solves:
#   Without a floor, the LocalRouter can learn to never use tools or memory
#   if it finds a shortcut through the decoder branch alone. This collapses
#   the routing diversity and makes those branches dead during training.
#
# How it works:
#   raw_score is a number in [0, 1] produced by LocalPrePro's tool-readiness head.
#   It represents how strongly the current tick's representation calls for
#   using an external resource (tool or memory).
#
#   The floor transformation squishes [0, 1] → [x, 1]:
#
#       final_score = raw_score * (1 - x) + x
#
#   With x = 0.05:
#     raw = 0.0  →  final = 0.05   (floor: never fully suppressed)
#     raw = 0.5  →  final = 0.525
#     raw = 1.0  →  final = 1.0    (ceiling unchanged)
#
#   This means the model always has at least a 5% push toward considering
#   tools/memory every tick, preventing routing collapse.
#
# x must be small (≤ 0.05) — large values would force tool use even when
# the model has clearly learned not to need it.


class ToolFloor:
    # Default floor constant — keep at or below 0.05
    X: float = 0.05

    def __init__(self, x: float = X):
        if not (0.0 <= x <= 0.05):
            raise ValueError(f"x must be in [0, 0.05], got {x}")
        self.x = x

    def apply(self, raw_score: float) -> float:
        # Applies: final = raw * (1 - x) + x
        # raw_score must be in [0, 1]; returns a value in [x, 1]
        raw_score = max(0.0, min(1.0, raw_score))       # clamp input
        return raw_score * (1.0 - self.x) + self.x

    def apply_tensor(self, raw_scores):
        # Same formula applied element-wise to a torch tensor.
        # Returns a tensor of the same shape with values in [x, 1].
        return raw_scores * (1.0 - self.x) + self.x
