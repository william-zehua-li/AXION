import torch
import torch.nn as nn

from tools.calculator import extract_expression


_BRANCHES   = ["tool", "rag", "mem", "decoder"]
_BRANCH_IDX = {b: i for i, b in enumerate(_BRANCHES)}
_HINT_VOCAB = ["rag", "tool", "mem", "decoder"]


class LocalRouter(nn.Module):
    """
    Per-tick branch selector — runs inside every loop iteration.

    Two modes:

    follow_plan=True  (demo / rule-based mode)
      Ignores the ML head entirely and executes the plan hint directly.
      Branch = plan_hint.  Confidence grows linearly with tick position
      ((tick+1)/plan_len), reaching 1.0 on the final planned step so
      StopCheck exits the loop exactly when the plan is complete.

    follow_plan=False (trained mode)
      Uses the learned branch_head + tool_score bias to pick a branch.
      The plan hint is a soft suggestion (+2.0 logit bias), not a hard rule.
    """

    def __init__(self, hidden_dim: int = 768):
        super().__init__()
        num_branches = len(_BRANCHES)
        num_hints    = len(_HINT_VOCAB) + 1   # +1 for "no hint" / None

        self.branch_head = nn.Sequential(
            nn.Linear(hidden_dim + num_hints, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_branches),
        )

        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        self.query_proj = nn.Linear(hidden_dim, hidden_dim)

    # ------------------------------------------------------------------

    def forward(self, local_representation: torch.Tensor, plan_hint,
                tool_score: float = 0.05, follow_plan: bool = False,
                tick: int = 0, plan_len: int = 0, raw_input: str = ""):
        """
        local_representation : (batch, seq_len, hidden_dim) from LocalPrePro
        plan_hint            : str label from the global plan, or None
        tool_score           : float in [x, 1] from ToolFloor
        follow_plan          : if True, skip the ML head and execute plan_hint exactly
        tick                 : current loop tick (used for rule-based confidence)
        plan_len             : total plan length (used for rule-based confidence)

        Returns
        -------
        branch   : str in {'tool', 'rag', 'mem', 'decoder'}
        metadata : dict — confidence, tool_score, and branch-specific keys
        """
        pooled = local_representation.mean(dim=1)   # (B, H) — used by both modes

        if follow_plan:
            return self._strict(plan_hint, tool_score, pooled, tick, plan_len,
                                raw_input)

        return self._learned(pooled, plan_hint, tool_score, raw_input)

    # ------------------------------------------------------------------
    # Strict / demo mode
    # ------------------------------------------------------------------

    def _strict(self, plan_hint, tool_score: float, pooled: torch.Tensor,
                tick: int = 0, plan_len: int = 0, raw_input: str = ""):
        """
        Execute the plan hint directly with no ML involvement.

        Confidence is computed from position in the plan so StopCheck fires
        naturally at the last planned step without running extra ticks:
          confidence = (tick + 1) / plan_len   → reaches 1.0 on the final step
          confidence = 1.0                      → when past the end of the plan
        The default StopCheck threshold is 0.9, so the loop exits on the tick
        where confidence first crosses that threshold (the last planned step).
        """
        branch = plan_hint if plan_hint in _BRANCHES else "decoder"

        if plan_hint is None or plan_len == 0:
            # Past the end of the plan — signal done immediately.
            confidence = 1.0
        else:
            # Linearly grows from 1/plan_len → 1.0 as the plan progresses.
            confidence = (tick + 1) / plan_len

        metadata: dict = {"confidence": confidence, "tool_score": tool_score}
        self._add_branch_meta(metadata, branch, pooled, raw_input)
        return branch, metadata

    # ------------------------------------------------------------------
    # Learned / trained mode
    # ------------------------------------------------------------------

    def _learned(self, pooled: torch.Tensor, plan_hint, tool_score: float,
                 raw_input: str = ""):
        """ML-based branch selection with plan-hint soft bias."""
        hint_vec      = self._encode_hint(plan_hint, pooled.device)
        hint_vec      = hint_vec.expand(pooled.size(0), -1)
        branch_input  = torch.cat([pooled, hint_vec], dim=-1)
        branch_logits = self.branch_head(branch_input)

        # tool_score nudge prevents routing collapse onto decoder
        for b in ("tool", "rag", "mem"):
            branch_logits[:, _BRANCH_IDX[b]] += tool_score

        # Plan hint gets a stronger push to respect the global plan
        if plan_hint in _BRANCH_IDX:
            branch_logits[:, _BRANCH_IDX[plan_hint]] += 2.0

        branch_idx = branch_logits.argmax(dim=-1)[0].item()
        branch     = _BRANCHES[int(branch_idx)]

        confidence = self.confidence_head(pooled)[0, 0].item()
        metadata: dict = {"confidence": confidence, "tool_score": tool_score}
        self._add_branch_meta(metadata, branch, pooled, raw_input)
        return branch, metadata

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _add_branch_meta(self, metadata: dict, branch: str,
                         pooled: torch.Tensor, raw_input: str = "") -> None:
        if branch == "rag":
            metadata["query"] = self._hidden_to_query_hint(pooled)
            metadata["top_k"] = 5
        elif branch == "tool":
            expr = extract_expression(raw_input) if raw_input else ""
            metadata["tool_name"] = "calculator"
            metadata["tool_args"] = {"expression": expr}

    def _encode_hint(self, hint, device: torch.device) -> torch.Tensor:
        num_hints = len(_HINT_VOCAB) + 1
        vec = torch.zeros(1, num_hints, device=device)
        if hint in _HINT_VOCAB:
            vec[0, _HINT_VOCAB.index(hint)] = 1.0
        else:
            vec[0, -1] = 1.0
        return vec

    @staticmethod
    def _hidden_to_query_hint(pooled: torch.Tensor) -> str:
        top_dims = pooled[0].abs().topk(5).indices.tolist()
        return " ".join(f"dim{d}" for d in top_dims)
