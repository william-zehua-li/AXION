import torch
import torch.nn as nn


# Ordered plan vocabulary — each label maps to one local-router branch.
_PLAN_VOCAB = ["rag", "tool", "mem", "decoder"]
_LABEL_TO_IDX = {label: i for i, label in enumerate(_PLAN_VOCAB)}


class GlobalRouter(nn.Module):
    """
    Coarse planning stage — runs once after GlobalPrePro.

    Produces:
      action_plan     : ordered list of step-type labels guiding the loop
      estimated_ticks : soft estimate of how many ticks the loop will need

    Architecture:
      1. Mean-pool the encoded sequence to a single context vector.
      2. A small MLP predicts a distribution over plan-step types for each
         position in the plan (up to max_plan_len steps).
      3. A separate scalar head predicts the estimated tick count.
    """

    _MAX_PLAN_LEN = 8   # matches Config.max_ticks

    def __init__(self, hidden_dim: int = 768, max_plan_len: int = _MAX_PLAN_LEN):
        super().__init__()
        self.max_plan_len = max_plan_len
        num_labels = len(_PLAN_VOCAB)

        # Shared encoder: pool → project to a fixed planning space.
        self.context_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

        # Per-position step classifier: predicts which branch for each plan slot.
        # Input: [context (H) | position_embed (H)] → logits over branches.
        self.step_head = nn.Linear(hidden_dim * 2, num_labels)

        # Scalar head: estimates how many ticks will be needed (1..max_plan_len).
        self.tick_head = nn.Linear(hidden_dim, 1)

        # Positional embeddings for each plan slot (learned).
        self.pos_embed = nn.Embedding(max_plan_len, hidden_dim)

    # ------------------------------------------------------------------

    def forward(self, encoded_input: torch.Tensor):
        """
        encoded_input : (batch, seq_len, hidden_dim)

        Returns
        -------
        action_plan     : list[str]  — e.g. ["rag", "decoder", "decoder", ...]
        estimated_ticks : int        — how many ticks the loop is expected to use
        """
        # 1. Pool sequence to context vector: (batch, hidden_dim)
        context = encoded_input.mean(dim=1)
        context = self.context_proj(context)            # (B, H)

        # 2. Predict a step type for each plan position.
        positions = torch.arange(self.max_plan_len, device=encoded_input.device)
        pos_embs  = self.pos_embed(positions)           # (max_plan_len, H)

        # Broadcast context across plan positions.
        ctx_expanded = context.unsqueeze(1).expand(
            -1, self.max_plan_len, -1
        )                                               # (B, max_plan_len, H)
        combined = torch.cat(
            [ctx_expanded, pos_embs.unsqueeze(0).expand(context.size(0), -1, -1)],
            dim=-1,
        )                                               # (B, max_plan_len, 2H)

        step_logits = self.step_head(combined)          # (B, max_plan_len, num_labels)
        step_ids    = step_logits.argmax(dim=-1)        # (B, max_plan_len)

        # 3. Estimate tick count (clamp to [1, max_plan_len]).
        tick_raw   = self.tick_head(context).squeeze(-1)    # (B,)
        tick_count = int(tick_raw.sigmoid().item() * self.max_plan_len)
        tick_count = max(1, min(tick_count, self.max_plan_len))

        # 4. Build the plan list for batch element 0 (pipeline runs one at a time).
        plan_ids   = step_ids[0, :tick_count].tolist()
        action_plan = [_PLAN_VOCAB[idx] for idx in plan_ids]

        return action_plan, tick_count
