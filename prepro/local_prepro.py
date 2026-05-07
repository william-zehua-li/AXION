import torch
import torch.nn as nn

from .tool_floor import ToolFloor


class LocalPrePro(nn.Module):
    """
    Per-tick preprocessing — runs at the start of every recurrent loop iteration.

    Combines three signals into a single conditioned representation:
      1. Current recurrent hidden state  (what we know so far)
      2. Mem token bias                  (where to focus attention this run)
      3. Tick position awareness         (sinusoidal + learned tick embedding)

    Also produces a tool_score: a floor-adjusted float in [x, 1] that tells
    the LocalRouter how strongly the current state calls for using an external
    resource (tool or memory).  Formula:  tool_score = raw * (1 - x) + x

    Returns
    -------
    conditioned : (batch, seq_len, hidden_dim)
    tool_score  : float in [x, 1]
    """

    def __init__(self, hidden_dim: int = 768, max_ticks: int = 8,
                 tool_floor_x: float = 0.05):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Learned tick embedding: encodes which iteration we are on.
        # Initialised with very small weights (std=0.01) so tick position adds
        # only negligible noise to the pretrained hidden state before training.
        self.tick_embed = nn.Embedding(max_ticks, hidden_dim)
        nn.init.normal_(self.tick_embed.weight, mean=0.0, std=0.01)

        # Projects mem bias (num_tokens, hidden_dim) → (1, 1, hidden_dim)
        # so it can be added to the hidden state sequence.
        self.mem_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # Final gate: decides how much of the mem signal to inject.
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid(),
        )

        self.norm = nn.LayerNorm(hidden_dim)

        # Tool-readiness head: pools the conditioned rep → scalar raw score.
        # The score is then passed through ToolFloor before being returned.
        self.tool_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid(),           # raw score in [0, 1]
        )

        self.tool_floor = ToolFloor(x=tool_floor_x)

    def forward(
        self,
        hidden_state: torch.Tensor,
        mem_bias: torch.Tensor,
        tick: int,
    ):
        """
        hidden_state : (batch, seq_len, hidden_dim) — current recurrent state
        mem_bias     : (num_tokens, hidden_dim) or (batch, num_tokens, hidden_dim)
        tick         : int — current loop iteration index

        Returns
        -------
        conditioned : (batch, seq_len, hidden_dim)
        tool_score  : float in [tool_floor_x, 1.0]
                      how strongly this tick's state calls for tool/memory use,
                      with a floor guarantee of at least tool_floor_x
        """
        device = hidden_state.device
        B, S, H = hidden_state.shape

        # 1. Tick embedding broadcast across sequence positions.
        tick_idx = torch.tensor([tick], dtype=torch.long, device=device)
        tick_vec = self.tick_embed(tick_idx)                    # (1, H)
        tick_vec = tick_vec.unsqueeze(1).expand(B, S, H)        # (B, S, H)

        # 2. Mem bias: pool to a single (H,) vector, project, broadcast.
        if mem_bias.dim() == 2:
            mem_bias = mem_bias.unsqueeze(0)                    # (1, num_tokens, H)
        mem_pooled = mem_bias.mean(dim=1)                       # (B or 1, H)
        mem_vec    = self.mem_proj(mem_pooled)                  # (B or 1, H)
        mem_vec    = mem_vec.unsqueeze(1).expand(B, S, H)       # (B, S, H)

        # 3. Compute gate from hidden state + mem signal.
        gate_input  = torch.cat([hidden_state, mem_vec], dim=-1)  # (B, S, 2H)
        gate        = self.gate(gate_input)                        # (B, S, H)

        # 4. Inject mem and tick into the hidden state, then normalise.
        conditioned = self.norm(hidden_state + gate * mem_vec + tick_vec)

        # 5. Tool-readiness score.
        #    Pool conditioned rep → scalar → apply floor transformation.
        #    Detached from the conditioning graph so the floor nudge doesn't
        #    back-propagate into the representation itself.
        pooled     = conditioned.mean(dim=1)                    # (B, H)
        raw_score  = self.tool_head(pooled).squeeze(-1)         # (B,)
        # Take batch element 0 (pipeline runs one input at a time in the demo)
        tool_score = self.tool_floor.apply(raw_score[0].item())

        return conditioned, tool_score
