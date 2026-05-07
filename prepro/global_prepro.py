import torch
import torch.nn as nn


class GlobalPrePro(nn.Module):
    """
    Stage 1 — tokenise raw text and encode it through the pretrained backbone.

    The backbone acts as the semantic encoder here; no separate encoder module
    exists. We run a full forward pass with output_hidden_states=True and take
    the final layer's hidden states as the encoded representation.

    A learned LayerNorm normalises the output so downstream modules (routers,
    loop) receive a stable representation regardless of backbone scale.
    """

    def __init__(self, backbone: nn.Module, tokenizer):
        super().__init__()
        self.backbone  = backbone
        self.tokenizer = tokenizer

        # Infer hidden dim from backbone config.
        hidden_dim = backbone.config.hidden_size
        self.norm  = nn.LayerNorm(hidden_dim)

        # Backbone weights are frozen; only norm is trained.
        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, raw_input: str) -> torch.Tensor:
        """
        raw_input : plain text string

        Returns
        -------
        encoded : (1, seq_len, hidden_dim)  — normalised final hidden states
        """
        device = next(self.norm.parameters()).device

        tokens = self.tokenizer(
            raw_input,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
            padding=False,
        )
        input_ids      = tokens["input_ids"].to(device)
        attention_mask = tokens["attention_mask"].to(device)

        with torch.no_grad():
            out = self.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )

        # Last layer hidden states: (1, seq_len, hidden_dim)
        last_hidden = out.hidden_states[-1]

        return self.norm(last_hidden)
