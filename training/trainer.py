import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from training.config import Config
from core.pipeline import Pipeline
from eval.evaluator import Evaluator


class _TextPairDataset(Dataset):
    """Minimal wrapper: expects a list of {"input": str, "answer": str} dicts."""
    def __init__(self, samples: list):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class Trainer:
    def __init__(self, config: Config):
        self.config   = config
        self.pipeline = Pipeline(config)
        self.device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._move_to_device()
        self.optimizer = self._build_optimizer()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self, train_dataset, val_dataset, num_epochs: int) -> None:
        loader = DataLoader(
            _TextPairDataset(train_dataset),
            batch_size=1,           # pipeline operates on single sequences
            shuffle=True,
        )

        for epoch in range(1, num_epochs + 1):
            self.pipeline.answer_decoder.backbone.train()
            total_loss = 0.0

            for step, sample in enumerate(loader, 1):
                input_text  = sample["input"][0]
                target_text = sample["answer"][0]

                self.optimizer.zero_grad()
                loss = self._forward_train(input_text, target_text)
                loss.backward()
                nn.utils.clip_grad_norm_(self._trainable_params(), max_norm=1.0)
                self.optimizer.step()

                total_loss += loss.item()

                if step % 50 == 0:
                    print(f"Epoch {epoch} | step {step} | "
                          f"loss {total_loss / step:.4f}")

            avg_loss = total_loss / max(len(loader), 1)
            print(f"Epoch {epoch} complete — avg loss: {avg_loss:.4f}")

            if val_dataset:
                self._validate(val_dataset, epoch)

    def save_checkpoint(self, path: str) -> None:
        # Keys are saved as "module_attr.param_name" so pipeline.load_checkpoint()
        # can match them by prefix (e.g. "global_router.context_proj.weight")
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(self._trainable_state_dict(), path)
        print(f"Checkpoint saved: {path}")

    def load_checkpoint(self, path: str) -> None:
        # Delegates to Pipeline.load_checkpoint() so loading logic is in one place
        self.pipeline.load_checkpoint(path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _forward_train(self, input_text: str, target_text: str) -> torch.Tensor:
        """
        Teacher-forced training pass.

        We run the full pipeline up to (but not including) autoregressive
        generation, then compute cross-entropy against the target tokens by
        appending the target embeddings and masking the context positions.
        """
        # Reset ephemeral state same as pipeline.run()
        self.pipeline.mem_tokens.reset()
        self.pipeline.recurrent_state.reset()

        # Stage 1-4: encode → route → loop → aggregate
        encoded     = self.pipeline.global_prepro.forward(input_text)
        action_plan, _ = self.pipeline.global_router.forward(encoded)
        final_hidden, loop_outputs = self.pipeline.loop.run(
            encoded_input=encoded,
            action_plan=action_plan,
            max_ticks=self.config.max_ticks,
        )
        aggregated = self.pipeline.final_prepro.forward(final_hidden, loop_outputs)

        return self._compute_loss(aggregated, target_text)

    def _compute_loss(self, aggregated: torch.Tensor, target: str) -> torch.Tensor:
        """
        Cross-entropy loss via teacher forcing.

        The aggregated context is prepended to the target token embeddings;
        positions in the context region are masked with -100 so only the
        target positions contribute to the loss.
        """
        backbone  = self.pipeline.answer_decoder.backbone
        tokenizer = self.pipeline.tokenizer

        target_ids = tokenizer(
            target, return_tensors="pt", truncation=True, max_length=512,
        ).input_ids.to(self.device)                                # (1, T)

        embed = backbone.get_input_embeddings()
        target_embeds = embed(target_ids)                          # (1, T, D)

        # Full sequence: [context tokens | target tokens]
        full_embeds = torch.cat([aggregated, target_embeds], dim=1)

        ctx_len = aggregated.size(1)
        # Labels: -100 for context (ignored), actual ids for target tokens.
        ignore  = torch.full((1, ctx_len), -100, dtype=torch.long, device=self.device)
        labels  = torch.cat([ignore, target_ids], dim=1)           # (1, ctx+T)

        out = backbone(inputs_embeds=full_embeds, labels=labels)
        return out.loss

    def _validate(self, val_dataset, epoch: int) -> None:
        self.pipeline.answer_decoder.backbone.eval()
        evaluator = Evaluator(self.pipeline)
        metrics   = evaluator.evaluate(val_dataset)
        print(f"[Validation epoch {epoch}] "
              f"exact_match={metrics.get('exact_match', 0):.3f}  "
              f"f1={metrics.get('f1', 0):.3f}")

    def _move_to_device(self) -> None:
        # Move every module that processes tensors to the target device.
        # global_prepro must be moved as a unit so its LayerNorm lands on the same
        # device as the backbone output it normalises.
        for mod in (
            self.pipeline.global_prepro,        # backbone + norm
            self.pipeline.global_router,
            self.pipeline.loop.local_prepro,
            self.pipeline.loop.local_router,
            self.pipeline.loop.shared_decoder,
            self.pipeline.mem_tokens,
            self.pipeline.final_prepro,
            self.pipeline.answer_decoder,       # backbone + embedding layer
        ):
            if isinstance(mod, nn.Module):
                mod.to(self.device)
        # Freeze backbone weights after moving; only new modules are trained.
        for param in self.pipeline.answer_decoder.backbone.parameters():
            param.requires_grad = False

    def _build_optimizer(self) -> torch.optim.Optimizer:
        trainable = [p for p in self._trainable_params() if p.requires_grad]
        return torch.optim.AdamW(trainable, lr=1e-4, weight_decay=1e-2)

    def _trainable_params(self):
        """Yields parameters of the non-frozen modules."""
        # Routers, mem token biases, stop-check head, shared decoder adapter,
        # and update-gate scoring head are all trainable.
        trainable_modules = [
            self.pipeline.global_router,
            self.pipeline.loop.local_prepro,
            self.pipeline.loop.local_router,
            self.pipeline.loop.shared_decoder.context_proj,
            self.pipeline.loop.stop_check,
            self.pipeline.mem_tokens,
            self.pipeline.final_prepro,
        ]
        for mod in trainable_modules:
            if isinstance(mod, nn.Module):
                yield from mod.parameters()

    def _trainable_state_dict(self) -> dict:
        # Keys are prefixed with the module name so pipeline.load_checkpoint()
        # can match them by prefix (e.g. "global_router.step_head.weight")
        prefix_map = {
            id(self.pipeline.global_router):                    "global_router",
            id(self.pipeline.loop.local_prepro):                "local_prepro",
            id(self.pipeline.loop.local_router):                "local_router",
            id(self.pipeline.loop.shared_decoder.context_proj): "context_proj",
            id(self.pipeline.loop.stop_check):                  "stop_check",
            id(self.pipeline.mem_tokens):                       "mem_tokens",
            id(self.pipeline.final_prepro):                     "final_prepro",
        }
        state: dict = {}
        for mod in (
            self.pipeline.global_router,
            self.pipeline.loop.local_prepro,
            self.pipeline.loop.local_router,
            self.pipeline.loop.shared_decoder.context_proj,
            self.pipeline.loop.stop_check,
            self.pipeline.mem_tokens,
            self.pipeline.final_prepro,
        ):
            if isinstance(mod, nn.Module):
                prefix = prefix_map[id(mod)]
                for name, param in mod.named_parameters():
                    state[f"{prefix}.{name}"] = param.data
        return state
