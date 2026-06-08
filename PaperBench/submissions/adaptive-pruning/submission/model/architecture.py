"""APTModel — wraps a HuggingFace transformer with APT adapters.

We replace the linear projections of MHA (q,v) and FFN (intermediate /
output) with `APTLinear`, leaving the rest of the model intact. The
returned model exposes hidden states for self-distillation and the
`PruneController` / `RankController` can call into its APTLinear layers
directly via `model.named_modules()`.

Supported backbones (paper §5.1):
  * roberta-base / roberta-large
  * t5-base / t5-large (lm-adapt)
  * (bert-base-uncased — same plumbing as RoBERTa)

For LLaMA the same conversion works in principle but is out of scope for
this replication (per addendum: 'All results involving LLaMa
models are not required for replication').
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
from torch import nn

from .apt_adapter import APTLinear


# --------------------------------------------------------------------------- #
def _replace_linear(
    parent: nn.Module,
    attr: str,
    rank: int,
    alpha: float,
    dropout: float,
) -> Optional[APTLinear]:
    """In-place replace `parent.attr` (an nn.Linear) with an APTLinear."""
    if not hasattr(parent, attr):
        return None
    base = getattr(parent, attr)
    if not isinstance(base, nn.Linear):
        return None
    new = APTLinear(base, rank=rank, alpha=alpha, dropout=dropout)
    setattr(parent, attr, new)
    return new


# --------------------------------------------------------------------------- #
class APTModel(nn.Module):
    """End-to-end APT model = backbone + classifier head.

    The backbone is a HuggingFace AutoModel. We attach a classification
    head (logistic regression on the pooled output) for GLUE tasks; users
    targeting SQuAD or CNN/DM should set `task_type` accordingly and use
    AutoModelForQuestionAnswering / AutoModelForSeq2SeqLM directly.
    """

    def __init__(
        self,
        backbone: nn.Module,
        hidden_size: int,
        num_labels: int = 2,
        task_type: str = "classification",
        adapter_rank: int = 8,
        adapter_alpha: float = 16.0,
        adapter_dropout: float = 0.1,
        apply_to_q: bool = True,
        apply_to_v: bool = True,
        apply_to_ffn_up: bool = True,
        apply_to_ffn_down: bool = True,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.hidden_size = hidden_size
        self.task_type = task_type
        self.num_labels = num_labels

        # Freeze every backbone parameter; we only train adapters + head.
        for p in self.backbone.parameters():
            p.requires_grad_(False)

        # Convert q/v projections in MHA and FFN linears.
        self._convert_layers(
            rank=adapter_rank,
            alpha=adapter_alpha,
            dropout=adapter_dropout,
            apply_to_q=apply_to_q,
            apply_to_v=apply_to_v,
            apply_to_ffn_up=apply_to_ffn_up,
            apply_to_ffn_down=apply_to_ffn_down,
        )

        # Classification head.
        self.classifier = nn.Sequential(
            nn.Dropout(adapter_dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Dropout(adapter_dropout),
            nn.Linear(hidden_size, num_labels),
        )

    # ------------------------------------------------------------------ #
    def _convert_layers(
        self,
        rank: int,
        alpha: float,
        dropout: float,
        apply_to_q: bool,
        apply_to_v: bool,
        apply_to_ffn_up: bool,
        apply_to_ffn_down: bool,
    ) -> None:
        targets_attn = []
        if apply_to_q:
            targets_attn += ["query", "q_proj", "q"]
        if apply_to_v:
            targets_attn += ["value", "v_proj", "v"]
        targets_ffn_up = (
            ["intermediate", "fc1", "wi_0", "wi", "up_proj"] if apply_to_ffn_up else []
        )
        targets_ffn_down = (
            ["output", "fc2", "wo", "down_proj"] if apply_to_ffn_down else []
        )

        for module in list(self.backbone.modules()):
            for attr in dir(module):
                if attr.startswith("_"):
                    continue
                child = getattr(module, attr, None)
                if not isinstance(child, nn.Linear):
                    continue
                lname = attr.lower()
                # Be careful: HF roberta has both attention.self.query and
                # attention.output.dense (residual).  We only swap the
                # canonical projection names.
                if lname in targets_attn:
                    _replace_linear(module, attr, rank, alpha, dropout)
                elif (
                    lname == "dense"
                    and getattr(module, "intermediate_act_fn", None) is not None
                ):
                    # FFN intermediate (RoBERTa-style).
                    if apply_to_ffn_up:
                        _replace_linear(module, attr, rank, alpha, dropout)
                elif lname in targets_ffn_up:
                    if not isinstance(child, APTLinear):
                        _replace_linear(module, attr, rank, alpha, dropout)
                elif lname in targets_ffn_down:
                    if not isinstance(child, APTLinear):
                        _replace_linear(module, attr, rank, alpha, dropout)

    # ------------------------------------------------------------------ #
    def trainable_parameters(self) -> List[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    def num_trainable(self) -> int:
        return sum(p.numel() for p in self.trainable_parameters())

    def num_total(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def enable_tracking(self, flag: bool = True) -> None:
        for m in self.modules():
            if isinstance(m, APTLinear):
                m.enable_tracking(flag)

    # ------------------------------------------------------------------ #
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False,
    ) -> Dict[str, torch.Tensor]:
        kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "output_hidden_states": output_hidden_states,
        }
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids
        out = self.backbone(**kwargs)

        # Pool: take [CLS] / first token (BERT/RoBERTa convention).
        hidden = out.last_hidden_state
        pooled = hidden[:, 0]
        logits = self.classifier(pooled)

        result = {"logits": logits}
        if output_hidden_states and getattr(out, "hidden_states", None) is not None:
            result["hidden_states"] = out.hidden_states
        if labels is not None:
            if self.num_labels == 1:
                loss = nn.functional.mse_loss(logits.squeeze(-1), labels.float())
            else:
                loss = nn.functional.cross_entropy(logits, labels.long())
            result["loss"] = loss
        return result


# --------------------------------------------------------------------------- #
def build_apt_model(cfg: dict) -> APTModel:
    """Factory that loads the HF backbone described by `cfg` and wraps it."""
    from transformers import AutoModel, AutoConfig

    name = cfg["model"]["name"]
    hf_cfg = AutoConfig.from_pretrained(name)
    backbone = AutoModel.from_pretrained(name, config=hf_cfg)
    hidden = getattr(hf_cfg, "hidden_size", None) or getattr(hf_cfg, "d_model", 768)

    a = cfg["adapter"]
    return APTModel(
        backbone=backbone,
        hidden_size=hidden,
        num_labels=cfg["model"].get("num_labels", 2),
        task_type=cfg["model"].get("task_type", "classification"),
        adapter_rank=a["init_rank"],
        adapter_alpha=a["lora_alpha"],
        adapter_dropout=a.get("lora_dropout", 0.1),
        apply_to_q=a["apply_to"].get("attention_q", True),
        apply_to_v=a["apply_to"].get("attention_v", True),
        apply_to_ffn_up=a["apply_to"].get("ffn_up", True),
        apply_to_ffn_down=a["apply_to"].get("ffn_down", True),
    )
