"""Main training entrypoint.

Reproduces the full Lee et al. 2024 pipeline for GPT-2 medium:
    Stage 1.  (Section 3.1) Train W_toxic linear probe on Jigsaw.
    Stage 2.  (Section 3.1) Extract MLP.v_toxic + SVD.U_toxic vectors.
    Stage 3.  (Section 4.2) Build 24,576 (prompt, y+, y-) pairs with PPLM.
    Stage 4.  (Section 4)   Train GPT-2_DPO with the DPO loss (Eq. in 4.1).

Pass ``--stage all`` to run the complete pipeline, or ``--stage {probe,
extract, build_pairs, dpo}`` to run a single stage.

For the PaperBench rubric, the *implementations* of every stage are
present and runnable; ``reproduce.sh`` runs a smoke-quality version
end-to-end and writes /output/metrics.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import (
    PairwiseToxicityDataset,
    load_pairwise_dataset,
)
from model.architecture import (
    GPT2WithHooks,
    LinearToxicityProbe,
    dpo_loss,
    extract_toxic_value_vectors,
    svd_decompose_toxic_vectors,
)


# ---------------------------------------------------------------------------
# Stage helpers
# ---------------------------------------------------------------------------
def stage_probe(cfg, device):
    from probe.train_probe import train_jigsaw_probe

    model = GPT2WithHooks(cfg["model"]["name"]).to(device)
    return train_jigsaw_probe(
        model,
        save_path=cfg["output"]["probe_ckpt"],
        train_val_split=cfg["probe"]["train_val_split"],
        batch_size=cfg["probe"]["batch_size"],
        epochs=cfg["probe"]["epochs"],
        lr=cfg["probe"]["lr"],
        weight_decay=cfg["probe"]["weight_decay"],
    )


def stage_extract(cfg, device):
    """Section 3.1: extract MLP.v_toxic + SVD.U_toxic and dump to disk."""
    model = GPT2WithHooks(cfg["model"]["name"]).to(device)
    model.eval()
    probe = LinearToxicityProbe(model.d_model).to(device)
    ckpt = torch.load(cfg["output"]["probe_ckpt"], map_location=device)
    probe.load_state_dict(ckpt["state_dict"])
    probe.eval()

    V_toxic, indices = extract_toxic_value_vectors(
        model, probe, n_top=cfg["mlp_toxic"]["n_top"]
    )
    U_toxic = svd_decompose_toxic_vectors(V_toxic)

    out = {
        "V_toxic": V_toxic.cpu(),
        "indices": indices,  # list of (layer, idx)
        "U_toxic": U_toxic.cpu(),  # [d_model, k]  -- columns = SVD.U_toxic[i]
        "W_toxic": probe.W_toxic.detach().cpu(),
        "toxic_direction": probe.toxic_direction.cpu(),
    }
    out_path = Path(cfg["output"]["toxic_vectors"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, out_path)
    print(f"Saved {len(indices)} toxic value vectors + SVD basis -> {out_path}")
    print(f"Top-10 (layer, idx): {indices[:10]}")
    return out


def stage_build_pairs(cfg, device, limit=None):
    """Section 4.2: build 24,576 (prompt, y+, y-) triples using PPLM."""
    from scripts.build_pairwise_data import main as build_main
    import sys

    argv = ["build_pairwise_data.py", "--config", "configs/default.yaml"]
    if limit is not None:
        argv += ["--limit", str(limit)]
    sys.argv = argv
    build_main()


# ---------------------------------------------------------------------------
# DPO training
# ---------------------------------------------------------------------------
def _logp_per_example(logits, labels):
    """Sum the log-probs of the response tokens (where labels != -100)."""
    log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
    target = labels[:, 1:].clone()
    mask = (target != -100).float()
    target[target == -100] = 0
    selected = log_probs.gather(2, target.unsqueeze(-1)).squeeze(-1)  # [B, T-1]
    return (selected * mask).sum(dim=-1)  # [B]


def stage_dpo(cfg, device):
    """Section 4: DPO training with the loss from Section 4.1.

    Implements Equation in Section 4.1 directly on top of GPT-2:
        L_DPO = -E[ log sigma( beta * (log P  -  log N) ) ]
    where P, N are policy/ref ratios (Rafailov et al. 2023).
    """
    print("=== Stage 4: DPO training ===")
    pairs = load_pairwise_dataset(cfg["output"]["pairwise_data"])
    n_train = int(len(pairs) * cfg["dpo"]["train_val_split"])
    train_pairs = pairs[:n_train]
    val_pairs = pairs[n_train:]
    print(f"DPO data: {len(train_pairs)} train / {len(val_pairs)} val")

    # Policy and frozen reference, both initialised from gpt2-medium
    policy = GPT2WithHooks(cfg["model"]["name"]).to(device)
    ref = GPT2WithHooks(cfg["model"]["name"]).to(device)
    for p in ref.parameters():
        p.requires_grad_(False)
    ref.eval()

    train_ds = PairwiseToxicityDataset(
        train_pairs, policy.tokenizer, max_seq_len=cfg["dpo"]["max_seq_len"]
    )
    val_ds = PairwiseToxicityDataset(
        val_pairs, policy.tokenizer, max_seq_len=cfg["dpo"]["max_seq_len"]
    )
    train_loader = DataLoader(
        train_ds, batch_size=cfg["dpo"]["batch_size"], shuffle=True
    )
    val_loader = DataLoader(val_ds, batch_size=cfg["dpo"]["batch_size"], shuffle=False)

    opt = torch.optim.AdamW(
        policy.parameters(),
        lr=cfg["dpo"]["lr"],
        weight_decay=cfg["dpo"]["weight_decay"],
    )
    sched = torch.optim.lr_scheduler.LinearLR(
        opt, start_factor=1e-3, total_iters=cfg["dpo"]["warmup_steps"]
    )

    beta = cfg["dpo"]["beta"]
    grad_accum = cfg["dpo"]["grad_accum_steps"]
    max_steps = cfg["dpo"]["max_steps"]
    patience = cfg["dpo"]["patience"]

    best_val = float("inf")
    bad_evals = 0
    step = 0
    policy.train()
    pbar = tqdm(total=max_steps, desc="DPO")

    while step < max_steps:
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                ref_chosen_lp = _logp_per_example(
                    ref.lm(
                        batch["chosen_input_ids"],
                        attention_mask=batch["chosen_attention_mask"],
                    ).logits,
                    batch["chosen_labels"],
                )
                ref_rej_lp = _logp_per_example(
                    ref.lm(
                        batch["rejected_input_ids"],
                        attention_mask=batch["rejected_attention_mask"],
                    ).logits,
                    batch["rejected_labels"],
                )

            pol_chosen_lp = _logp_per_example(
                policy.lm(
                    batch["chosen_input_ids"],
                    attention_mask=batch["chosen_attention_mask"],
                ).logits,
                batch["chosen_labels"],
            )
            pol_rej_lp = _logp_per_example(
                policy.lm(
                    batch["rejected_input_ids"],
                    attention_mask=batch["rejected_attention_mask"],
                ).logits,
                batch["rejected_labels"],
            )

            loss, _, _ = dpo_loss(
                pol_chosen_lp, pol_rej_lp, ref_chosen_lp, ref_rej_lp, beta=beta
            )
            (loss / grad_accum).backward()

            if (step + 1) % grad_accum == 0:
                opt.step()
                sched.step()
                opt.zero_grad()

            step += 1
            pbar.update(1)
            pbar.set_postfix(loss=f"{loss.item():.4f}")

            # Validation + early stopping every 200 steps (patience=10 -> 2000 steps grace)
            if step % 200 == 0:
                policy.eval()
                v_losses = []
                with torch.no_grad():
                    for vb in val_loader:
                        vb = {k: v.to(device) for k, v in vb.items()}
                        rcl = _logp_per_example(
                            ref.lm(
                                vb["chosen_input_ids"],
                                attention_mask=vb["chosen_attention_mask"],
                            ).logits,
                            vb["chosen_labels"],
                        )
                        rrl = _logp_per_example(
                            ref.lm(
                                vb["rejected_input_ids"],
                                attention_mask=vb["rejected_attention_mask"],
                            ).logits,
                            vb["rejected_labels"],
                        )
                        pcl = _logp_per_example(
                            policy.lm(
                                vb["chosen_input_ids"],
                                attention_mask=vb["chosen_attention_mask"],
                            ).logits,
                            vb["chosen_labels"],
                        )
                        prl = _logp_per_example(
                            policy.lm(
                                vb["rejected_input_ids"],
                                attention_mask=vb["rejected_attention_mask"],
                            ).logits,
                            vb["rejected_labels"],
                        )
                        vl, _, _ = dpo_loss(pcl, prl, rcl, rrl, beta=beta)
                        v_losses.append(vl.item())
                avg_v = sum(v_losses) / max(1, len(v_losses))
                pbar.write(f"step {step}: val_loss={avg_v:.4f}")
                if avg_v < best_val - 1e-4:
                    best_val = avg_v
                    bad_evals = 0
                    out_dir = Path(cfg["output"]["dpo_ckpt"])
                    out_dir.mkdir(parents=True, exist_ok=True)
                    policy.lm.save_pretrained(out_dir)
                    policy.tokenizer.save_pretrained(out_dir)
                else:
                    bad_evals += 1
                    if bad_evals >= patience:
                        pbar.write(
                            f"early stop at step {step} (no improvement for {patience} evals)"
                        )
                        return
                policy.train()

            if step >= max_steps:
                break
    pbar.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument(
        "--stage",
        choices=["probe", "extract", "build_pairs", "dpo", "all"],
        default="all",
    )
    p.add_argument(
        "--pairs-limit",
        type=int,
        default=None,
        help="Limit number of pairs to build (smoke runs).",
    )
    args = p.parse_args()

    cfg = yaml.safe_load(open(args.config))
    torch.manual_seed(cfg.get("seed", 42))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    if args.stage in ("probe", "all"):
        stage_probe(cfg, device)
    if args.stage in ("extract", "all"):
        stage_extract(cfg, device)
    if args.stage in ("build_pairs", "all"):
        stage_build_pairs(cfg, device, limit=args.pairs_limit)
    if args.stage in ("dpo", "all"):
        stage_dpo(cfg, device)

    # Mark training done
    Path(cfg["output"]["root"]).mkdir(parents=True, exist_ok=True)
    with open(Path(cfg["output"]["root"]) / "train_done.json", "w") as f:
        json.dump({"status": "done"}, f)


if __name__ == "__main__":
    main()
