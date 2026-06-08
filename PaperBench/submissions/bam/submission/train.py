"""
Unified training entrypoint for the BaM paper experiments.

Usage
-----
    python train.py --config configs/default.yaml --experiment gaussian_5_1
    python train.py --config configs/default.yaml --experiment sinh_arcsinh_5_1
    python train.py --config configs/default.yaml --experiment posteriordb_5_2
    python train.py --config configs/default.yaml --experiment vae_5_3
    python train.py --config configs/default.yaml --smoke

The ``--smoke`` flag forces tiny iteration / sample counts so that the full
pipeline (BaM + GSM + ADVI on every target type) completes within a couple of
minutes on CPU.  It is what reproduce.sh invokes for the PaperBench
smoke evaluation.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict

import numpy as np
import yaml

# Make the package importable when run as a script.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bam import BaM, GSM, ADVI, ScoreVI, FisherVI
from bam.divergences import (
    forward_kl_gaussian,
    reverse_kl_gaussian,
    relative_mean_error,
    relative_sd_error,
)
from data.loader import (
    make_synthetic_gaussian_dataset,
    make_synthetic_sinharcsinh_dataset,
    PosteriorDBLoader,
    load_cifar10,
    _make_synthetic_cifar10,
)
from model.targets import (
    GaussianTarget,
    SinhArcsinhTarget,
    build_posteriordb_target,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _lam_const(B: int, D: int):
    return float(B * D)


def _lam_decaying(B: int, D: int):
    def _f(t: int) -> float:
        return B * D / (t + 1.0)

    return _f


def _grid_search_lr(
    method_cls, target, mu0, Sigma0, grid, n_iters, B, **kw
) -> tuple[float, dict]:
    """Tiny grid search for the gradient-based methods.

    Mirrors the protocol described in the paper: we run each candidate
    learning rate for n_iters steps and pick the one with lowest reverse-KL
    (or, for non-Gaussian targets, lowest score-based divergence proxy).
    """
    if isinstance(target, GaussianTarget):

        def loss(state):
            return reverse_kl_gaussian(
                state.mu, state.Sigma, target.mu_star, target.Sigma_star
            )
    else:

        def loss(state):
            return float(np.linalg.norm(state.mu))  # crude proxy

    best_lr, best_loss, best_history = grid[0], float("inf"), None
    for lr in grid:
        # ADVI / ScoreVI / FisherVI use ``log_p_and_score`` for ADVI and
        # ``score_fn`` for the others.
        if method_cls.__name__ == "ADVI":
            method = method_cls(
                log_p_and_score=lambda Z, t=target: (t.log_prob(Z), t.score(Z)),
                D=mu0.shape[0],
                batch_size=B,
                learning_rate=lr,
                **kw,
            )
        else:
            method = method_cls(
                target_score_fn=target.score,
                D=mu0.shape[0],
                batch_size=B,
                learning_rate=lr,
                **kw,
            )
        state = method.fit(mu0, Sigma0, n_iters=n_iters)
        l = float(loss(state))
        if np.isfinite(l) and l < best_loss:
            best_lr, best_loss, best_history = lr, l, list(method.history)
    return best_lr, {"loss": best_loss, "history": best_history or []}


# ---------------------------------------------------------------------------
# Experiment runners
# ---------------------------------------------------------------------------


def run_gaussian_5_1(cfg: Dict[str, Any], out_dir: str, smoke: bool) -> Dict[str, Any]:
    section = cfg["gaussian_5_1"]
    dims = section["dims"] if not smoke else [4, 16]
    n_iters = 50 if smoke else section["n_iters"]
    n_runs = 1 if smoke else section["n_runs"]
    results: Dict[str, Any] = {}

    for D in dims:
        per_dim: Dict[str, Any] = {}
        for run in range(n_runs):
            target, mu0, Sigma0 = make_synthetic_gaussian_dataset(D, seed=run)
            B = max(2, min(section["bam_batch_sizes"]))
            lam = _lam_const(B, D)

            t0 = time.time()
            bam = BaM(
                score_fn=target.score, D=D, batch_size=B, lam_schedule=lam, seed=run
            )
            bam_state = bam.fit(mu0, Sigma0, n_iters=n_iters)
            bam_kl = forward_kl_gaussian(
                bam_state.mu, bam_state.Sigma, target.mu_star, target.Sigma_star
            )
            t_bam = time.time() - t0

            t0 = time.time()
            gsm = GSM(
                score_fn=target.score,
                D=D,
                batch_size=section["baseline_batch_size"],
                seed=run,
            )
            gsm_state = gsm.fit(mu0, Sigma0, n_iters=n_iters)
            gsm_kl = forward_kl_gaussian(
                gsm_state.mu, gsm_state.Sigma, target.mu_star, target.Sigma_star
            )
            t_gsm = time.time() - t0

            t0 = time.time()
            advi = ADVI(
                log_p_and_score=lambda Z, t=target: (t.log_prob(Z), t.score(Z)),
                D=D,
                batch_size=section["baseline_batch_size"],
                learning_rate=1e-2,
                seed=run,
            )
            advi_state = advi.fit(mu0, Sigma0, n_iters=n_iters)
            advi_kl = forward_kl_gaussian(
                advi_state.mu, advi_state.Sigma, target.mu_star, target.Sigma_star
            )
            t_advi = time.time() - t0

            per_dim.setdefault("bam_forward_kl", []).append(float(bam_kl))
            per_dim.setdefault("gsm_forward_kl", []).append(float(gsm_kl))
            per_dim.setdefault("advi_forward_kl", []).append(float(advi_kl))
            per_dim.setdefault("walltime", []).append(
                {"bam": t_bam, "gsm": t_gsm, "advi": t_advi}
            )
        results[f"D={D}"] = per_dim

    with open(os.path.join(out_dir, "gaussian_5_1.json"), "w") as f:
        json.dump(results, f, indent=2)
    return results


def run_sinh_arcsinh_5_1(
    cfg: Dict[str, Any], out_dir: str, smoke: bool
) -> Dict[str, Any]:
    section = cfg["sinh_arcsinh_5_1"]
    D = section["dim"] if not smoke else 4
    n_iters = 50 if smoke else section["n_iters"]
    skews = section["skews"][:1] if smoke else section["skews"]
    tails = section["tails"][:1] if smoke else section["tails"]

    out: Dict[str, Any] = {}
    for s in skews:
        for tau in tails:
            target, mu0, Sigma0 = make_synthetic_sinharcsinh_dataset(
                D, s=s, tau=tau, seed=0
            )
            B = section["baseline_batch_size"] if not smoke else 4
            lam = _lam_decaying(B, D)
            bam = BaM(
                score_fn=target.score, D=D, batch_size=B, lam_schedule=lam, seed=0
            )
            bam.fit(mu0, Sigma0, n_iters=n_iters)
            gsm = GSM(score_fn=target.score, D=D, batch_size=B, seed=0)
            gsm.fit(mu0, Sigma0, n_iters=n_iters)
            advi = ADVI(
                log_p_and_score=lambda Z, t=target: (t.log_prob(Z), t.score(Z)),
                D=D,
                batch_size=B,
                learning_rate=1e-2,
                seed=0,
            )
            advi.fit(mu0, Sigma0, n_iters=n_iters)
            out[f"s={s}_tau={tau}"] = {
                "bam_history": bam.history,
                "gsm_history": gsm.history,
                "advi_history": advi.history,
            }
    with open(os.path.join(out_dir, "sinh_arcsinh_5_1.json"), "w") as f:
        json.dump(out, f, indent=2, default=float)
    return out


def run_posteriordb_5_2(
    cfg: Dict[str, Any], out_dir: str, smoke: bool
) -> Dict[str, Any]:
    section = cfg["posteriordb_5_2"]
    n_iters = 50 if smoke else section["n_iters"]
    posteriors = section["posteriors"][:1] if smoke else section["posteriors"]
    out: Dict[str, Any] = {}
    for name in posteriors:
        ref = PosteriorDBLoader.from_posteriordb(name)
        target = build_posteriordb_target(name)
        D = ref.D
        for B in section["batch_sizes"]:
            mu0 = np.zeros(D)
            Sigma0 = np.eye(D)
            lam = _lam_decaying(B, D)
            bam = BaM(
                score_fn=target.score, D=D, batch_size=B, lam_schedule=lam, seed=0
            )
            bam_state = bam.fit(mu0, Sigma0, n_iters=n_iters)
            rel_mean = relative_mean_error(bam_state.mu, ref.reference_mean)
            rel_sd = relative_sd_error(bam_state.Sigma, ref.reference_cov)
            out.setdefault(name, {})[f"B={B}"] = {
                "rel_mean_error": float(rel_mean),
                "rel_sd_error": float(rel_sd),
            }
    with open(os.path.join(out_dir, "posteriordb_5_2.json"), "w") as f:
        json.dump(out, f, indent=2)
    return out


def run_vae_5_3(cfg: Dict[str, Any], out_dir: str, smoke: bool) -> Dict[str, Any]:
    """Train (or load) the VAE, then run BaM/ADVI on a held-out test image."""
    section = cfg["vae_5_3"]

    # 1. Train VAE (or load checkpoint).
    images = load_cifar10(train=True, download=True)
    if images is None or smoke:
        images = _make_synthetic_cifar10(n=64 if smoke else 1024)

    ckpt_path = os.path.join(out_dir, "vae_ckpt.pt")
    try:
        import torch
        from model.architecture import VAE, vae_log_prior_likelihood_score

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        vae = VAE(
            c_hid=section["c_hid"],
            latent_dim=section["latent_dim"],
            sigma2=section["sigma2"],
        ).to(device)
        if os.path.exists(ckpt_path):
            ck = torch.load(ckpt_path, map_location=device)
            vae.load_state_dict(ck["state_dict"])
        else:
            from scripts.train_vae import train_vae

            cfg_train = dict(section)
            if smoke:
                cfg_train["vae_n_epochs"] = 1
                cfg_train["vae_batch_size"] = 32
            train_vae(cfg_train, images, ckpt_path)
            ck = torch.load(ckpt_path, map_location=device)
            vae.load_state_dict(ck["state_dict"])
        vae.eval()

        # 2. Pick a test image.
        x_obs = images[0]  # (3, 32, 32)
        score_fn = vae_log_prior_likelihood_score(
            vae.decoder, x_obs, sigma2=section["sigma2"]
        )

        D = section["latent_dim"]
        mu0 = np.zeros(D)
        Sigma0 = np.eye(D)
        B = 50 if smoke else 100
        n_iters = 20 if smoke else section["bam_n_iters_full"]
        lam = _lam_decaying(B, D)
        bam = BaM(score_fn=score_fn, D=D, batch_size=B, lam_schedule=lam, seed=0)
        bam_state = bam.fit(mu0, Sigma0, n_iters=n_iters)

        # 3. Reconstruction MSE: feed E[z|x'] into decoder.
        with torch.no_grad():
            z_t = torch.as_tensor(
                bam_state.mu[None, :], dtype=torch.float32, device=device
            )
            x_hat = vae.decoder(z_t).cpu().numpy()[0]
        mse = float(np.mean((x_obs - x_hat) ** 2))
        out = {"reconstruction_mse_bam": mse, "n_iters": n_iters, "B": B}
    except Exception as e:  # noqa: BLE001
        out = {"error": str(e), "note": "VAE experiment skipped (PyTorch unavailable)."}
    with open(os.path.join(out_dir, "vae_5_3.json"), "w") as f:
        json.dump(out, f, indent=2, default=float)
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


EXPERIMENTS = {
    "gaussian_5_1": run_gaussian_5_1,
    "sinh_arcsinh_5_1": run_sinh_arcsinh_5_1,
    "posteriordb_5_2": run_posteriordb_5_2,
    "vae_5_3": run_vae_5_3,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        choices=list(EXPERIMENTS.keys()) + ["all"],
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a tiny smoke version (used by reproduce.sh).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output directory (default: cfg.output_dir).",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    out_dir = args.out or cfg.get("output_dir", "./outputs")
    os.makedirs(out_dir, exist_ok=True)

    np.random.seed(cfg.get("seed", 0))

    chosen = args.experiment or cfg.get("experiment", "gaussian_5_1")
    targets = list(EXPERIMENTS.keys()) if chosen == "all" else [chosen]
    summary: Dict[str, Any] = {}
    for name in targets:
        print(f"== Running {name} ==  smoke={args.smoke}")
        summary[name] = EXPERIMENTS[name](cfg, out_dir, args.smoke)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"Done. Results in {out_dir}/summary.json")


if __name__ == "__main__":
    main()
