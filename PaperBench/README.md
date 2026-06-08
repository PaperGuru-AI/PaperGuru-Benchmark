# PaperBench Results — CCM-Backed Paper-to-Code Reproduction

This directory contains the masked artifacts of our CCM-backed run on the
23-paper [PaperBench](https://openreview.net/forum?id=xF5PuTLPbn) benchmark
(Starace et al., ICML 2025). It is referenced in Section 5.3 of the main paper.

## Layout

```
PaperBench/
├── submissions/             23 per-paper code submissions
│                            (one subdir per paper-id; .git history removed)
├── aggregate-final.json     aggregate per-paper scores (frozen final)
├── REPORT.md                human-readable headline report
└── PER_PAPER_COMPARISON.md  per-paper deltas vs. published baselines
```

## Headline Numbers (Table 6 in the main paper)

Full-mode evaluation (code development + reproduction stage),
`o3-mini-2025-01-31` leaf judge. Baselines below are taken from PaperBench
(Starace et al., 2025) and AiScientist (Chen et al., 2026):

| Method                                          | Score    |
|-------------------------------------------------|----------|
| BasicAgent + GPT-4o (Starace et al., 2025)      | 4.10%    |
| BasicAgent + Claude-3.5 (Starace et al., 2025)  | 21.00%   |
| IterativeAgent + o1-high (Starace et al., 2025) | 26.00%   |
| AiScientist + Gemini-3-Flash (Chen et al., 2026)| 30.52%   |
| AiScientist + GLM-5 (Chen et al., 2026)         | 33.73%   |
| Human Expert (48 h budget)                      | 41.00%   |
| **CCM (this paper)**                            | **66.05%**|

## What is masked in this directory

- Internal command-line tools and frontend shells are still referred to with
  neutral tokens (`[anon-cli]`, `[anon-frontend]`).
- Local and remote machine paths and host details have been omitted; commands
  are described in terms of the repo-relative `submissions/` directory instead.
- `.git` histories have been stripped from the per-paper submission trees.
- Prompt templates, agent system prompts, and runner code are NOT shipped
  here; only the produced code submissions (the bench-grading inputs) and
  the grader's output are released.

## Notes

- The 23 per-paper submission trees live under `submissions/<paper-id>/submission/`
  and are byte-faithful (modulo masking) to what was scored by the
  PaperBench grader.
- The aggregate of the per-paper scores is recorded in `aggregate-final.json`.

## Reproducibility & Evaluation Protocol

We report PaperGuru's numbers under the **official PaperBench grading
pipeline** so that any third party — including the PaperBench authors —
can independently re-grade our artifacts without trusting our self-reported
scores.

**Grading was produced with the official grader.** Every score in the table
above comes from the PaperBench rubric-tree grader
(`openai/preparedness` → `project/paperbench`), not from any in-house
scoring. The aggregate in `aggregate-final.json` follows the canonical
PaperBench schema and can be diffed against a fresh grading run.

**How to re-grade our submissions (anyone can do this):**

1. Clone the official benchmark:
   `git clone https://github.com/openai/preparedness && cd preparedness/project/paperbench`
2. `uv sync` to install the grading environment.
3. Point the provided `PBDirectSubmissionSolver` at our `submissions/`
   directory (already in the exact `<paper-id>/submission/` layout the
   solver expects — see the official README section *"I have submissions
   and just want to run grading"*).
4. Run the grader with your own leaf-judge model; the aggregate should
   reproduce the **66.05%** Full-mode figure within judge variance.

**Scope of the claim.** We report the **Full** setting (code development plus
the reproduction stage), the same setting in which the published baselines —
AiScientist + GLM-5 (33.73%), AiScientist + Gemini-3-Flash (30.52%), and
IterativeAgent + o1-high (26.00%) — are evaluated. The human-expert 41%
reference is the Full-mode 48-hour ML-PhD bar reported by Starace et al.
(2025); our 66.05% Full-mode mean is therefore directly comparable to both
the published baselines and the human bar.

**No cherry-picking.** All 23 papers are released, including the single
regression (`pinn`). Per-paper deltas — wins and losses — are in
`PER_PAPER_COMPARISON.md`.

> If you are a PaperBench maintainer and would like us to format these
> results for the leaderboard, or want a held-out re-grading run, please
> open an issue on this repository — we are happy to cooperate on
> independent verification.
