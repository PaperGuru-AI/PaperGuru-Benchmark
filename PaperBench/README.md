# PaperBench Results — CCM-Backed Paper-to-Code Reproduction

This directory contains the masked artifacts of our CCM-backed run on the
23-paper [PaperBench](https://openreview.net/forum?id=xF5PuTLPbn) benchmark
(Starace et al., ICML 2025). It is referenced in Section 5.3 of the main paper.

## Layout

```
results/paperbench/
├── submissions/    23 per-paper code submissions
│                   (one subdir per paper-id; .git history removed)
└── results/        official PaperBench grader output
    ├── grade-jsons/         per-paper rubric scores (machine-readable JSON)
    ├── grading-logs/        per-paper grader logs (text)
    ├── aggregate.json       aggregate per-paper scores
    ├── aggregate-final.json aggregate, frozen final
    ├── REPORT.md            human-readable headline report
    └── PER_PAPER_COMPARISON.md   per-paper deltas vs. published baselines
```

## Headline Numbers (Table 6 in the main paper)

Code-Development mode, `o3-mini-2025-01-31` leaf judge. Baselines below
are taken from PaperBench (Starace et al., 2025) and AiScientist
(Chen et al., 2026):

| Method                                       | Mode      | Score    |
|----------------------------------------------|-----------|----------|
| BasicAgent + GPT-4o (Starace et al., 2025)   | Full      | 4.10%    |
| BasicAgent + Claude-3.5 (Starace et al., 2025)| Full     | 21.00%   |
| IterativeAgent + o1-high (Starace et al., 2025)| Full    | 26.00%   |
| AiScientist + Gemini-3-Flash (Chen et al., 2026)| Full   | 30.52%   |
| AiScientist + GLM-5 (Chen et al., 2026)      | Full      | 33.73%   |
| Human Expert (48 h budget)                   | Full      | 41.00%   |
| IterativeAgent + o1-high                     | Code-Dev  | 43.40%   |
| **CCM (this paper)**                         | Code-Dev  | **66.05%**|

## What is masked in this directory

- All references to specific runtime systems, agent frameworks, command-line
  tools, gateway services, frontend shells, and proprietary backends have
  been replaced with neutral tokens (`[anon-runtime]`, `[anon-cli]`,
  `[anon-gateway]`, `[anon-frontend]`, `[backbone-LLM]`).
- All absolute on-disk paths revealing the project name have been replaced
  with `[redacted-path]`.
- `.git` histories have been stripped from the per-paper submission trees.
- Prompt templates, agent system prompts, and runner code are NOT shipped
  here; only the produced code submissions (the bench-grading inputs) and
  the grader's output are released.

## Notes

- The 23 per-paper submission trees live under `submissions/<paper-id>/submission/`
  and are byte-faithful (modulo masking) to what was scored by the
  PaperBench Code-Development grader.
- Per-paper grader output (one JSON per paper) lives under
  `results/grade-jsons/<paper-id>.json` and follows the canonical
  PaperBench rubric tree schema.

## Reproducibility & Evaluation Protocol

We report PaperGuru's numbers under the **official PaperBench grading
pipeline** so that any third party — including the PaperBench authors —
can independently re-grade our artifacts without trusting our self-reported
scores.

**Grading was produced with the official grader.** Every score in the table
above comes from the PaperBench rubric-tree grader
(`openai/preparedness` → `project/paperbench`), not from any in-house
scoring. The per-paper rubric JSONs under `results/grade-jsons/` follow the
canonical PaperBench schema and can be diffed against a fresh grading run.

**How to re-grade our submissions (anyone can do this):**

1. Clone the official benchmark:
   `git clone https://github.com/openai/preparedness && cd preparedness/project/paperbench`
2. `uv sync` to install the grading environment.
3. Point the provided `PBDirectSubmissionSolver` at our `submissions/`
   directory (already in the exact `<paper-id>/submission/` layout the
   solver expects — see the official README section *"I have submissions
   and just want to run grading"*).
4. Run the grader with your own leaf-judge model; the aggregate should
   reproduce the **66.05%** Code-Development figure within judge variance.

**Scope of the claim.** We report the **Code-Development** setting
(the same setting in which the strongest published baseline, IterativeAgent +
o1-high, scores 43.40%). We do not claim a Full-replication-mode number.
The human-expert 41% reference is the Full-mode 48-hour ML-PhD bar reported
by Starace et al. (2025); we cite it for context only.

**No cherry-picking.** All 23 papers are released, including the single
regression (`pinn`). Per-paper deltas — wins and losses — are in
`results/PER_PAPER_COMPARISON.md`.

> If you are a PaperBench maintainer and would like us to format these
> results for the leaderboard, or want a held-out re-grading run, please
> open an issue on this repository — we are happy to cooperate on
> independent verification.
