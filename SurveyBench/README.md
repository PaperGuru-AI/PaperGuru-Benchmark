# SurveyBench Results — CCM-Backed Long-Form Survey Generation

This directory contains the **best version** of the 20 generated surveys
on the [SurveyBench](https://github.com/...) benchmark, used to substantiate
the long-document generalization claims of CCM in Section 5.2 of the main paper.

The system under test combines a Claude Opus 4.7 backbone with the CCM
lifecycle-aware memory described in Section 3 of the paper; all
prompt templates, system-level scaffolding, and runtime details are
deliberately excluded from this release.

## Layout

```
results/surveybench/
├── md/        20 markdown surveys (post-revision body, references inline)
├── latex/     20 LaTeX projects with figures (one subdir per topic)
└── pdf/       20 compiled PDFs (single-column ICML-style)
```

Each topic appears in all three subfolders under the same slug
(e.g. `3d-gaussian-splatting.md`, `latex/3d-gaussian-splatting/`,
`pdf/3d-gaussian-splatting.pdf`).

## Headline Numbers (Table~5 in the main paper)

Under the official `claude-opus-4.7` judge at the 200K-character truncation
cap, averaged over three independent judge runs:

| Method                            | Content avg | Richness |
|-----------------------------------|-------------|----------|
| AutoSurvey (Wang et al., 2024)    | 3.510       | 0.00     |
| SurveyForge (Yan et al., 2025)    | 3.770       | 0.00     |
| LLM×MR-V2 (Liang et al., 2025)    | 3.970       | 5.09     |
| OpenAI DeepResearch (2024)        | 4.030       | 1.56     |
| **CCM (this paper)**              | **4.733**   | **10.94**|

## What is masked in this directory

- All authorship / affiliation metadata in the LaTeX headers has been
  rewritten to `Anonymous / Anonymous Institution`.
- All references to specific runtime systems, agent frameworks, prompt
  templates, and proprietary infrastructure have been removed.
- Only the final per-topic survey body is shipped; intermediate revision
  passes, per-pass logs, and judge-side traces are omitted.

## Reproducibility & Evaluation Protocol

We report PaperGuru's numbers under the **official SurveyBench judge** so
that the SurveyBench authors or any third party can independently re-score
our 20 surveys without trusting our self-reported numbers.

**Judge.** Scores use the official `claude-opus-4.7` judge at the 200K-character
truncation cap, averaged over **three independent judge runs** (variance
reported in the main paper). The content score is the mean over the five
SurveyBench content dimensions (Coverage, Coherence, Depth, Focus, Fluency);
Richness is the composite figure/table/code measure defined by SurveyBench.

**How to re-score our surveys (anyone can do this):**

1. Take the 20 final surveys in `md/` (or the compiled `pdf/`).
2. Run them through the official SurveyBench judging harness
   (`OpenDataBox/SurveyBench`) with your own judge model.
3. The content average should reproduce **94.66%** (raw 4.733 / 5) and the
   composite richness **43.76%** (raw 10.94 / 25) within judge variance.

**Why the Richness gap is robust.** Composite richness is a *file-system
measurement* — it counts resolved figures, tables, and executable code
blocks actually present in the survey artifact. It does **not** depend on
the LLM judge, so the gap (43.76% vs. the strongest baseline's 20.36%) is
invariant under judge swap and under truncation swap. Two baselines
(ASur, SurveyForge) score exactly 0.00% here because they emit no figures
or tables at all — this is directly verifiable by inspecting their outputs.

**Normalization.** All SurveyBench dimensions in our tables are normalized
to [0, 100%] from the raw 0–5 (content) and 0–25 (richness) scales for
readability; raw values are given alongside.

> If you are a SurveyBench maintainer and would like us to format these
> results for a leaderboard, or want a held-out re-scoring run, please open
> an issue on this repository — we are happy to cooperate on independent
> verification.
