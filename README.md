<div align="center">

<img src="assets/figures/hero_banner.png" alt="PaperGuru — research at the speed of light" width="100%" />

<br/>

# PaperGuru &nbsp;·&nbsp; The Lifecycle-Aware Memory Benchmark

**The first long-term memory primitive for long-horizon LLM agents — with state-of-the-art results on PaperBench and SurveyBench, and a peer-reviewed track record across FSE 2026, ICML 2026, TOSEM, AEI, and ICoGB.**

[![Paper](https://img.shields.io/badge/paper-PDF-1f3a8a?style=for-the-badge&logo=arxiv&logoColor=white)](paper/PaperGuru-CCM.pdf)
[![PaperBench](https://img.shields.io/badge/PaperBench-66.05%25-0a0a0b?style=for-the-badge)](PaperBench/)
[![SurveyBench](https://img.shields.io/badge/SurveyBench-94.66%25-0a0a0b?style=for-the-badge)](SurveyBench/)
[![Accepted](https://img.shields.io/badge/Accepted-10%20papers-c9a14a?style=for-the-badge)](#-track-record)
[![中文 README](https://img.shields.io/badge/中文-README-dc2626?style=for-the-badge)](README.zh-CN.md)
[![Join WeChat](https://img.shields.io/badge/JOIN-WECHAT-07c160?style=for-the-badge&logo=wechat&logoColor=white&labelColor=555555)](assets/badges/wechat_qr.png)

</div>

---

## 📰 TL;DR

> AI infrastructure has three commodity primitives — **compute** (NVIDIA), **models** (frontier LLM weights), and **retrieval** (Pinecone-class vector databases). A fourth primitive — **long-term memory with lifecycle semantics** — is missing, and every long-horizon LLM system reinvents it badly. **PaperGuru** is the first system designed from a 4-axiom formalisation of *Lifecycle-Aware Memory* (LAM), and on the two most rigorous published benchmarks it delivers state-of-the-art from a single algorithmic mechanism.

| Benchmark | Metric | PaperGuru | Best published baseline | Lift |
|---|---|---:|---:|---:|
| **PaperBench** (OpenAI, 2025) | Mean reproduction across 23 papers | **66.05%** | 35.74% | **+30.21%** |
| **PaperBench** | Papers above 41% human ML-PhD bar | **20 / 23** | 4 / 23 | **+16 papers** |
| **SurveyBench** (Yan et al., 2025) | Content score (5-axis avg.) | **94.66%** | 80.60% | **+14.06%** |
| **SurveyBench** | Composite richness (figures · tables · code) | **43.76%** | 20.36% | **+23.40%** |
| **Real world** | Peer-reviewed acceptances since Q4 2025 | **10 papers** | — | 5 venues |

<br/>

<div align="center">
<img src="assets/figures/before_after.png" alt="Before / After PaperGuru" width="92%" />
<br/>
<sub><i>Same researcher, same task. The difference is the memory primitive underneath.</i></sub>
</div>

---

## 📑 Table of Contents

- [What is PaperGuru?](#-what-is-paperguru)
- [Why memory, why now](#-why-memory-why-now)
- [The CCM architecture](#-the-ccm-architecture)
- [Pipeline at a glance](#-pipeline-at-a-glance)
- [Results · PaperBench](#-results--paperbench)
- [Results · SurveyBench](#-results--surveybench)
- [Track Record](#-track-record)
- [What is in this repository](#-what-is-in-this-repository)
- [Reproducing the figures](#-reproducing-the-figures)
- [License](#-license)

---

## ✨ What is PaperGuru?

PaperGuru is a memory architecture for long-horizon LLM agents. It is **not** another RAG library and **not** another agent framework. It is the first concrete instantiation of **Lifecycle-Aware Memory (LAM)** — a four-axiom system primitive that production agents have been quietly missing.

The four LAM axioms (formalised in §3 of the paper):

1. **Versioned content** &nbsp;·&nbsp; statements once correct must become *stale* after revision, deprecation, or retraction — the memory layer must know.
2. **Structural multi-hop relevance** &nbsp;·&nbsp; the right evidence is two citations away, not one cosine-similarity hop.
3. **Bounded query cost under unbounded archive growth** &nbsp;·&nbsp; the archive grows every day; routing cost cannot grow with it.
4. **Provenance-grounded composition** &nbsp;·&nbsp; every claim in the agent's output must trace back to a verifiable artifact in memory.

PaperGuru satisfies all four axioms jointly through a single mechanism — the **Capital Chunk Memory (CCM)** — described next.

---

## 🌍 Why memory, why now

Context windows have moved from 4K to 1M tokens in eighteen months, and the next milestone — *persistent* memory across sessions — is on every foundation lab's roadmap. The dominant unit of LLM deployment is no longer the single-prompt completion; it is the **long-horizon agentic system**:

- a multi-day software-engineering session that touches hundreds of files,
- a literature-grounded research assistant that drafts a 200K-token survey from a citation graph spanning ten years,
- a paper-to-code reproduction agent that turns a paper PDF into a runnable submission tree,
- a clinical-evidence agent that reads a decade of trial records before recommending treatment.

Across every published evaluation of these systems — **SurveyBench**, **PaperBench**, **SWE-bench-Live** — the ceiling is no longer set by the backbone's reasoning ability. It is set by what the system *remembers* between turns and *retrieves* from outside the working set.

---

## 🏛 The CCM architecture

<div align="center">
<img src="assets/figures/architecture.png" alt="PaperGuru CCM architecture" width="92%" />
</div>

PaperGuru separates memory into two surfaces:

- **Chunk heads** — a compact, bounded routing surface (one head per artifact)
- **Chunk contents** — the unbounded raw text, accessed lazily on demand

A central **capital chunk** indexes all heads and supports *capital-first routing* over a **temporal artifact graph** that unifies two edge classes:

| Edge class | Examples |
|---|---|
| **Structural edges** | `cites`, `benchmarked-on`, `introduced-by`, `implements` |
| **Historical-causality edges** | `discussed-in`, `deprecated-by`, `retracted-by`, `superseded-by` |

Query-time context is constructed through a **route-first → expand-second → distill-last** pipeline that yields compact, provenance-grounded **evidence cards** — the single data structure on which the rest of the system operates.

> **Why this matters in practice.** Flat retrieval breaks the moment a paper is revised; agent-specific memory hacks (MemGPT tiers, Ebbinghaus forgetting, knowledge-graph wrappers) each handle one or two of the four axioms but never all four. CCM is the first design we know of that satisfies all four jointly without per-task hand-tuning.

---

## ⚙️ Pipeline at a glance

<div align="center">
<img src="assets/figures/pipeline.png" alt="PaperGuru CCM pipeline" width="92%" />
<br/>
<sub><i>Search → Extract → Reason → Verify. The Reason stage is where the Compose / Critique / Mutate cycle runs.</i></sub>
</div>

<br/>

<div align="center">
<img src="assets/demos/pipeline_animated.svg" alt="PaperGuru live pipeline" width="92%" />
<br/>
<sub><i>Live SVG animation — the same pipeline, rendered as a beating system. Three sub-blocks of the Reason stage cycle through Compose / Critique / Mutate; verification ticks resolve as evidence cards land.</i></sub>
</div>

| Stage | Input | Output | Compute share* |
|---|---|---|---:|
| **01 · SEARCH**  | Topic query, candidate archive | Ranked artifact heads | ~15% |
| **02 · EXTRACT** | Heads + chunk contents | Evidence cards (text + provenance) | ~20% |
| **03 · REASON**  | Evidence cards | Draft segments (Compose → Critique → Mutate loop) | ~45% |
| **04 · VERIFY**  | Draft segments | Cited, provenance-checked output | ~20% |

<sup>* Indicative share of total wall-clock for a typical 200K-token survey run; varies by task.</sup>

---

## 📊 Results · PaperBench

PaperBench (OpenAI, 2025) is the canonical paper-to-code reproduction benchmark: each submission is a runnable code tree scored by a leaf-judge LLM against a hand-written rubric. The official human-expert baseline is **41% over a 48-hour ML-PhD budget**.

### Aggregate

<div align="center">
<img src="assets/figures/paperbench_topline.png" alt="PaperBench top-line comparison" width="88%" />
</div>

PaperGuru reaches a **per-paper mean of 66.05% across all 23 papers**, beating every published baseline and clearing the human-expert bar by **+25 points**.

### Per-paper

<div align="center">
<img src="assets/figures/paperbench_bar.png" alt="PaperBench per-paper breakdown" width="100%" />
</div>

For the 20 papers that have a published baseline:

- **19 / 20** papers improve over the strongest baseline
- Mean lift: **+30.21% absolute**
- Median lift: **+27.0% absolute**
- Largest gain: `stay-on-topic-with-classifier-free-guidance` (+68.03%)
- Only regression: `pinn` (−4.47%, where the cited PINN baseline already used hand-tuned domain priors)

### Distribution of lifts

<div align="center">
<img src="assets/figures/lift_distribution.png" alt="Distribution of per-paper lifts" width="80%" />
</div>

Three additional papers (`semantic-self-consistency`, `self-composing-policies`, `self-expansion`) have no published baseline; PaperGuru scores **95.45%**, **65.03%**, and **39.77%** respectively.

📂 **All 23 reproduction submissions are in [`PaperBench/submissions/`](PaperBench/submissions/).** Aggregate scores are in [`PaperBench/aggregate-final.json`](PaperBench/aggregate-final.json) and the per-paper comparison report is in [`PaperBench/PER_PAPER_COMPARISON.md`](PaperBench/PER_PAPER_COMPARISON.md).

---

## 📊 Results · SurveyBench

SurveyBench (Yan et al., 2025) evaluates long-form survey writing along three dimensions — Content, Outline, and Richness — under an LLM judge. PaperGuru is evaluated under the official `claude-opus-4.7` judge with all dimensions normalised to [0, 100%].

### Content quality (5-axis radar)

<div align="center">
<img src="assets/figures/surveybench_radar.png" alt="SurveyBench radar comparison" width="80%" />
</div>

PaperGuru scores **94.66%** on the content average — a **+14.06%** absolute lift over the strongest baseline (AutoSurvey at 80.60%) — and reaches the ceiling on **Focus** and **Fluency**.

| Dimension | PaperGuru | AutoSurvey | LLM×MR-v2 | SurveyForge | ASur | Lift |
|---|---:|---:|---:|---:|---:|---:|
| Coverage  | **94.00%** | 61.00% | 70.00% | 60.00% | 59.00% | **+24.00%** |
| Coherence | **87.40%** | 80.00% | 78.00% | 79.00% | 76.00% | **+7.40%**  |
| Depth     | **92.00%** | 75.00% | 75.00% | 72.00% | 60.00% | **+17.00%** |
| Focus     | **100.00%**| 99.00% | 94.00% | 86.00% | 76.00% | **+1.00%**  |
| Fluency   | **100.00%**| 88.00% | 80.00% | 80.00% | 80.00% | **+12.00%** |
| **Content avg.** | **94.66%** | 80.60% | 79.40% | 75.40% | 70.20% | **+14.06%** |

### Composite richness (the structural advantage)

<div align="center">
<img src="assets/figures/surveybench_richness.png" alt="SurveyBench composite richness" width="92%" />
</div>

Richness measures whether the generated survey contains *evidence-grounded* artifacts: figures, tables, executable code blocks, and resolved citations. **Two of four baselines produce zero**. PaperGuru reaches **43.76%**, more than **2× the strongest baseline** — and crucially this is a *file-system measurement* (no LLM judge), so the gap is preserved under judge swap and under input truncation.

📂 **All 20 generated surveys are in [`SurveyBench/`](SurveyBench/)** in three formats: [`pdf/`](SurveyBench/pdf/), [`markdown/`](SurveyBench/markdown/), and [`latex/`](SurveyBench/latex/) (full LaTeX sources for reproducibility).

---

## 🏆 Track Record

PaperGuru-assisted manuscripts have been **formally accepted** at top-tier venues across software engineering, machine learning, and engineering informatics — with thirty more under active review at NeurIPS 2026, CCS 2026, and adjacent venues.

<div align="center">
<img src="assets/figures/trophy_wall.png" alt="PaperGuru track record · 5 venues, 10 acceptances" width="100%" />
</div>

| Venue | Tier | Year | Status |
|---|---|---|---|
| **FSE 2026** (ACM, CCF-A) | Diamond | 2026 | 5 papers accepted (3 IVR + 2 Poster) |
| **ICML 2026** (CCF-A) | Diamond | 2026 | 1 long paper accepted to main proceedings |
| **TOSEM** (ACM Trans., CCF-A) | Diamond | 2026 | 2 articles accepted (under publication embargo) |
| **AEI** (Elsevier, SCI Q1) | Platinum | 2026 | 1 paper, minor revision accepted |
| **ICoGB 2026** (Civil Engineering) | Gold | 2026 | 1 cross-disciplinary paper accepted |

> **What this proves.** The same algorithmic memory that wins PaperBench (CS reproduction) and SurveyBench (CS literature synthesis) also writes the publishable manuscript that gets through peer review at venues spanning software engineering, ML, civil engineering, and engineering informatics. **Only the artifact archive differs.**

---

## 📦 What is in this repository

```
PaperGuru-Benchmark/
├── README.md                          ← you are here
├── README.zh-CN.md                    ← 中文版
├── LICENSE                            ← MIT
│
├── paper/
│   └── PaperGuru-CCM.pdf              ← the full paper (NeurIPS 2026 submission)
│
├── PaperBench/                        ← 23 reproduction submissions
│   ├── README.md
│   ├── aggregate-final.json           ← all scores in machine-readable form
│   ├── PER_PAPER_COMPARISON.md        ← per-paper PaperGuru vs baselines
│   ├── REPORT.md                      ← narrative report
│   └── submissions/
│       ├── adaptive-pruning/          ← runnable code tree, one per paper
│       ├── all-in-one/
│       ├── ...                        ← 23 directories total
│       └── what-will-my-model-forget/
│
├── SurveyBench/                       ← 20 generated surveys, three formats
│   ├── README.md
│   ├── pdf/                           ← compiled PDFs (review-ready)
│   ├── markdown/                      ← markdown source (web-friendly)
│   └── latex/                         ← full LaTeX sources (rebuild-able)
│
└── assets/
    ├── badges/                        ← 5 venue badges (transparent PNG)
    │   ├── fse.png    icml.png    tosem.png    aei.png    icogb.png
    ├── figures/                       ← every figure in this README
    │   ├── hero_banner.png
    │   ├── architecture.png
    │   ├── pipeline.png
    │   ├── before_after.png
    │   ├── trophy_wall.png
    │   ├── paperbench_bar.png         paperbench_topline.png
    │   ├── surveybench_radar.png      surveybench_richness.png
    │   ├── lift_distribution.png
    │   ├── data.json                  ← raw numbers used to build all charts
    │   ├── build_figures.py           ← rebuild data charts
    │   └── build_trophy_wall.py       ← rebuild trophy wall composite
    └── demos/
        └── pipeline_animated.svg      ← the live SVG animation
```

**Repository size**: ~350 MB (PaperBench submissions and SurveyBench LaTeX sources dominate). No file exceeds 20 MB; the repository works on standard `git`/`git lfs`-free GitHub.

---

## 🔬 Reproducing the figures

All data figures in this README are built from a single Python script with the raw numbers stored alongside as JSON. To rebuild them:

```bash
cd assets/figures
python3 -m pip install matplotlib numpy pillow
python3 build_figures.py        # rebuilds the 5 data charts + data.json
python3 build_trophy_wall.py    # rebuilds the trophy_wall.png composite
```

The raw numbers — every cell in every results table — are in [`assets/figures/data.json`](assets/figures/data.json). If you use these numbers, please cite the paper.

---

## 📜 License

This release is distributed under the [MIT License](LICENSE). The PaperBench reproduction submissions inherit the licenses of their corresponding original papers; check each subdirectory before redistributing. The SurveyBench generated surveys may be cited and quoted with attribution.

---

<div align="center">

**PaperGuru** &nbsp;·&nbsp; the missing memory primitive for long-horizon LLM agents.

<sub>Built for researchers, by researchers. Verified on the hardest published benchmarks. Carried by ten peer-reviewed publications and counting.</sub>

<br/>

[paper](paper/PaperGuru-CCM.pdf) &nbsp;·&nbsp;
[PaperBench](PaperBench/) &nbsp;·&nbsp;
[SurveyBench](SurveyBench/) &nbsp;·&nbsp;
[中文](README.zh-CN.md)

</div>
