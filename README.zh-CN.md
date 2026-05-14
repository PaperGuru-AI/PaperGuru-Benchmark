<div align="center">

<img src="assets/figures/hero_banner.png" alt="PaperGuru — 研究的光速" width="100%" />

<br/>

# PaperGuru &nbsp;·&nbsp; 长程记忆基准

**首个面向长程 LLM 智能体的「生命周期感知记忆」原语 —— 在 PaperBench 与 SurveyBench 上同时取得 SOTA，并在 FSE 2026、ICML 2026、TOSEM、AEI、ICoGB 等顶级会议/期刊获得真实录用。**

[![Paper](https://img.shields.io/badge/论文-PDF-1f3a8a?style=for-the-badge&logo=arxiv&logoColor=white)](paper/PaperGuru-CCM.pdf)
[![PaperBench](https://img.shields.io/badge/PaperBench-66.05%25-0a0a0b?style=for-the-badge)](PaperBench/)
[![SurveyBench](https://img.shields.io/badge/SurveyBench-94.66%25-0a0a0b?style=for-the-badge)](SurveyBench/)
[![Accepted](https://img.shields.io/badge/录用-10%20篇-c9a14a?style=for-the-badge)](#-真实战绩)
[![English](https://img.shields.io/badge/English-README-2563eb?style=for-the-badge)](README.md)
[![加入微信群](https://img.shields.io/badge/%E5%8A%A0%E5%85%A5-%E5%BE%AE%E4%BF%A1%E7%BE%A4-07c160?style=for-the-badge&logo=wechat&logoColor=white&labelColor=555555)](assets/badges/wechat_qr.png)

</div>

---

## 📰 一句话概述

> AI 基础设施已经收敛到三个商品化原语 —— **算力**（NVIDIA）、**模型**（前沿 LLM 权重）、**检索**（Pinecone 类向量库）。但**第四个原语 —— 带有生命周期语义的长期记忆 —— 是缺失的**，每一个长程 LLM 系统都在以糟糕的方式重复发明它。**PaperGuru** 是首个从 4 公理形式化「生命周期感知记忆」(Lifecycle-Aware Memory, LAM) 出发设计的系统，在两个最严苛的公开基准上用同一套算法机制同时取得 SOTA。

| 基准 | 指标 | PaperGuru | 已发表最强 baseline | 提升 |
|---|---|---:|---:|---:|
| **PaperBench**（OpenAI, 2025） | 23 篇论文平均复现分 | **66.05%** | 35.74% | **+30.21%** |
| **PaperBench** | 超过 41% 人类 ML 博士门槛的篇数 | **20 / 23** | 4 / 23 | **+16 篇** |
| **SurveyBench**（Yan et al., 2025） | 内容质量（5 维平均） | **94.66%** | 80.60% | **+14.06%** |
| **SurveyBench** | 复合丰富度（图·表·代码） | **43.76%** | 20.36% | **+23.40%** |
| **真实部署** | 2025 Q4 起的同行评审录用 | **10 篇** | — | 5 个会议/期刊 |

<br/>

<div align="center">
<img src="assets/figures/before_after.png" alt="使用 PaperGuru 前后对比" width="92%" />
<br/>
<sub><i>同一个研究者，同一个任务。差别只在底层的记忆原语。</i></sub>
</div>

---

## 📑 目录

- [PaperGuru 是什么](#-paperguru-是什么)
- [为什么是记忆，为什么是现在](#-为什么是记忆为什么是现在)
- [CCM 系统架构](#-ccm-系统架构)
- [核心流水线](#-核心流水线)
- [PaperBench 实验结果](#-paperbench-实验结果)
- [SurveyBench 实验结果](#-surveybench-实验结果)
- [真实战绩](#-真实战绩)
- [仓库内容](#-仓库内容)
- [复现图表](#-复现图表)
- [许可证](#-许可证)

---

## ✨ PaperGuru 是什么

PaperGuru 是一个**面向长程 LLM 智能体的记忆架构**。它**不是**又一个 RAG 库，**也不是**又一个 agent 框架。它是「生命周期感知记忆」(LAM) 这一系统原语的首个具体实现 —— 而这个原语是生产环境中的智能体一直缺失的。

LAM 的四条公理（论文 §3 形式化）：

1. **版本化内容** &nbsp;·&nbsp; 一个曾经正确的陈述，在论文修订、API 弃用、研究撤稿后必须变成「过时」状态 —— 记忆层必须知道这一点。
2. **结构化多跳相关性** &nbsp;·&nbsp; 真正相关的证据往往在「两个引用之外」，而不是「一次余弦相似度查询之内」。
3. **档案无限增长下的有界查询成本** &nbsp;·&nbsp; 文献库每天都在增长；路由成本不能跟着一起增长。
4. **可溯源的合成** &nbsp;·&nbsp; 智能体输出的每一条主张，都必须能反向追溯到记忆中可验证的具体 artifact。

PaperGuru 通过单一机制 —— **资本块记忆 (Capital Chunk Memory, CCM)** —— 同时满足全部四条公理，下文详述。

---

## 🌍 为什么是记忆，为什么是现在

LLM 的上下文窗口在 18 个月内从 4K tokens 走到 1M tokens，下一站是 **跨会话的持久化记忆** —— 这已经写进每个前沿实验室的公开路线图。LLM 部署的主要单元已经不再是「单 prompt 一次完成」，而是**长程智能体系统**：

- 一次跨数天、涉及数百个文件的软件工程会话；
- 一个从十年跨度的引文图谱出发、生成 200K-token 综述的文献研究助手；
- 一个把论文 PDF 转换成可运行代码树的复现智能体；
- 一个在给出治疗方案前先读完十年临床记录的医疗证据智能体。

在所有这些系统的公开评估中 —— **SurveyBench**、**PaperBench**、**SWE-bench-Live** —— 上限早已不是底座模型的推理能力，而是**系统在轮次之间记住了什么、又能从工作集之外检索到什么**。

---

## 🏛 CCM 系统架构

<div align="center">
<img src="assets/figures/architecture.png" alt="PaperGuru CCM 架构" width="92%" />
</div>

PaperGuru 把记忆切成两层表面：

- **Chunk Heads（块头）** —— 紧凑、有界的路由表面（每个 artifact 一个 head）
- **Chunk Contents（块内容）** —— 无界的原始文本，按需懒加载

中央 **资本块（Capital Chunk）** 索引所有 head，并支持基于**时序 artifact 图**的「资本优先路由」。该图统一了两类边：

| 边类别 | 例子 |
|---|---|
| **结构边** | `cites`、`benchmarked-on`、`introduced-by`、`implements` |
| **历史因果边** | `discussed-in`、`deprecated-by`、`retracted-by`、`superseded-by` |

查询时，上下文通过 **「先路由 → 再扩展 → 后蒸馏」** 三阶段流水线构造，最终产出紧凑的、可溯源的**证据卡片 (Evidence Cards)** —— 这是后续所有阶段操作的唯一数据结构。

> **它在生产环境为什么重要。** 平面检索在论文一被修订就崩；MemGPT 分层、Ebbinghaus 遗忘、知识图谱包装等 agent-specific 手段各自只能满足 1-2 条公理，从未同时满足 4 条。CCM 是我们所知第一个无需 per-task 调参就能同时满足 4 条公理的设计。

---

## ⚙️ 核心流水线

<div align="center">
<img src="assets/figures/pipeline.png" alt="PaperGuru CCM 流水线" width="92%" />
<br/>
<sub><i>Search → Extract → Reason → Verify。Reason 阶段内部跑的是 Compose / Critique / Mutate 循环。</i></sub>
</div>

<br/>

<div align="center">
<img src="assets/demos/pipeline_animated.svg" alt="PaperGuru 实时流水线" width="92%" />
<br/>
<sub><i>同一条流水线的 SVG 实时动画 —— Reason 阶段的三个子块按 Compose / Critique / Mutate 循环高亮，验证勾在证据卡片就位时逐一打勾。</i></sub>
</div>

| 阶段 | 输入 | 输出 | 计算占比* |
|---|---|---|---:|
| **01 · SEARCH**  | 主题查询 + 候选文献库 | 排序后的 artifact heads | ~15% |
| **02 · EXTRACT** | Heads + 块内容 | 证据卡片（文本 + 溯源） | ~20% |
| **03 · REASON**  | 证据卡片 | 草稿段落（Compose → Critique → Mutate 循环） | ~45% |
| **04 · VERIFY**  | 草稿段落 | 引用过、溯源过的最终输出 | ~20% |

<sup>* 200K-token 综述任务的典型 wall-clock 占比；不同任务略有浮动。</sup>

---

## 📊 PaperBench 实验结果

PaperBench（OpenAI 2025）是论文复现的标杆基准：每份提交是一个可运行的代码树，由叶节点 LLM 评判员按照人工撰写的评分细则打分。官方人类专家基线是 **41%（48 小时 ML 博士预算）**。

### 总览

<div align="center">
<img src="assets/figures/paperbench_topline.png" alt="PaperBench 总览对比" width="88%" />
</div>

PaperGuru 在全部 23 篇论文上取得 **66.05% 的均分**，超过所有已发表 baseline，并比 41% 的人类专家门槛高出 **+25 分**。

### 逐篇分布

<div align="center">
<img src="assets/figures/paperbench_bar.png" alt="PaperBench 逐篇对比" width="100%" />
</div>

在 20 篇有公开 baseline 的论文上：

- **19 / 20** 篇实现正向超越
- 平均提升：**+30.21% 绝对值**
- 中位数提升：**+27.0% 绝对值**
- 最大涨幅：`stay-on-topic-with-classifier-free-guidance`（+68.03%）
- 唯一回退：`pinn`（−4.47%，原 PINN baseline 已用了人工调参的领域先验）

### 提升幅度分布

<div align="center">
<img src="assets/figures/lift_distribution.png" alt="逐篇提升直方图" width="80%" />
</div>

另外 3 篇没有公开 baseline 的论文（`semantic-self-consistency`、`self-composing-policies`、`self-expansion`），PaperGuru 分别取得 **95.45%**、**65.03%**、**39.77%**。

📂 **23 个复现仓库全部位于 [`PaperBench/submissions/`](PaperBench/submissions/)。** 总分见 [`PaperBench/aggregate-final.json`](PaperBench/aggregate-final.json)，逐篇对比报告在 [`PaperBench/PER_PAPER_COMPARISON.md`](PaperBench/PER_PAPER_COMPARISON.md)。

---

## 📊 SurveyBench 实验结果

SurveyBench（Yan et al., 2025）从三个维度评估长篇综述生成 —— 内容、大纲、丰富度 —— 均使用 LLM 评判员。PaperGuru 使用官方 `claude-opus-4.7` 评判员，所有维度归一到 [0, 100%]。

### 内容质量（5 维雷达）

<div align="center">
<img src="assets/figures/surveybench_radar.png" alt="SurveyBench 雷达图" width="80%" />
</div>

PaperGuru 内容均分 **94.66%** —— 比最强 baseline（AutoSurvey 80.60%）高 **+14.06% 绝对值** —— 在 **Focus** 和 **Fluency** 上都触顶。

| 维度 | PaperGuru | AutoSurvey | LLM×MR-v2 | SurveyForge | ASur | 提升 |
|---|---:|---:|---:|---:|---:|---:|
| Coverage（覆盖度） | **94.00%** | 61.00% | 70.00% | 60.00% | 59.00% | **+24.00%** |
| Coherence（连贯性） | **87.40%** | 80.00% | 78.00% | 79.00% | 76.00% | **+7.40%** |
| Depth（深度） | **92.00%** | 75.00% | 75.00% | 72.00% | 60.00% | **+17.00%** |
| Focus（聚焦度） | **100.00%** | 99.00% | 94.00% | 86.00% | 76.00% | **+1.00%** |
| Fluency（流畅度） | **100.00%** | 88.00% | 80.00% | 80.00% | 80.00% | **+12.00%** |
| **内容均分** | **94.66%** | 80.60% | 79.40% | 75.40% | 70.20% | **+14.06%** |

### 复合丰富度（结构性优势）

<div align="center">
<img src="assets/figures/surveybench_richness.png" alt="SurveyBench 丰富度对比" width="92%" />
</div>

丰富度衡量综述是否包含**有据可查的 artifact**：图、表、可执行代码块、解析过的引用。**4 个 baseline 中有 2 个直接是 0**。PaperGuru 取得 **43.76%**，超过最强 baseline **2 倍以上** —— 关键在于这是一个**纯文件系统测量**（无需 LLM 评判员），所以这个差距在更换评判员或更换截断策略时都能稳定保持。

📂 **20 篇生成的综述全部位于 [`SurveyBench/`](SurveyBench/)**，提供三种格式：[`pdf/`](SurveyBench/pdf/)、[`markdown/`](SurveyBench/markdown/)、[`latex/`](SurveyBench/latex/)（含完整 LaTeX 源码可重新编译）。

---

## 🏆 真实战绩

PaperGuru 辅助撰写的论文已在跨学科顶级会议/期刊正式录用 —— 还有 30+ 篇正在 NeurIPS 2026、CCS 2026 及邻近顶会评审中。

<div align="center">
<img src="assets/figures/trophy_wall.png" alt="PaperGuru 战绩 · 5 个 venue 10 篇录用" width="100%" />
</div>

| Venue | 等级 | 年份 | 状态 |
|---|---|---|---|
| **FSE 2026**（ACM, CCF-A） | Diamond | 2026 | 5 篇录用（3 IVR + 2 Poster） |
| **ICML 2026**（CCF-A） | Diamond | 2026 | 1 篇 long paper 录用主会程 |
| **TOSEM**（ACM Trans., CCF-A） | Diamond | 2026 | 2 篇录用（出版 embargo 中） |
| **AEI**（Elsevier, SCI Q1） | Platinum | 2026 | 1 篇 minor revision 录用 |
| **ICoGB 2026**（土木工程） | Gold | 2026 | 1 篇跨学科论文录用 |

> **这证明了什么。** 同一套算法记忆 —— 在 PaperBench（CS 复现）和 SurveyBench（CS 文献综述）上拿 SOTA 的同一套机制 —— 同时也能写出**通过同行评审**的可发表手稿，跨越软件工程、ML、土木、工程信息学。**变的只有 artifact 库。**

---

## 📦 仓库内容

```
PaperGuru-Benchmark/
├── README.md                          ← 英文版
├── README.zh-CN.md                    ← 你正在看
├── LICENSE                            ← MIT
│
├── paper/
│   └── PaperGuru-CCM.pdf              ← 完整论文（NeurIPS 2026 投稿稿）
│
├── PaperBench/                        ← 23 个复现仓库
│   ├── README.md
│   ├── aggregate-final.json           ← 机器可读的全部分数
│   ├── PER_PAPER_COMPARISON.md        ← 逐篇 PaperGuru vs baselines
│   ├── REPORT.md                      ← 叙述性报告
│   └── submissions/
│       ├── adaptive-pruning/          ← 可运行代码树，每篇论文一个
│       ├── all-in-one/
│       ├── ...                        ← 共 23 个目录
│       └── what-will-my-model-forget/
│
├── SurveyBench/                       ← 20 篇生成的综述，3 种格式
│   ├── README.md
│   ├── pdf/                           ← 编译好的 PDF（可直接审阅）
│   ├── markdown/                      ← markdown 源码（网页友好）
│   └── latex/                         ← 完整 LaTeX 源码（可重新编译）
│
└── assets/
    ├── badges/                        ← 5 个 venue badge（透明 PNG）
    │   ├── fse.png    icml.png    tosem.png    aei.png    icogb.png
    ├── figures/                       ← 本 README 用到的全部图
    │   ├── hero_banner.png
    │   ├── architecture.png
    │   ├── pipeline.png
    │   ├── before_after.png
    │   ├── trophy_wall.png
    │   ├── paperbench_bar.png         paperbench_topline.png
    │   ├── surveybench_radar.png      surveybench_richness.png
    │   ├── lift_distribution.png
    │   ├── data.json                  ← 所有图表的原始数据
    │   ├── build_figures.py           ← 重建数据图脚本
    │   └── build_trophy_wall.py       ← 重建战绩墙拼接脚本
    └── demos/
        └── pipeline_animated.svg      ← SVG 实时动画
```

**仓库大小**: 约 350 MB（PaperBench 复现仓库 + SurveyBench LaTeX 源码占大头）。无单文件超过 20 MB；标准 GitHub 仓库即可承载，无需 git-lfs。

---

## 🔬 复现图表

本 README 中所有数据图都由一个 Python 脚本从 JSON 原始数据重建。要重新生成：

```bash
cd assets/figures
python3 -m pip install matplotlib numpy pillow
python3 build_figures.py        # 重建 5 张数据图 + data.json
python3 build_trophy_wall.py    # 重建 trophy_wall.png 拼接图
```

所有原始数字 —— 论文中每张表的每个单元格 —— 都在 [`assets/figures/data.json`](assets/figures/data.json)。如果你使用了这些数据，请引用论文。

---

## 📜 许可证

本发布在 [MIT License](LICENSE) 下分发。PaperBench 的复现仓库继承各自原论文的许可证，转发布前请检查每个子目录。SurveyBench 生成的综述可在标注出处的前提下引用与摘录。

---

<div align="center">

**PaperGuru** &nbsp;·&nbsp; 长程 LLM 智能体缺失的那个记忆原语。

<sub>研究者为研究者打造。在最严苛的公开基准上验证。由 10 篇同行评审录用论文背书 —— 而且还在不断累积。</sub>

<br/>

[论文](paper/PaperGuru-CCM.pdf) &nbsp;·&nbsp;
[PaperBench](PaperBench/) &nbsp;·&nbsp;
[SurveyBench](SurveyBench/) &nbsp;·&nbsp;
[English](README.md)

</div>
