---
layout: post
title: "【论文笔记】AgenticDataBench：数据科学 Agent 的全面基准测试"
date: 2026-07-05
categories: [paper-notes]
tags: [benchmark, data-agent, LLM-agent, data-science, skill-extraction]
paper_title: "AgenticDataBench: A Comprehensive Benchmark for Data Agents"
paper_authors: "Zhaoyan Sun et al."
paper_link: "https://arxiv.org/abs/2607.01647"
---

> **论文**：[AgenticDataBench: A Comprehensive Benchmark for Data Agents](https://arxiv.org/abs/2607.01647)
> **作者**：Zhaoyan Sun, Shan Zhong, Daizhou Wen, Jiaxing Han, Guoliang Li, Ying Yan, Peng Zhang, Yu Su, Xiang Qi, Baolin Sun, Chengyuan Yang, Tao Fang, Huaiyu Ruan
> **机构**：Tsinghua University, Ant Digital Technologies (Ant Group)

## 数据科学 Agent 的评测困境

LLM 驱动的 data agent（能自动完成数据探索、清洗、建模、可视化等端到端数据科学流程的 agent）正在快速涌现，但现有 benchmark 存在明显不足：任务类型有限、缺少真实业务场景、评估粒度太粗（只有 task-level 分数，看不出 agent 在哪个环节失败）。AgenticDataBench 试图从三个维度补上这些缺口——覆盖 15 个垂直领域的真实数据、基于 "data science skill" 的细粒度标注、以及包含蚂蚁集团 5 个真实 B2B 业务场景的工业级复杂任务。

## Benchmark 构建方法

![Overview](/assets/images/agentic-data-bench/overview-v3.png)
*AgenticDataBench 构建流程总览：从技能提取到任务选择与生成。*

### 核心概念：Data Science Skill

论文将数据科学操作中反复出现的模式抽象为 "skill"，例如"处理缺失值""时间序列对齐""模型训练与推理"等。这些 skill 构成一个层次树：高层节点是泛化能力（如"数据预处理"），叶节点是具体操作（如"用中位数填充缺失值"）。skill 分为 7 大类：

1. Data Format Handling（数据格式解析）
2. Data Preprocessing（清洗、转换、特征工程）
3. Data Manipulation（重构、过滤、合并）
4. Data Analysis（统计分析、聚合）
5. Data Modeling（建模、训练、评估）
6. Data Visualization（图表生成）
7. Cross-Stage Skills（环境管理、SQL、错误处理等跨阶段通用技能）

最终提取出 433 个代表性 skill，来源于 6,510 个 Stack Overflow 数据科学问答的解决方案。

### 层次化技能提取

从 Stack Overflow 的 6,510 个数据科学任务解答出发，提取过程分四步：

1. **LLM 分解**：用 LLM 将每个解答拆解为逐步的 skill 描述，得到 29,602 条 stepwise skill
2. **Embedding 聚类**：用 Qwen3-Embedding 编码 skill 描述，UMAP 降维后用 GMM 软聚类
3. **LLM 精炼**：对每个 cluster，用 LLM 将细粒度 skill 合并为更高层的抽象 skill，用 DBSCAN 去重同义 skill。如果 skill 数量仍然过多，递归执行 cluster-and-refine
4. **人工审核**：数据科学专家最终审核 433 个顶层 skill 的质量

![Skill Distribution](/assets/images/agentic-data-bench/skill-distribution-v4.png)
*433 个 skill 在 7 大类中的分布。*

### 基于 Skill 的 Benchmark 构建

benchmark 的任务来自两条路径：

**路径 1：真实业务任务选择。** 蚂蚁集团 5 个业务线的 30 位领域专家花费 600 人时整理了 600 个生产级任务。由于很多任务的操作模式重复（只是参数/数据不同），用贪心算法选择 skill 覆盖最大化的子集，最终保留 102 个任务。

**路径 2：Skill 覆盖驱动的任务生成。** 对于缺少现成任务的公开数据集（10 个领域、58 个数据集），构建 skill graph（节点=skill，边=任务解答中相邻 skill 的共现关系），然后：
- 从 skill graph 中采样路径作为目标 skill 组合
- 用 LLM 生成结构化数据 profile → 基于 skill 的 workflow → task description
- 动态降低已覆盖 skill 的采样权重以促进多样性
- 8 位专家花费 960 人时标注解答和评测方法

最终得到 242 个生成任务，加上 102 个真实业务任务，总计 344 个 benchmark 实例。

### Benchmark 规模

| 指标 | AgenticDataBench | DSBench | DA-Code | DataSciBench |
|------|-----------------|---------|---------|--------------|
| 覆盖 Skill 数 | 433 | -- | -- | 281 |
| 每任务标签数 | 433 | 2 | 10 | 6 |
| 平均代码行数 | 113.6 | -- | 85 | 33.4 |
| 每任务数据量 | 493.4 MB | 11.1 MB | 23.0 MB | 0.8 MB |
| 数据来源 | 真实业务+公开 | 公开竞赛 | 公开数据 | 公开数据 |

总数据量 27.3 GB，18 种文件格式，1.23 亿行，3.5 万个属性。

## 评测系统

![Pipeline](/assets/images/agentic-data-bench/pipeline.png)
*AgenticDataBench 评测流水线：任务实例 → Data Agent → Docker 执行器 → 排行榜。*

评测在 Docker 环境中执行，支持 5 种评分模式：表格匹配、建模指标（如 MSE 归一化）、JSON 匹配、图表匹配、文本匹配。除了 task-level 得分，还通过 LLM 对比 agent 解答与 ground truth，识别出每个 skill 应用的得分，从而定位失败环节。

## 实验结果

### 被评测的系统

4 种 agent harness × 3 种 LLM 的组合：
- Harness：DA-Agent（数据科学专用）、Smolagents（通用 ReAct）、Claude Code、CodeX
- LLM：Qwen3.5-397B-A17B、Kimi-K2.5、Claude Sonnet 4.6

### 整体表现

| Agent | Score (%) |
|-------|-----------|
| CodeX (Kimi-K2.5) | **48.8** |
| Smolagents (Qwen3.5) | 47.1 |
| Smolagents (Claude 4.6) | 46.7 |
| Claude Code (Claude 4.6) | 46.6 |
| DA-Agent (Claude 4.6) | 46.1 |
| DA-Agent (Kimi-K2.5) | 44.8 |
| CodeX (Qwen3.5) | 39.9 |
| CodeX (Claude 4.6) | 31.6 |

几个关键发现：

1. **通用 agent harness 优于数据科学专用 harness**：CodeX > Smolagents > Claude Code > DA-Agent（按最佳 LLM 配对）。DA-Agent 的轻量设计（固定 memory window、1 分钟 step timeout）在复杂场景下成为瓶颈。

2. **最佳 LLM 因 harness 而异**：Claude 4.6 在 DA-Agent 和 Claude Code 中最强（coding 能力好，语法错误少），Qwen3.5 在 Smolagents 中最强（其他模型不适应其 prompt 格式），Kimi-K2.5 在 CodeX 中最强（善于利用 CodeX 的并发探索机制）。特别值得注意的是，Claude 4.6 在 CodeX 中得分仅 31.6%（所有组合最低），这是一个典型的 harness-model mismatch 现象——Claude 的谨慎风格与 CodeX 的激进探索设计严重冲突，导致 Claude 频繁提前终止且无法利用 CodeX 的 context auto-compaction。关于这一现象的详细分析，参见 [为什么 Claude 在 CodeX harness 中表现崩塌](/2026/07/05/claude-codex-mismatch.html)。

3. **没有任何 agent 在所有领域最优**——不同 harness 在不同领域有各自优势。

### 成本与效率

![Cost-Score Tradeoff](/assets/images/agentic-data-bench/cost_score_scatter.png)
*各 agent 的 cost-score 散点图。Smolagents (Qwen3.5) 和 DA-Agent (Qwen3.5) 在低成本区间表现优异。*

| Agent | 平均步数 | Token (K) | 成本 ($) | 成功步比例 |
|-------|---------|-----------|----------|-----------|
| CodeX (Kimi-K2.5) | 40.2 | 1091.2 | 0.19 | 59.5% |
| Smolagents (Qwen3.5) | 18.1 | 319.4 | 0.07 | 94.6% |
| Claude Code (Claude 4.6) | 16.8 | 408.3 | 0.47 | 96.7% |
| DA-Agent (Qwen3.5) | 20.2 | 263.0 | 0.06 | 95.5% |

CodeX (Kimi-K2.5) 得分最高但 step 成功率只有 59.5%——它通过大量探索性尝试和快速失败来找到正确路径。Claude 4.6 一致比开源替代品贵得多，但在 Claude Code harness 中的 token 利用率最高。

### Skill-Level 分析

![Skill Category](/assets/images/agentic-data-bench/skill_type_boxplot.png)
*各 skill 类别的得分对比。Data Format Handling 和 Cross-Stage Skills 普遍得分较高，Data Analysis 失败率最高。*

- 全局最弱 skill：Data Alignment & Merging、Histogram Creation、Text Processing and Cleaning——agent 在处理异构和非结构化数据上普遍薄弱
- Claude 4.6 在 Statistical Modeling and Uncertainty 上一致强于其他 LLM（更倾向调用成熟的 Python 包如 ARIMA、t-distribution）
- DA-Agent 在 Model Training and Inference 上持续最弱（1 分钟 step timeout 会终止高维特征表的训练）

### 失败模式分析

![Error Distribution](/assets/images/agentic-data-bench/error_category_dist.png)
*各 agent 和各领域的失败类别分布。*

Data Analysis 占最大失败比例（尽管不是调用最频繁的 skill 类别），原因是数据验证、统计计算等环节的错误会向下游传播。Global Limit Exceeded 和 Self-Repair Failure 反映了 LLM 与 harness 之间的适配问题。消融实验表明，单纯增加执行步数上限或 step timeout 并不能改善性能——只是延长了无效循环。

## 讨论

AgenticDataBench 的主要贡献在于提出了 skill-based 的评测框架，使得 "agent 在哪个环节失败" 变得可量化、可比较。433 个 skill + 344 个任务 + 15 个领域的规模也确实比现有 benchmark 大一个量级。几个值得关注的点：

- **LLM-harness 适配性**比模型能力本身更重要：最强的 LLM（Claude 4.6）在不适配的 harness（CodeX）中得分只有 31.6%，是所有组合中最差的。这说明 agent 系统设计中，prompt 格式、memory 管理、并发策略等工程选择的影响不亚于底层模型的选择。
- **轻量 agent 的成本效益**：DA-Agent (Qwen3.5) 平均每个任务 $0.06，得分 44.4%；CodeX (Kimi-K2.5) 平均 $0.19，得分 48.8%。在成本敏感场景下，轻量方案可能是更实际的选择。
- **真实业务任务的加入**是关键差异化：平均每任务 493.4 MB 数据、113.6 行解答代码，远超公开 benchmark（通常 < 50 MB、< 40 行代码）。

## AI 犀利评判

**skill 提取的"代表性"缺乏严格验证。** 433 个 skill 号称代表了数据科学的核心操作模式，但整个提取过程（LLM 分解 → 聚类 → LLM 精炼 → 人工审核）中的每一步都依赖 LLM 的主观判断和人工审核的品味。论文没有给出 skill 集合的覆盖率度量——面对一个全新领域的数据科学任务，这 433 个 skill 能覆盖多少比例？所谓的"comprehensive"缺少定量支撑。

**任务生成的循环论证问题。** 用从 Stack Overflow 提取的 skill 来生成新任务，再用这些任务评测 agent，那么评测结果本质上反映的是"agent 在 Stack Overflow 常见模式上的表现"，而非"真实数据科学能力"。生产环境中的数据科学问题往往涉及大量 domain-specific 知识、数据质量判断、多次试错，这些在标准化 benchmark 中很难体现。

**Skill-level 评分依赖 LLM 打分，可靠性存疑。** 论文的 fine-grained 评测核心是"用 LLM 判断 agent 的哪个 skill 应用出了问题"。但 LLM 本身的判断准确率是多少？没有人工验证的 inter-annotator agreement 或 accuracy 数字。如果 skill-level 评分本身不可靠，那基于它的所有分析（哪个 agent 在哪个 skill 上强/弱）都要打问号。

**344 个任务在 12 种 agent 配置上只跑了一次。** 没有重复实验、没有置信区间。agent 执行有大量随机性（LLM 采样、执行环境状态），单次运行的得分波动可能不小。例如 Smolagents (Qwen3.5) 47.1% vs Smolagents (Claude 4.6) 46.7%，这 0.4% 的差距在统计上几乎没有意义。

**工业级任务的不可复现性。** 蚂蚁集团的 5 个业务领域、102 个真实任务使用的是内部数据，外部研究者无法复现这部分结果。Benchmark 开源的只是公开数据集上的 242 个任务。论文的 headline 是"包含真实业务场景"，但对社区来说实际可用的只有不到 70% 的内容。

**表面上是 benchmark paper，实质上也在卖 platform。** 评测流程与蚂蚁的 Bailian 平台深度绑定（skill 标注使用其异步批处理推理），GitHub 仓库是否能脱离这套基础设施独立运行、评测成本如何，论文未充分说明。

总体判断：一个规模可观、工程投入扎实的 benchmark 工作，skill-based 评测框架的想法有价值，但 skill 提取和评分环节的可靠性验证不足，且工业数据的不可复现削弱了论文声称的"comprehensive"。
