---
layout: post
title: "【论文笔记】RewardHarness：用上下文演化替代参数优化的图像编辑奖励建模"
date: 2026-05-19
categories: [paper-notes]
tags: [reward-model, agent, self-evolution, image-editing, RLHF]
paper_title: "RewardHarness: Self-Evolving Agentic Post-Training"
paper_authors: "Yuxuan Zhang et al."
paper_link: "https://arxiv.org/abs/2605.08703"
---

> **论文**：[RewardHarness: Self-Evolving Agentic Post-Training](https://arxiv.org/abs/2605.08703)
> **作者**：Yuxuan Zhang, Penghui Du, Bo Li, Cong Wei, Junwen Miao, Huaisong Zhang, Songcheng Cai, Yubo Wang, Dongfu Jiang, Yuyu Zhang, Ping Nie, Wenhu Chen, Changqian Yu, Kelsey R. Allen
> **机构**：University of British Columbia, Vector Institute, Kuaishou Kolors Team, CMU, University of Waterloo, Tsinghua University, Georgia Tech

## 一句话总结

当前 reward model 需要大量人类偏好标注 + 参数训练，RewardHarness 提出了一条不同路径：冻结 VLM 参数不动，只用 100 个偏好示例通过迭代演化一套 Skills & Tools 库来获得 reward 能力。在图像编辑评估 benchmark 上超过 GPT-5 达 5.3 个百分点。

## 问题：Reward Modeling 的数据效率瓶颈

指令引导的图像编辑（instruction-guided image editing）进步很快，但评估仍是瓶颈。现有方法依赖大规模人类偏好标注训练 reward model（如 EditReward 使用 200K 偏好对），成本高、不透明、且不适用于 API-only 的闭源模型。

一个值得注意的不对称性：人类标注员往往从少量校准样本就能内化评估标准并一致地应用，而模型通常需要数十万标注对才能获得类似的偏好判断能力。

论文的核心问题是：如果人类能从少量示范中习得图像编辑偏好，模型能否仅通过上下文演化、不更新任何参数就做到同样的事？

## 方法：上下文演化作为 Reward Modeling

![RewardHarness 范式对比](/assets/images/reward-harness/x1.png)
*Figure 1: 传统范式收集大规模偏好数据训练 reward model；RewardHarness 从少量示范出发，通过自演化生成 Skills & Tools 库，构建可解释的 reward 系统。*

RewardHarness 的架构由两个核心组件构成：

### Orchestrator + Sub-Agent 的双层结构

![系统架构](/assets/images/reward-harness/x2.png)
*Figure 2: RewardHarness 完整流程。Orchestrator 从库中选择相关 Skills 和 Tools 注入 Sub-Agent 上下文，Sub-Agent 构建推理链产出偏好判断，反馈回路驱动库的演化。*

给定源图像 $I_s$、编辑指令 $p$、$K$ 个候选编辑图像 $\{I_1, \ldots, I_K\}$，系统输出标量偏好分数和排序：

$$\mathbf{s}, \pi = \mathcal{M}(I_s, \{I_k\}_{k=1}^K, p, \mathcal{C})$$

其中 $\mathcal{M}$ 是冻结的 VLM（默认 Qwen2.5-VL-7B），$\mathcal{C}$ 是 Orchestrator 从库中检索并组装的上下文。模型参数从不更新。

**Orchestrator**（基于 Claude 的 LLM）有两个角色：
- 推理时：检查输入，路由选择最相关的 Skills 和 Tools，组装上下文 $\mathcal{C}$
- 演化时：分析推理链的成功/失败，提出库更新

**Sub-Agent**（冻结 VLM）的推理链分三步：
1. Rubric application：对每个 Skill，按评分准则评估每张候选图
2. Tool-guided analysis（可选）：满足调用条件时，执行针对性视觉分析（OCR、空间关系验证、物体计数等）
3. Aggregation and ranking：综合所有评估结果产出分数和排序

### Skills 与 Tools：两类评估知识

![Skills 和 Tools 示例](/assets/images/reward-harness/x3.png)
*Figure 3: 库中 Skill 和 Tool 的示例（来自迭代 69）。*

**Skill** 是结构化的 Markdown 评估指南，包含名称、描述、评分 rubric 和应用示例。属于声明性知识——定义"评什么"和"如何打分"。例如 `realism-and-artifact-penalties` 区分视觉伪影（必须扣分）和概念性不现实（如果指令明确要求则可接受）。

**Tool** 是结构化的 Markdown 程序性规范，定义名称、目的、输入/输出、调用条件和执行协议。属于过程性知识——让通用 VLM 临时充当特定视觉分析专家。例如 `text-and-ocr-analyzer` 指导 Sub-Agent 提取、比较和验证源图与编辑图中的文字内容。

两者的关键区别：Skills 回答"按什么标准评"，Tools 回答"如何执行特定检查"。

### 自演化循环

演化的输入是仅 $N=100$ 个人类偏好示范，拆分为训练集（60 例）和验证集（40 例）。每轮迭代包含五步：

1. **Evaluation**：用当前库对训练集样本进行评估
2. **Scoring**：将预测排序与人类标签对比，区分正确/错误
3. **Chain analysis**：Orchestrator 对推理链做根因分析——错误来自缺失评估维度（需新 Skill）、错误 rubric 应用（需修改 Skill）、还是感知幻觉（需新/改 Tool）
4. **Library update**：创建、修改或弃用库中条目
5. **Validation and gating**：在验证集上评估，仅当准确率提升时接受更新，否则回滚

关键细节：
- 库从空开始，先膨胀后收缩。峰值时有 13 条（8 Skills + 5 Tools），约迭代 50 后进入剪枝阶段
- 最终库仅 7 条（3 Skills + 4 Tools），验证准确率从空库的 42.5% 提升到 62.5%
- Skill 提案的接受率低于 Tool 提案——修改声明性 rubric 更容易产生回归

## 实验结果

### 图像编辑评估 Benchmark

在 EditReward-Bench（K=2/3/4）和 GenAI-Bench 上的表现：

| 方法 | K=2 | K=3 | K=4 | GenAI | Avg. | $\Delta$ |
|------|-----|-----|-----|-------|------|----------|
| GPT-4o | 45.7 | 27.3 | 7.3 | 53.5 | 33.5 | -- |
| GPT-5 | 57.5 | 38.5 | 12.8 | 59.6 | 42.1 | +8.6 |
| Gemini-2.5-Flash | 58.6 | 39.9 | 12.2 | 57.0 | 41.9 | +8.4 |
| Qwen2.5-VL-7B (vanilla) | 52.7 | 24.7 | 3.4 | 40.5 | 30.3 | -3.2 |
| EditReward (Qwen, 200K data) | 57.0 | 36.0 | 10.8 | 64.0 | 42.0 | +8.5 |
| EditReward (MiMo, 200K data) | 56.5 | 42.7 | 11.5 | 65.7 | 44.1 | +10.6 |
| **RewardHarness (Qwen)** | 57.9 | **46.7** | 10.8 | **67.5** | 45.7 | +12.2 |
| **RewardHarness (Gemini-2.0-Flash)** | **66.2** | 45.3 | **13.5** | 64.4 | **47.4** | +13.9 |

几个值得注意的数字：
- 同样是 Qwen2.5-VL-7B 作为骨干，vanilla 版本 30.3，加上演化库后达 45.7，提升 15.4 个百分点
- 仅用 EditReward 0.05% 的偏好数据（100 vs 200K），就超过了在全量数据上 SFT 训练的 EditReward
- GenAI-Bench 上 67.5 的得分说明学到的 Skills/Tools 捕获了通用编辑质量标准，而非 benchmark-specific 的 heuristic

### 作为 GRPO Reward Signal 的下游验证

将 RewardHarness 的分数用于 FLUX.2-klein-base-4B 的 GRPO fine-tuning，在 ImgEdit-Bench 上：

| 方法 | Overall |
|------|---------|
| FLUX.2-klein-base-4B (base) | 3.32 |
| +RL (EditReward) | 3.45 |
| +RL (RewardHarness) | **3.52** |
| Flux.1 Kontext [dev] (参考) | 3.52 |

用 RewardHarness 做 reward signal 的 RL-tuned 模型达到了 Flux.1 Kontext [dev] 的水平，而后者是一个更大的模型。

### 演化动态

![演化动态](/assets/images/reward-harness/analysis_evolution.png)
*Figure 6: 77 轮迭代中的自演化动态。左：验证准确率（点为每轮提案，实线为 best-so-far）；右：Skills 和 Tools 数量随时间变化。*

验证准确率在库膨胀期（~13 条）时 plateau 在 52.5%，进入剪枝阶段后继续提升，最终在迭代 69 达到 62.5%。这暗示"少即是多"——过多的 Skills 可能让 Sub-Agent 的评估变得冗余或矛盾。

## 值得讨论的几个点

**上下文演化 vs. 参数优化作为 reward capability 的获取路径**。论文提出了一个有意思的观察：reward capability 不一定要通过梯度更新来获得。将评估知识外化为可编辑的文档（Skills、Tools），让一个冻结模型通过读取这些文档来获得 reward 能力，这条路在数据效率上有明显优势。当然代价是推理时需要更长的 context 和更复杂的 pipeline。

**Orchestrator 对闭源模型的依赖**。当前 Orchestrator 使用 Claude，这在可复现性和成本上是个限制。Orchestrator 需要做路由、根因分析和库更新提案，这些任务对模型能力要求较高。用开源 LLM 替换 Orchestrator 是否可行，论文承认尚未验证。

**库的容量上限与自修剪现象**。最终库仅 3 Skills + 4 Tools，非常精简。这说明在当前任务上，评估知识的维度可能本身就不多。对更复杂的评估场景（如视频编辑、3D 场景操控），是否还能保持这样的精简性是个开放问题。有意思的是，系统自发地学会了"先探索后收缩"的演化策略，这与人类专家积累经验后提炼核心原则的过程有相似之处。
