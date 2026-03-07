---
layout: post
title: "【论文笔记】Trust or Escalate: 给 LLM Judge 加上人类一致性的统计保证"
date: 2026-03-07
categories: [paper-notes]
tags: [LLM-as-judge, evaluation, calibration, selective-prediction, conformal]
paper_title: "Trust or Escalate: LLM Judges with Provable Guarantees for Human Agreement"
paper_authors: "Jaehun Jung et al."
paper_link: "https://arxiv.org/abs/2407.18370"
---

# Trust or Escalate: 不确定就换更强的裁判

> **论文**：[Trust or Escalate: LLM Judges with Provable Guarantees for Human Agreement](https://arxiv.org/abs/2407.18370)
> **作者**：Jaehun Jung, Faeze Brahman, Yejin Choi
> **机构**：University of Washington, Allen Institute for AI

---

"用 GPT-4 打分"已经成了 LLM 评测的默认选项，但一个根本性的问题很少被正面回答：**GPT-4 的评分到底在多大程度上能代替人类？** 不同任务、不同样本之间的可靠性差异巨大，而 GPT-4 本身往往对此过度自信。这篇论文的思路是：与其盲目信任一个模型的判断，不如先评估它的置信度——**不够确定就弃权（abstain），交给更强的模型来判**。通过统计假设检验来校准弃权阈值，可以为"LLM 评分与人类一致"提供严格的概率保证，同时让大部分评测由廉价的小模型完成。

---

## 核心框架：Selective Evaluation

### 基本设定

给定一个 LLM judge $f_{\text{LM}}$（输入一对生成结果，输出偏好标签）和一个置信度函数 $c_{\text{LM}}(x) \in [0,1]$，selective evaluator 的工作方式很简单：

- 如果 $c_{\text{LM}}(x) \geq \lambda$：信任模型判断，输出 $f_{\text{LM}}(x)$
- 如果 $c_{\text{LM}}(x) < \lambda$：弃权，不输出

关键在于阈值 $\lambda$ 怎么选。论文的目标是：给定用户指定的风险容忍度 $\alpha$，保证被评测的样本中，LLM 与人类的一致率 $\geq 1 - \alpha$。

### 阈值校准：从假设检验到严格保证

用一个小规模标注集 $D_{\text{cal}}$（约 500 条带人类标注的样本），对每个候选阈值 $\lambda$ 计算经验风险（不一致比例）：

$$\hat{R}(\lambda) = \frac{1}{n(\lambda)} \sum_{(x, y_{\text{human}}) \in D_{\text{cal}}} \mathbb{1}\{f_{\text{LM}}(x) \neq y_{\text{human}} \wedge c_{\text{LM}}(x) \geq \lambda\}$$

由于经验风险服从二项分布，可以计算精确的 $(1-\delta)$ 置信上界 $\hat{R}^+(\lambda)$。然后用 **fixed sequence testing**（从最严格的 $\lambda$ 开始向下搜索，找到最后一个满足 $\hat{R}^+(\lambda) \leq \alpha$ 的值）来确定最终阈值 $\hat{\lambda}$。

**Theorem 1** 保证：这样选出的 $\hat{\lambda}$ 满足 $P(f_{\text{LM}} \text{ 与人类一致} \mid c_{\text{LM}}(x) \geq \hat{\lambda}) \geq 1 - \alpha$，概率至少为 $1 - \delta$。

这里的保证是对单次校准数据的条件保证（conditional guarantee），比 conformal prediction 常见的 marginal guarantee 更强。

---

## Simulated Annotators：让模型学会"不确定"

保证的严格性不依赖于置信度函数的具体形式——任何近单调的置信度度量都能用。但**置信度的质量直接决定了 coverage**（有多少样本不用弃权）。如果置信度估计很差，要么保证不住就得大量弃权，要么覆盖率很低。

论文发现，常用的两种置信度方法都有严重的 over-confidence 问题：

- **Predictive probability**（取预测标签的 token 概率）：ECE 高达 0.2-0.4
- **Verbalized confidence**（让模型自己说一个置信度数字）：甚至更差

![Figure 2: 校准曲线对比 — Simulated Annotators 将 ECE 降低约 50%，显著缓解 over-confidence](/assets/images/cascaded-selective-eval/x2.png)

受人类评测实践的启发（多个标注者独立标注，看一致率），作者提出 **Simulated Annotators**：用 $N$ 组不同的 few-shot examples（每组 $K$ 条，来自不同标注者）分别 prompt 模型 $N$ 次，取投票一致比例作为置信度：

$$c_{\text{LM}}(x) = \max_y \frac{1}{N} \sum_{j=1}^{N} p_{\text{LM}}(y | x; \text{examples}_j)$$

直觉：当多个"模拟标注者"意见不一致时，说明该样本本身就有主观性，模型不该过于确定。

实验显示这个方法效果显著（ECE 即 Expected Calibration Error，衡量模型输出的置信度与实际准确率之间的偏差，越低说明校准越好）：

| Judge 模型 | 方法 | ECE $\downarrow$ | AUROC | AUPRC |
|:---|:---|:---:|:---:|:---:|
| GPT-4-turbo | Predictive Probability | 0.217 | 0.642 | 0.852 |
| GPT-4-turbo | Verbalized Confidence | 0.215 | 0.550 | 0.774 |
| GPT-4-turbo | **Simulated Annotators** | **0.095** | **0.723** | **0.877** |
| Mistral-7B | Predictive Probability | 0.374 | 0.457 | 0.579 |
| Mistral-7B | **Simulated Annotators** | **0.075** | **0.632** | **0.772** |

一个意外的发现：弱模型（Mistral-7B）经过 Simulated Annotators 后的 ECE 甚至优于 GPT-4 的 baseline 方法。也就是说，弱模型虽然准确率低，但它"知道自己不知道"的能力反而不差。

---

## Cascaded Selective Evaluation：从便宜模型开始

既然弱模型能准确估计自己的置信度，一个自然的想法是：先让便宜的模型判，不确定的再交给贵的模型。

![Figure 1: Cascaded Selective Evaluation 流程 — 从 Mistral-7B 开始，不确定的交 GPT-3.5，还不确定的交 GPT-4](/assets/images/cascaded-selective-eval/x1.png)

每个 judge 模型有各自的置信阈值 $\lambda_i$，在同一个校准集上通过扩展的 fixed sequence testing 一起校准（详见论文附录 A.2）。整个 cascade 的风险保证覆盖所有层级的模型。

---

## 实验结果

### TL;DR 摘要评测

目标一致率 $1 - \alpha = 0.9$，使用 Mistral-7B → GPT-3.5 → GPT-4 的 cascade：

| 方法 | 评测组成 (Mistral / 3.5 / GPT-4) | Coverage | 保证成功率 |
|:---|:---:|:---:|:---:|
| GPT-4 不弃权 | 0 / 0 / 100% | 100% | 0% |
| 启发式选择 (GPT-4) | 0 / 0 / 100% | 89.6% | 42.0% |
| 启发式 Cascade | 59.6 / 15.0 / 25.5% | 64.6% | 0% |
| **Cascaded Selective Eval** | **28.3 / 28.2 / 43.5%** | **55.7%** | **90.8%** |

GPT-4 不弃权时，1000 次随机实验中没有一次达到 90% 的人类一致率。而 Cascaded Selective Evaluation 在 90.8% 的实验中成功达标，且超过一半的评测由 Mistral-7B 和 GPT-3.5 完成。

![Figure 3: TL;DR 结果 — 保证的一致率远超 GPT-4 无弃权（左），大部分评测由弱模型完成（右）](/assets/images/cascaded-selective-eval/x3.png)

### ChatArena 对话评测

![Figure 4: ChatArena 结果 — 目标一致率保证达成（左），评测成本构成（右）](/assets/images/cascaded-selective-eval/x4.png)

在 ChatArena 上，GPT-4 无弃权的一致率约 77.8%，几乎不可能达到 80%。Cascaded Selective Evaluation 以 80% 为目标：

| 配置 | 一致率 | Coverage | 保证成功率 | 相对 API 成本 |
|:---|:---:|:---:|:---:|:---:|
| GPT-4 无弃权 | 77.8% | 100% | 13.9% | 1.000 |
| Cascade（含 GPT-4）| 80.2% | 77.6% | 90.5% | 0.215 |
| Cascade（无 GPT-4）| 80.3% | 68.3% | 90.8% | **0.126** |

即使完全不用 GPT-4（只用 Mistral-7B + Mixtral-8×7B + GPT-3.5），仍然保证了 80% 一致率，API 成本仅为 GPT-4 的 12.6%。

### 弃权策略的合理性

一个自然的担心是：弃权的样本会不会只是那些表面特征明显的"简单题"？论文收集了额外的人类标注来验证：

| 维度 | 弃权样本 | 被评测样本 |
|:---|:---:|:---:|
| 人类标注者间一致率 (IAA) | 0.815 | 0.902 |
| 长度比 | 0.242 | 0.245 |
| Token 重叠 (ROUGE-L) | 0.623 | 0.592 |

弃权的样本确实是人类标注者之间也不太一致的"主观题"（IAA 差异显著，$p < 10^{-8}$），而长度比和 token 重叠无显著差异。这说明弃权策略确实在捕捉任务的内在主观性，而非依赖浅层特征。

---

## 思考与讨论

**这篇论文的根本贡献不是一个更好的 judge 模型，而是一个更诚实的评测框架。** 它直面了一个常被忽略的事实：LLM judge 在某些样本上可靠，在另一些样本上不可靠，而且它自己对此的判断（vanilla confidence）往往不准。Simulated Annotators 用一种不需要额外监督的方式显著改善了这一点。

**Coverage 和保证之间的 trade-off 是实际落地时需要仔细考量的。** 在 TL;DR 上目标 90% 一致率时，coverage 约 56%——意味着接近一半的样本被弃权。对于排行榜评测（如 AlpacaEval）这可能可以接受，但对于需要全量评测的场景（如 RLHF 数据筛选）就需要进一步权衡。论文没有讨论如何处理弃权的样本——是直接丢弃，还是回退到人类标注？

**Simulated Annotators 的一个有趣推论是：模型的置信度校准可以和准确率解耦。** Mistral-7B 准确率低但校准好、GPT-4 准确率高但校准差——这意味着"知道自己不知道"可能是一种独立于"知道正确答案"的能力。这对 uncertainty estimation 领域是一个有启发的观察。

**从方法论角度看，fixed sequence testing + selective prediction 的组合并非本文首创（源自 Bates et al., 2021 和 Geifman & El-Yaniv, 2017），但将其应用于 LLM-as-judge 的 cascading 场景是一个恰当且实用的创新。** 这个框架的可扩展性值得关注：随着开源模型能力提升，cascade 中的模型可以随时替换，保证机制不需要改变。
