---
layout: post
title: "【论文笔记】用判别能力预测 Base Model 的 Post-training 潜力"
date: 2026-05-19
categories: [paper-notes]
tags: [LLM, evaluation, post-training, alignment, base-model]
paper_title: "On Predicting the Post-training Potential of Pre-trained LLMs"
paper_authors: "Xiaoyuan Li et al."
paper_link: "https://arxiv.org/abs/2605.11978"
---

> **论文**：[On Predicting the Post-training Potential of Pre-trained LLMs](https://arxiv.org/abs/2605.11978)
> **作者**：Xiaoyuan Li, Yubo Ma, Kexin Yang, Moxin Li, Keqin Bao, Wenjie Wang, Fuli Feng, Dayiheng Liu
> **机构**：University of Science and Technology of China, Alibaba Group, National University of Singapore

## 一句话总结

在做 post-training 之前，如何判断哪个 base model 更有潜力？这篇论文提出用 base model 的判别能力（区分好回复和差回复）来预测它经过 SFT/RL 后的生成表现，并构建了 RuDE 评测框架来度量这种判别能力，实验表明判别得分与 post-training 后表现的 Pearson 相关系数超过 0.9。

## 问题动机

LLM 开发的标准流程是 pre-training + post-training。Post-training 的效果很大程度上取决于 base model 本身的能力，但目前缺乏可靠手段在 post-training 之前就预测哪个 base model 表现更好。

传统做法是用 MMLU 等多选知识 benchmark 来评估 base model 质量。但论文发现这些指标与模型 post-training 后在开放式任务上的表现**相关性很弱**。原因在于，MMLU 测的是知识储备，而 post-training 任务需要的是指令遵循、安全合规、格式控制等综合能力。

![Figure 1: 传统 benchmark 与 post-training 表现的相关性分析](/assets/images/predict-post-training-potential/x1.png)
*Figure 1: 横轴是 base model 在 MMLU 等知识 benchmark 上的准确率，纵轴是对应模型 post-training 后在 HealthBench 上的表现。Pearson 相关系数很低，说明传统 benchmark 无法预测 post-training 潜力。*

## 核心假设：GD-Potential Hypothesis

GD 是 Generative-Discriminative 的缩写——用 Discriminative（判别）能力预测 Generative（生成）潜力。论文的关键洞察来自一个简单的逻辑链条：

1. Base model 不能直接生成好的开放式回复（generation gap），所以不能直接评估其生成能力
2. 但可以评估它的**判别能力**——能否区分好回复和差回复
3. 如果一个 base model 已经能从概率上偏好好的回复，那说明它的内部表示已经接近人类偏好，post-training 时需要的分布偏移更小

形式化表述：给定 query $x$，正例回复 $y^+$ 和负例回复 $y^-$，base model $\theta$ 的判别得分定义为：

$$S_{\text{dis}}(\theta, T) = \mathbb{E}_{(x, y^+, y^-)} \left[ \mathbb{I}\left(\ell_\theta(y^+ \mid x) > \ell_\theta(y^- \mid x)\right) \right]$$

其中 $\ell_\theta(y \mid x)$ 是模型给 response $y$ 的条件 log-likelihood。通俗地说，就是看模型给好回复打的概率是否高于差回复。

GD-Potential Hypothesis 的核心主张是：这个判别得分与 post-training 后的生成表现强正相关。从 DPO 的视角理解：base model 就是 reference policy，如果它本身已经倾向于 preferred response，那 post-training 只需要更小的 distributional shift，自然表现更好。

## RuDE 框架：如何构造高质量的对比样本

框架名称 RuDE 代表 Rubric-based Discriminative Evaluation。其核心挑战是构造 $(y^+, y^-)$ 对——两者必须只在特定维度上有差异，不能引入长度、文风等混淆因素。

![RuDE 构造流程](/assets/images/predict-post-training-potential/x2.png)
*Figure 2: RuDE 的三阶段构造流程。Phase 1 通过迭代精炼生成满足所有 rubric 的正例，Phase 2 通过 Controlled Degradation 生成精确违反特定 rubric 的负例，Phase 3 随机排列形成判别任务。*

构造流程分三个阶段：

**Phase 1: 正例生成** — 用强生成器（Gemini-3-Pro）生成回复，用验证器（GPT-4.1）逐条检查 rubric 是否满足。不满足则生成反馈，让生成器迭代修正，直到所有 rubric 都通过。

**Phase 2: 负例生成（Controlled Degradation）** — 这是方法的核心。选定一组 target violation rubrics $\mathcal{V}_{target}$，要求生成器产出一个**精确违反这些 rubric、同时保持其余所有 rubric 满足**的回复。关键约束包括：
- 长度约束：负例的 token 数必须接近正例，避免模型通过长度差异"抄近路"
- 双重验证：验证器同时检查 (1) 目标 rubric 是否确实被违反，(2) 保留的 rubric 是否仍然满足
- 迭代循环：不满足任一条件则继续修正

**Phase 3: 对比任务** — 将 $y^+$ 和 $y^-$ 配对，随机分配到 A/B 位置以消除位置偏差。评估时给 base model 2-shot（一个选 A，一个选 B），看它能否基于条件概率正确选出更好的回复。

### 为什么必须用 Controlled Degradation？

论文通过两组 ablation 说明了这一点：

- **Ablation A (Natural Sampling)**：从强模型采样 32 个回复，取最好和最差做对比。结果即使是 GPT-5 自己也只能达到 27.1% 准确率——因为自然生成的回复差异来自分布噪声，rubric 违反极其隐蔽且伴随大量混淆因素，任务太难。
- **Ablation B (Locate-and-Rewrite)**：直接定位正例中与目标 rubric 相关的文本段，重写引入违反。结果 Qwen3-30B-A3B-Base 和 Qwen3-235B-A22B-Base 分别达到 79.2% 和 80.6%，差距仅 1.4%——因为局部重写会留下明显的"拼接痕迹"（语气突变、指代断裂），模型不需要理解 rubric 就能识别，任务太简单。

RuDE 的 Controlled Degradation 处于两个极端之间：负例维持全局连贯性，没有拼接痕迹，但确实精确违反了目标约束。

## 4C 分类体系

为了提供细粒度诊断，论文将所有 rubric 归入 4C Taxonomy：

| 维度 | 子类别 | 描述 |
|------|--------|------|
| Competence | Factuality, Logic, Procedure | 模型的核心智力和可靠性：事实准确性、推理正确性、程序遵循 |
| Content | Completeness, Coherence, Relevance | 语言质量：完整性、连贯性、相关性 |
| Control | Format, Length, Scope | 对刚性约束的遵循：格式、长度、关键词包含/排除 |
| Compliance | Safety, Persona, Utility | 与人类意图的对齐：安全边界、角色扮演、实用性 |

## 实验验证

### 数据集构成

评测覆盖四个领域，共 28,683 个样本：

| 领域 | 数据源 | 样本数 | 平均 Token |
|------|--------|--------|-----------|
| 医疗咨询 | HealthBench | 3,673 | 3,787 |
| 法律金融 | PRBench | 12,433 | 14,231 |
| 复杂指令 | AdvancedIF | 4,838 | 6,369 |
| 创意写作 | WritingBench | 7,739 | 18,191 |

难度通过违反 rubric 的数量 $\lvert\mathcal{V}\rvert$ 来分层：$\lvert\mathcal{V}\rvert=1$ 最难（仅一个微小差异），$\lvert\mathcal{V}\rvert=5$ 相对较易。人工验证显示 pipeline 标签与人类判断有 92% 一致性。

### GD-Potential 假设验证

![相关性分析](/assets/images/predict-post-training-potential/x3.png)
*Figure 3: Base model 在 RuDE 上的判别得分（X 轴）与其 instruct-tuned 版本在标准开放式 benchmark 上的表现（Y 轴）。Pearson 相关系数超过 0.90。*

在 AdvancedIF 上，RuDE 得分与 post-training 表现的 Pearson 相关系数达到 0.91（$p < 0.001$）。DeepSeek-V3.1 和 GLM-4.5 等模型持续占据散点图的右上角。这远超 MMLU 等传统指标的预测力。

### 主要性能排名

| Model | AdvancedIF | HealthBench | WritingBench | PRBench | AVG |
|-------|-----------|-------------|-------------|---------|-----|
| DeepSeek-V3.1 | 93.1 | 84.3 | 73.1 | 64.6 | 78.8 |
| GLM-4.5 | 85.7 | 78.1 | 65.0 | 72.0 | 75.2 |
| Kimi-K2 | 83.5 | 77.7 | 80.1 | 57.6 | 74.7 |
| Seed-OSS-36B | 86.6 | 76.8 | 63.0 | 58.4 | 71.2 |
| Qwen3-235B-A22B | 85.8 | 72.7 | 68.5 | 56.5 | 70.9 |
| Qwen2.5-72B | 80.5 | 67.1 | 57.4 | 51.8 | 64.2 |
| Qwen3-14B | 77.7 | 62.9 | 62.9 | 51.4 | 63.7 |
| Qwen2.5-7B | 43.4 | 40.7 | 42.1 | 39.5 | 41.4 |

几个值得注意的发现：
- DeepSeek-V3.1 总分最高（78.8%），显示出最强的 post-training 潜力
- 同系列模型内部遵循 scaling law：Gemma3 系列从 4B 到 27B 单调递增
- 领域特化现象明显：Kimi-K2 在创意写作上超过 DeepSeek-V3.1，GLM-4.5 在法律金融领域表现突出
- 代际差异大于规模差异：Qwen3-4B 全面超越 Qwen2.5-7B（53.5 vs 41.4），尽管参数量更小

### 难度梯度验证

![Violation Number 影响](/assets/images/predict-post-training-potential/x4.png)
*Figure 4: 随着违反 rubric 数量从 Min（1）到 Max（3 或 5），判别任务难度降低，所有模型准确率上升。*

当 $\lvert\mathcal{V}\rvert$ 从 5 降到 1，DeepSeek-V3.1 在 AdvancedIF 上的准确率从 93.1% 降至 70.1%。这验证了难度分层机制的有效性——RuDE 不会饱和，能在不同粒度探测模型能力。

### RL 训练验证

![RL 训练曲线](/assets/images/predict-post-training-potential/x5.png)
*Figure 5: 三个 base model 在 HealthBench 上经过 RL 训练后的表现轨迹。*

为验证 RuDE 的预测效力在实际 post-training 中是否成立，论文对 Qwen2.5-7B、Qwen2.5-14B 和 Qwen3-4B 进行了 RL 训练。结果：
- 排名一致性：Qwen2.5-14B 最终得分最高，Qwen3-4B 超过了更大的 Qwen2.5-7B，与 RuDE 预测一致
- 学习效率：Qwen3-4B 尽管参数少于 Qwen2.5-7B，但展现出更陡的学习曲线，更快利用 rubric reward 信号

这是该方法最有实际价值的验证——RuDE 确实能在 post-training 之前识别出"小而强"的模型。

## 4C 细粒度诊断

4C Taxonomy 的另一个价值在于提供能力画像而非单一分数：

- 全局趋势：Compliance 是多数模型最强的维度（均值 52.0%），Content 最弱（49.3%）
- 小模型短板：sub-7B 模型在 Control 维度上一致最差，表明小模型对格式、长度等刚性约束的理解最为薄弱
- DeepSeek-V3.1 不仅总分最高（61.4%），而且四个维度间的标准差最小（$\sigma_{4C}=1.5$），说明它是"全面选手"
- Qwen3-235B-A22B 在 PRBench 上 Content（74.6%）与 Competence（46.5%）差距达 28.1，表现为"流畅但不准确"——专业领域的事实性是其短板

## 讨论

**判别-生成的对应关系有边界条件。** 论文承认 GD-Potential 假设不是万能的——存在"理论家"模型：判别能力强但生成执行力弱，尤其在高度创造性任务中。因此判别得分应被视为 post-training 潜力的必要条件而非充分条件。

**框架受限于 teacher model 的能力上限。** 正例生成依赖 Gemini-3-Pro，负例构造需要 GPT-4.1 验证。当被测 base model 的能力接近或超越这些 teacher 时，构造的对比样本可能不再具有区分力。

**从"模型选择"到"训练预算分配"的延伸。** RuDE 的直接应用是在多个候选 base model 中选出最有 post-training 潜力的那个。但 4C 诊断还能告诉我们：对于特定模型，应该把 post-training 资源集中在哪个维度上——这是比单纯排名更有意义的信息。
