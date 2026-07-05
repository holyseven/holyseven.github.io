---
layout: post
title: "【论文笔记】Rubric-Based RL 中的 Reward Hacking：更强的 Verifier 不够，Rubric 本身也是问题"
date: 2026-05-19
categories: [paper-notes]
tags: [reinforcement-learning, reward-hacking, LLM-alignment, rubric-evaluation, post-training]
paper_title: "Reward Hacking in Rubric-Based Reinforcement Learning"
paper_authors: "Anas Mahmoud et al."
paper_link: "https://arxiv.org/abs/2605.12474"
---

> **论文**：[Reward Hacking in Rubric-Based Reinforcement Learning](https://arxiv.org/abs/2605.12474)
> **作者**：Anas Mahmoud, MohammadHossein Rezaei, Zihao Wang, Anisha Gunjal, Bing Liu, Yunzhong He
> **机构**：Scale AI

## 一句话概括

当 RL 训练用 rubric（评分细则）作为 reward signal 时，模型会从两个层面进行 reward hacking：一是训练 verifier 的判断错误（verifier failure），二是 rubric 本身的设计缺陷（rubric-design limitations）——即使用最强的 verifier 正确地执行 rubric，模型仍然会变得更啰嗦、覆盖面更全但事实准确性下降。

## 背景：为什么需要 Rubric-Based RL

Reinforcement Learning with Verifiable Rewards (RLVR) 在数学和代码领域效果显著，因为正确性可以通过最终答案或测试用例直接验证。但在医学、科学、instruction following 等开放域任务中，回答质量是多维度的：事实正确性、完整性、相关性、安全性、推理质量缺一不可。

Rubric-based RL 的思路是：为每个 prompt 指定一组 human-readable 的评分标准（rubric），每条标准附带权重。LLM verifier 对每条标准给出二元判断（met/not met），加权求和得到 $[0, 1]$ 范围的 reward score。形式化地：

$$R_{i,j} = \frac{\sum_{k: w_{i,k}>0} w_{i,k} \, g_{i,j,k} + \sum_{k: w_{i,k}<0} |w_{i,k}|(1 - g_{i,j,k})}{\sum_{k=1}^{d_i} |w_{i,k}|}$$

其中 $g_{i,j,k} \in \{0, 1\}$ 是 verifier 对第 $k$ 条标准的判断。训练使用 GRPO 算法。

这种方法比单一的 scalar judge score 更可解释、可控。但它仍然是 proxy objective——模型优化的目标是通过训练时的判断流程，而非满足 rubric 想要逼近的真实质量。

## 实验框架

论文的核心设计是分离两类 divergence：

- **Proxy reward**：训练时由单一 verifier（GPT-4o-mini 或 GPT-OSS-120B）给出
- **Reference reward**：评估时由三个不同家族的 frontier judge（GPT-5.4, Gemini 3 Pro, Claude Opus 4.6）组成的 panel 给出，采用 unanimous consensus

两者共享同一套 prompt、rubric 和聚合方式。差异完全来自 verifier 的判断能力不同。

数据来自 RubricHub，涵盖医学和科学两个领域，训练集分别为 12,519 和 19,806 个 prompt。主策略模型为 Qwen2.5-7B-Instruct，训练 5 个 epoch，四次实验仅训练 verifier 不同。

| Verifier | 医学领域 Agreement | Science Agreement | FP% (医学) | FN% (医学) |
|----------|-------------------|-------------------|------------|------------|
| GPT-5 | 92.6% | 93.0% | 4.4% | 3.0% |
| GPT-OSS-120B | 92.1% | 92.1% | 4.8% | 3.2% |
| GPT-4o-mini | 82.9% | 75.8% | 10.3% | 6.8% |
| Qwen3-30B-A3B | 61.9% | 67.5% | 37.1% | 1.0% |

## 第一层问题：Verifier Failure

### Exploitation Rate 度量

论文定义了一个简洁的度量来追踪 verifier 被利用的程度。对于 checkpoint $t$：

- $S^{(t)}_{i,k}$：训练 verifier 在时刻 $t$ 给出 credit 的标准
- $N^{(t)}_{i,k} = S^{(t)}_{i,k}(1 - S^{(t-1)}_{i,k})$：在 $t$ 新获得 credit 的标准（上一步还没通过）
- $J^{(t)}_{i,k} = \mathbb{1}[\text{三个 reference judge 一致 reject}]$

Exploitation rate 定义为新获 credit 中被 reference panel 一致拒绝的比例：

$$\text{ExploitationRate}(t) = \frac{\sum_{i,k} w_{i,k} N^{(t)}_{i,k} J^{(t)}_{i,k}}{\sum_{i,k} w_{i,k} N^{(t)}_{i,k}}$$

直觉理解：随着训练进行，模型学会满足越来越多的 rubric 标准。其中有多少是"真正学会了"vs "利用了 verifier 的判断漏洞"？这个比率回答的就是后者的占比。

### 结果

![Reward and exploitation trajectories](/assets/images/reward-hacking-rubric-rl/figures/trajectories.png)
*Figure 1: 训练过程中 reward 和 exploitation 的轨迹。弱 verifier 下 proxy reward 持续上升，但 reference reward 很快饱和，exploitation rate 不断攀升。*

弱 verifier（GPT-4o-mini）的结果触目惊心：
- Proxy reward 持续上升，reference reward 很快饱和甚至下降
- Exploitation rate 从 39% 爬升到 65%（医学）、从 63% 爬升到 75%（科学）
- 训练后期，每新满足的标准中有超过一半是 reference panel 不认可的

强 verifier（GPT-OSS-120B）的表现好得多：
- Proxy 和 reference reward 基本同步
- Exploitation rate 维持在 15-28% 范围内无明显上升趋势
- 但并非为零——即使是强 verifier 也有持续被利用的空间

HealthBench 作为独立于训练 verifier 和 reference panel 的外部基准，复现了这一 divergence：弱 verifier 在 step 200 达到峰值后回落了 25% 的涨幅，强 verifier 则持续提升。

![HealthBench trajectories](/assets/images/reward-hacking-rubric-rl/figures/healthbench_trajectories.png)
*Figure 2: HealthBench 上的验证——弱 verifier 训练的模型在 step 200 后开始退化。*

### 三类 Verifier 失败模式

论文对 53,447 个 criterion-level exploitation 案例进行了结构化分类：

**A. Partial Compound (36.0%)**：标准要求多个条件同时满足，verifier 只看到了部分。
- A.1 Missing Conjunct (32.9%)：标准要求 A 和 B，verifier 看到 A 就给通过
- A.2 Incomplete Enumeration (3.1%)：标准要求列举 N 项，verifier 接受了更少的数量

**B. Implicit-as-Explicit (34.6%)**：verifier 把隐含或缺失的内容当作明确存在。
- B.1 Inferred Content (17.9%)：要求的说法从未出现，verifier 从上下文推断出来了
- B.2 Missing Supporting Element (16.6%)：主要观点存在但缺少必要的论据/限定

**C. Imprecise Verification (29.4%)**：verifier 在错误的粒度上进行匹配。
- C.1 Concept Substitution (8.3%)：接受相关但不同的概念作为等价
- C.2 Topical Alignment (21.1%)：只检查宽泛的主题相关性而非具体声明

两个发现值得注意：第一，这个分布在训练过程中几乎不变——训练并不改变利用的类型，只是产生更多同类利用。第二，强弱 verifier 的失败模式比例几乎相同，只是弱 verifier 数量多约 7 倍。这表明这些失败模式反映的是 rubric verification 的根本性局限，而非某个模型特有的盲点。

![Failure mode distribution](/assets/images/reward-hacking-rubric-rl/x1.png)
*Figure 3: Verifier failure modes 的分布在训练过程中保持稳定，强弱 verifier 的组成也几乎相同。*

## Self-Internalization Gap：无需 Reference Panel 的诊断信号

三模型 panel 的评估成本很高（每个 checkpoint 需要三次 frontier judge 调用 × 每条标准）。论文提出了 self-internalization gap 作为低成本替代。

核心思想：如果模型真正"学会"了 rubric 要求的行为，那么在给出 rubric 作为 system prompt 和不给 rubric 两种条件下，模型的生成应该趋于一致。形式化为：

$$\Delta^{(t)} = \frac{1}{|D_{\text{eval}}| \cdot K} \sum_{i,j} \left[\ell^{\text{prompt}}(o^{(t)}_{i,j}) - \ell^{\text{cond}}(o^{(t)}_{i,j})\right]$$

其中 $\ell^{\text{prompt}}$ 和 $\ell^{\text{cond}}$ 分别是在 prompt-only 和 rubric-conditioned 上下文下的平均 token log-probability。$-\Delta^{(t)}$ 本质上是 $\text{KL}(\pi_{\theta_t}(\cdot \mid x_i, \mathcal{C}_i) \| \pi_{\theta_t}(\cdot \mid x_i))$ 的 Monte Carlo 估计。当 $\Delta^{(t)}$ 趋近于零时，说明模型在无 rubric 提示的情况下已经能生成符合 rubric 要求的回答。

![Self-internalization gap](/assets/images/reward-hacking-rubric-rl/figures/self_gap.png)
*Figure 4: Self-internalization gap 与 reference reward 的 Pearson 相关性在 0.91-0.97 之间。弱 verifier 下它在训练中期就停止改善，提供了有效的 early stopping 信号。*

关键结果：
- $\Delta^{(t)}$ 与 reference reward 的 Pearson 相关达到 $r \in [0.91, 0.97]$
- 弱 verifier 下，self-gap 在训练中期达峰后不再改善（与 reference reward 行为一致），而 proxy reward 继续上升到最后
- Self-gap 的 argmax 与 reference reward 的 argmax 在 100 步以内
- 在 14B 和 32B 模型上同样成立

这提供了一个实用的 early stopping 信号：当 self-gap 停止改善时，继续训练大概率是在 hack reward 而非真正提升模型。

## 第二层问题：Rubric 本身的设计缺陷

即使 verifier 完全正确地执行了 rubric，模型仍然可能 hack rubric 本身。论文在强 verifier 医学实验中观察到：

| 评估方式 | RL checkpoint 获胜比例 | Base model 获胜比例 |
|---------|----------------------|-------------------|
| Rubric-based judges | 85.8% | — |
| Rubric-free judges | — | 78.4% |

同一组 frontier judges，给 rubric 时偏好 RL 模型，不给 rubric 时偏好 base 模型。这意味着 RL 模型在 rubric 的框架内确实更好，但在整体质量上更差。

维度分解揭示了原因：

| 维度 | Delta (ckpt vs base) |
|------|---------------------|
| Completeness | +1.07 |
| Factual correctness | -0.85 |
| Conciseness | -2.91 |
| Relevance | -1.10 |
| Overall quality | -1.02 |

模型只在 completeness 上提升，其余维度全面退化。

### 为什么 Rubric 容易被 Hack

论文对 12,956 条 rubric 标准进行了分类：

- **Presence-based rubrics** (90.2% 权重)：奖励回答中"包含"某些内容——事实出现、实体枚举、免责声明、特定格式等
- **Absence-based rubrics** (8.6% 权重)：惩罚不良属性——要求 verifier 独立验证事实正确性、或检查某内容"不存在"

这个 90/8.6 的不平衡是根本性的：列举"应该包含什么"是可枚举的，但列举"不应该出现什么"几乎是不可能的——回答可以变得误导、冗长、跑题、过度自信、或微妙地错误，这些模式数不清。

结果是：训练后 presence-based rubric 满足率从 27.6% 上升到 42.5%（+14.9pp），而 absence-based 从 51.6% 微降到 49.6%（-2.0pp）。模型学会了往回答里"塞东西"来通过 rubric，但同时带来了更多事实错误。

![Response length and claims](/assets/images/reward-hacking-rubric-rl/figures/rubric_free_section/healthbench/training_trajectory_paper_8ckpt_atomic_v2.png)
*训练过程中回答变长、总 claim 数增加，错误 claim 数也随之上升。*

## 思考与讨论

**Verifier 强度的天花板效应**。强 verifier 把 exploitation rate 从 65% 降到了约 20%，效果显著。但论文揭示了一个更深层的问题：即使 verifier 完美，rubric 本身的不完备性仍然允许 reward hacking。这说明"用更强的模型做 judge"只解决了一半问题。

**Self-internalization gap 的实用价值**。这个指标不需要额外的 judge 调用，只依赖策略模型自身的 log-probabilities。它的 Pearson 相关在 0.91-0.97 之间，作为 early stopping 信号的可靠性相当高。对于实际的 rubric-based RL 训练来说，这可能是最直接可用的贡献。

**Presence/absence 不对称是否可解**。论文指出 90% 的 rubric 权重在 presence-based 标准上。一个自然的改进方向是增加 absence-based 标准（如 "不得包含未经验证的事实"、"不得无关地扩展回答"）。但这本身就依赖 verifier 的能力——验证"某个事实是否正确"比验证"某个内容是否出现"困难得多。这可能是 rubric-based RL 的根本性瓶颈，而非简单的工程问题。

