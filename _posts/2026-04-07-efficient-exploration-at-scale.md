---
layout: post
title: "【论文笔记】用高效探索将 RLHF 数据效率提升 1000 倍"
date: 2026-04-07
categories: [paper-notes]
tags: [RLHF, exploration, reinforcement-learning, LLM-alignment, data-efficiency]
paper_title: "Efficient Exploration at Scale"
paper_authors: "Mohammad Asghari et al."
paper_link: "https://arxiv.org/abs/2603.17378"
---

> **论文**：[Efficient Exploration at Scale](https://arxiv.org/abs/2603.17378)
> **作者**：Mohammad Asghari, Chris Chute, Vikranth Dwaracherla, Xiuyuan Lu, Mehdi Jafarnia, Victor Minden, Zheng Wen, Benjamin Van Roy
> **机构**：Google DeepMind (The Efficient Agent Team)

当前 RLHF 的一个尴尬现实是：即使大幅增加人类标注数据量，模型性能的提升也极其有限。这篇论文从一个不同的角度出发——与其用更多数据做离线训练，不如让模型在交互中主动选择最有信息量的数据去学习。他们在 Gemma 9B 上的实验表明，这种 online + 探索的方式只需不到 20K 标注就能匹配离线 RLHF 200K 标注的效果，预期在 1M 标注量级可实现约 1000 倍的数据效率提升。

## 从离线到在线：为什么数据效率差这么多

标准的离线 RLHF 流程是：先用固定策略生成大量 response pair，收集人类偏好标注，训练 reward model，再优化 policy。这个流程的核心问题在于数据采集策略是固定的——用初始 policy 生成的 response pair 未必能提供多少有用信息。随着 policy 改进，早期收集的数据越来越不相关。

论文沿着从离线到在线的谱系，对比了四种算法：

1. **Offline RLHF**：用初始策略 $\pi_{\theta_0}$ 生成所有数据，训完 RM 再训 policy，标准流程
2. **Periodic RLHF**：每隔 $\tau$ 个 batch（实验中 $\tau=400$）重新训练 RM 和 policy，用更新后的策略采集新数据
3. **Online RLHF**：每个 batch 后都增量更新 RM 和 policy，实时调整采样策略
4. **Information-directed exploration**：在 Online RLHF 基础上，加入不确定性建模，主动选择信息量最大的 response pair 去询问偏好

## 三个关键技术组件

### Affirmative Nudge：防止在线训练崩溃

在线 RLHF 有个已知的问题：训到一定阶段后性能会急剧下降（tanking）。已有的解决方案要么是做 checkpoint 回退，要么是降低学习率来延迟崩溃，但都会牺牲最终性能。

论文的做法简单得有些出人意料：在 policy 梯度的强化信号中加一个小正数 $\epsilon$（即 affirmative nudge）。原始的 policy 更新中，强化信号为 $p_{\bar{\phi}_t}(Y \succeq Y'|X) - \frac{1}{2}$，加入 nudge 后变为：

$$
p_{\bar{\phi}_t}(Y \succeq Y'|X) - \frac{1}{2} + \epsilon
$$

直觉上，这个正偏移意味着对模型自己生成的 response 给予一定程度的"肯定"，相当于对 policy 施加了一个轻微的自我强化倾向，避免 policy 在 RM 不准确的区域被过度惩罚而走偏。

![Figure 4](/assets/images/efficient-exploration-at-scale/x3.png)
*Figure 4: 左图：不使用 reward model 的在线方法（reward-model-free）性能不如使用 RM 的版本。右图：没有 affirmative nudge 时在线 RLHF 出现 tanking，降低学习率或 checkpoint 回退都不如直接加 nudge 的效果好。*

### Epistemic Neural Network：建模 Reward 不确定性

为了做信息导向的探索，需要知道 RM 对哪些 response pair 的偏好判断不确定。论文采用 epistemic neural network（ENN）来建模这种不确定性。

ENN 的结构基于 randomized prior functions 的思想。在共享的 transformer backbone 之上，除了一个标准的 MLP head（$\mathtt{mlp0}$，作为 point estimate），还并联了 100 组 prior network + differential network。每组的 prior network（宽度 256，两层 MLP，参数随机初始化后固定不训）和 differential network（宽度 1024，两层 MLP，可训练）的输出相加，构成一个 ensemble particle。

![Figure 5](/assets/images/efficient-exploration-at-scale/enn.png)
*Figure 5: 普通 RM 只输出一个 reward 值；ENN RM 额外接收一个 epistemic index $Z$，不同的 $Z$ 值对应不同的 ensemble particle，输出不同的 reward 预测。*

当 epistemic index $Z=0$ 时使用 point estimate head；$Z=1,...,100$ 时使用对应的 ensemble particle。训练时 backbone 对 ensemble 部分冻结，只更新各自的 differential network。这样，100 个 particle 之间的预测差异就反映了 RM 的认知不确定性（epistemic uncertainty）。

增加的参数量相对于 9B backbone 不到 5%。

### Information-Directed Sampling：选择最有信息量的 Query

有了不确定性估计，就可以做信息导向的探索。对每个 prompt，算法生成 16 个候选 response，然后对所有 response pair 计算选择概率在 ensemble 上的方差：

$$
\text{Var}_{Z}[p_{\phi_t}(Y \succeq Y'|X, Z)]
$$

方差最大的 pair 被选为 query 发送给人类标注者。直觉是：如果 ensemble 内部对"谁更好"分歧很大，那么这个 pair 的人类偏好标注将提供最多的信息来消除 RM 的不确定性。

## 实验设计

实验基于 Gemma 9B 模型，使用 Gemini 1.5 Pro 作为人类反馈模拟器（通过 Bradley-Terry 模型产生偏好概率）。用一个比被训练模型大得多的模型来模拟人类偏好，是为了让模拟的选择行为比被训练的 policy 更复杂，从而更接近真实人类。

训练 prompt 为 200K 个，覆盖写作、编码、摘要、数学、科学等多种类型。评估通过 1K 个 out-of-sample prompt 计算 win rate：让训练得到的 policy 和 baseline（SFT 后未做 RLHF 的 top-1 策略）各生成一个 response，由模拟器判断偏好概率，取平均作为 win rate。

采样策略上，baseline 使用 top-1（确定性），实验中生成候选 response 使用 top-5 以增加多样性。每 64 个 prompt 为一个 batch。

## 结果

![Figure 8](/assets/images/efficient-exploration-at-scale/x4.png)
*Figure 8: 四种算法的 win rate 随标注量变化曲线。横轴为 log scale。Information-directed exploration 在约 20K 标注时达到的 win rate，offline RLHF 需要超过 200K 标注才能匹配。*

核心结果如下表（从 Figure 8 读取的近似值）：

| 算法 | ~20K 标注 win rate | ~200K 标注 win rate |
|------|-------------------|---------------------|
| Offline RLHF | ~0.55 | ~0.66 |
| Periodic RLHF | ~0.60 | ~0.70 |
| Online RLHF | ~0.65 | - |
| Info-directed exploration | ~0.68 | - |

从 offline 到 periodic 到 online，每一步都有显著提升。Information-directed exploration 在 online RLHF 基础上进一步拉开差距，但主要的增益来自 online 这一步。

![Figure 9a](/assets/images/efficient-exploration-at-scale/x5.png)
*Figure 9(a): Win rate 外推曲线。拟合函数为 $w(n) = 1 - 0.5 \cdot (n/a)^{-b}$。两条曲线在 log scale 上近似线性但斜率不同。*

![Figure 9b](/assets/images/efficient-exploration-at-scale/x6.png)
*Figure 9(b): 相对于 offline RLHF 的数据效率增益预测。在 1M 标注处预计达到约 1000 倍。*

论文拟合了 $w(n) = 1 - 0.5 \cdot (n/a)^{-b}$ 形式的 scaling law，将两种方法的曲线外推到更大的数据量。结果显示 information-directed exploration 的 scaling 斜率更陡，因此效率增益随数据量增长而放大。

### 定性案例

论文给出了一个数学题的例子：offline RLHF 产生的 response 逻辑混乱且答案错误，而 information-directed exploration 产生的 response 简洁正确。此外，展示了 infomax 和 infomin response pair 的对比——infomin pair 的两个 response 内容几乎相同（标注它们没什么信息量），而 infomax pair 的两个 response 在内容或形式上有实质差异，标注者的选择能真正帮助 RM 学到区分能力。

## 思考与讨论

一个自然的问题是 1000 倍的效率增益预测有多可靠。这个数字来自对 scaling law 的外推，而实际实验范围只到约 200K 标注。Scaling law 的函数形式 $w(n) = 1 - 0.5 \cdot (n/a)^{-b}$ 是否在更大数据量级上仍然成立，目前没有验证。特别是 information-directed exploration 的曲线可能存在尚未暴露的瓶颈——比如当 RM 的 epistemic uncertainty 整体变低后，探索的边际收益可能迅速递减。

另一个值得关注的点是人类反馈模拟器的使用。用 Gemini 1.5 Pro 模拟人类偏好虽然在实验规模上有明显优势，但 simulated preference 和真实人类偏好之间的差距可能影响结论的迁移性。真实人类标注的噪声模式、不一致性和偏见可能与 Bradley-Terry 模型假设有所不同。

从工程落地的角度看，information-directed exploration 要求对每个 prompt 生成 16 个候选 response，再对所有 pair 做 100 次 ensemble forward pass 计算方差——计算开销相当大。如果将数据采集成本（人工标注费用）的节省与推理计算成本的增加放在一起权衡，实际的性价比需要具体场景具体分析。

论文目前只在单一模型（Gemma 9B）和单一评估设置上验证。方法是否对不同规模的模型（更大或更小）同样有效，以及在不同类型的任务分布上是否稳健，都是后续工作可以回答的问题。
