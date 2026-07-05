---
layout: post
title: "【论文笔记】Auto-Rubric as Reward：用显式评分标准替代隐式偏好建模"
date: 2026-05-19
categories: [paper-notes]
tags: [RLHF, reward-model, multimodal-generation, image-generation, preference-alignment]
paper_title: "Auto-Rubric as Reward: From Implicit Preferences to Explicit Multimodal Generative Criteria"
paper_authors: "Juanxi Tian et al."
paper_link: "https://arxiv.org/abs/2605.08354"
---

> **论文**：[Auto-Rubric as Reward: From Implicit Preferences to Explicit Multimodal Generative Criteria](https://arxiv.org/abs/2605.08354)
> **作者**：Juanxi Tian, Fengyuan Liu, Jiaming Han, Yilei Jiang, Yongliang Wu, Yesheng Liu, Haodong Li, Furong Xu, Wanhua Li
> **机构**：Nanyang Technological University, Ant Group, MMLab (CUHK), UIUC

## 一句话概括

当前多模态 RLHF 的瓶颈不是偏好知识不够，而是缺少一个结构化的接口来表达和使用它。ARR 通过让 VLM 将隐式偏好知识外化为 prompt 级别的 rubric（评价维度），把不透明的标量奖励替换为可验证的多维度判断，从而同时提升评估可靠性和生成质量。

## 问题动机

多模态生成模型（文生图、图像编辑）的对齐主要依赖 RLHF。现有方案存在几个结构性问题：

1. 标量奖励模型（PickScore、ImageReward、HPSv3）将人类多维度判断压缩成一个数字，信息丢失严重，且容易被 reward hacking 利用
2. VLM-as-Judge 的 pairwise 比较虽然更平衡，但存在严重的 positional bias —— 同一对图片交换顺序后准确率差距可达 30+ 个百分点
3. 已有 Rubric-as-Reward 方法需要手工或监督式构造评分标准，难以泛化到多模态场景

论文的核心观察：VLM 本身已经拥有足够的偏好知识，问题在于没有提供一个合适的"接口"来稳定地调用这些知识。

## ARR：将隐式知识外化为显式 Rubric

![ARR-RPO 框架概览](/assets/images/auto-rubric-reward/x1.png)
*Figure 1: ARR-RPO 框架整体流程*

### 形式化

给定 prompt $x$ 和一对输出 $(y^+, y^-)$，传统隐式偏好建模通过 Bradley-Terry 模型估计标量奖励 $r_\phi$。ARR 则将偏好建模转化为寻找最优 rubric：

$$R^* = \arg\max_{R \subset \mathcal{S}} \sum_{i=1}^N \log P^*(y_i^+ \succ y_i^- | x_i, R)$$

由于 rubric 空间 $\mathcal{S}$ 是离散且巨大的，实际中简化为从有限候选集 $\mathcal{D}_R$ 中选出使判断正确率最高的子集。

### 三阶段 Pipeline：生成-验证-结构化

**阶段一：Per-Instance Rubric 生成。** 对每个偏好对 $(x_i, y_i^+, y_i^-)$，VLM 生成一份自然语言 rubric $r_i$，解释为什么 $y_i^+$ 优于 $y_i^-$。具体要求包括：将偏好分解为独立的可验证质量轴（语义保真度、属性准确性、空间一致性等），每个轴表述为二元判据。

**阶段二：验证与精炼。** 独立的 verifier 检查 rubric 能否复现原始偏好判断。如果验证失败，进入迭代精炼（最多 $T_{\max}=5$ 次）。实验中约 87% 的 rubric 一次通过验证，最终仅约 4% 被丢弃。

$$v_i = \mathcal{M}_{\text{verify}}(x_i, y_i^+, y_i^-, r_i)$$

**阶段三：层次化结构组织。** 所有通过验证的 rubric 被 LLM 整合为一个层次化评价协议 $R_{\text{structured}}$，按语义粒度（整体对齐、构图属性、局部细节）组织，作为后续评估和训练的标准化条件。

### 从 Rubric 到 Reward

VLM judge 基于 rubric 给出二元偏好决策后，reward 定义为简单的常数：

$$r(x, y; y') = \begin{cases} +\lambda & \text{if } \mathcal{M}_\theta \text{ prefers } y \\ -\gamma & \text{otherwise} \end{cases}$$

其中 $\lambda, \gamma > 0$ 是正负奖励幅度。这种设计刻意避免连续标量奖励带来的 reward hacking 风险。

## RPO：Rubric Policy Optimization

RPO 将 ARR 的评估能力直接转化为训练信号。它是一个在线策略优化算法，核心流程：

1. 从当前策略 $\pi_\theta$ 对同一 prompt 采样两个候选输出
2. 冻结的 ARR judge 基于 rubric 做 pairwise 比较
3. 胜者获得 advantage $+\lambda$，败者获得 $-\gamma$
4. 将 advantage 均匀分配到所有去噪时间步
5. 使用 PPO-style clipped objective + KL 正则化更新策略

RPO 目标函数：

$$\mathcal{L}_{\text{RPO}}(\theta) = \mathbb{E}\left[\frac{1}{2}\sum_{i=1}^{2}\left(\frac{1}{T}\sum_{t=0}^{T-1}\min(r_t^i(\theta)A_i, \text{clip}(r_t^i(\theta), 1-\epsilon, 1+\epsilon)A_i) - \beta D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})\right)\right]$$

与传统 RLHF 相比，RPO 不需要训练单独的标量奖励模型；奖励来自冻结的 VLM judge，不存在 reward model drift 问题。

## 实验结果

### 偏好评估

在 HPDv3、MM-RewardBench2、EditReward-Bench 三个 benchmark 上评估 ARR 作为 preference evaluator 的表现：

| Method | HPDv3 | MM-RB2 (T2I) | MM-RB2 (Edit) | EditReward-Bench |
|--------|-------|---------------|---------------|------------------|
| HPSv3 | 76.9 | 60.2 | — | — |
| Qwen3-VL-8B | 67.2 | 57.6 | 59.2 | 54.01 |
| + ARR | 70.2 (+3.0) | 62.7 (+5.1) | 65.5 (+6.3) | 57.22 (+3.21) |
| GPT-5 | 72.4 | 70.5 | 73.8 | 57.53 |
| + ARR | 76.1 (+3.7) | 74.7 (+4.2) | 77.5 (+3.7) | 61.01 (+3.48) |
| Gemini 3.1 Pro | 76.6 | 75.1 | 77.4 | 61.23 |
| + ARR | 78.3 (+1.7) | 78.9 (+3.8) | 79.2 (+1.8) | 63.27 (+2.04) |

ARR conditioning 在所有 VLM judge 上都带来 1.7-6.3 个百分点的提升，且 VLM 越弱提升越大。

### 生成质量

以 FLUX.1-dev 为基座模型进行 T2I 的 RPO 训练：

| Method | GenEval | DPG-Bench | TIIF | UniGenBench++ (Short) |
|--------|---------|-----------|------|----------------------|
| FLUX.1-Dev (baseline) | 0.66 | 83.84 | 71.09 | 60.97 |
| + RPO-Gemini 3.1 Pro-ARR | 0.80 (+0.14) | 85.76 (+1.92) | 76.85 (+5.76) | 65.89 (+4.92) |
| BAGEL (当前 SOTA 之一) | 0.82 | 85.07 | 71.50 | 59.91 |

RPO 将 FLUX.1-dev 从中等水平提升到接近甚至超过 BAGEL 等专用模型的水平。值得注意的是，在 TIIF（指令遵循）上提升最为显著（+5.76），说明 rubric 式奖励对结构化约束的传递特别有效。

### 位置偏差消除

![消融实验](/assets/images/auto-rubric-reward/x4.png)
*Figure 4(a): 不同评估器的 Forward-Reverse 偏好差距*

位置偏差 $\Delta = \text{Acc}_{\text{fwd}} - \text{Acc}_{\text{rev}}$ 的变化：

| Model | Baseline $\Delta$ | + ARR (zero-shot) | + ARR w/ guide |
|-------|-------------------|-------------------|----------------|
| Qwen3-VL-8B | 34.6 | 31.6 | 10.3 |
| GPT-5 | 32.6 | 28.2 | 9.3 |
| Gemini 3.1 Pro | 30.2 | 27.8 | 8.9 |

基线 VLM 的位置偏差惊人地大（30+），且不随模型能力增强而缓解。ARR 的 zero-shot 版本带来温和改善（3-5 点），而加入少量人类偏好样例（guide）后，$\Delta$ 锐减到 9 左右。这表明位置偏差是隐式偏好表示的结构性缺陷，而非能力不足。

### 跨模型 Rubric 迁移

![跨模型迁移](/assets/images/auto-rubric-reward/x5.png)
*Figure 4(b): 固定 judge 为 Gemini 3.1 Pro，变换 rubric 生成器的效果*

即使用最弱的 Qwen3-VL-8B 生成 rubric，再由 Gemini 3.1 Pro 作为 judge 使用，准确率仍从 75.9% 提升到 77.5%，恢复了 same-family 设置（79.2%）一半以上的差距。这说明 rubric 的结构本身（而非模型间的共享偏差）是性能提升的主要贡献者。

## 讨论

**Rubric 数量的效用递增问题。** 消融实验显示，将每个样本的 rubric 维度从 $K=1$ 增加到 $K=20$，准确率从 69.8% 单调上升到 74.4%。但论文默认使用 $K=5$ 作为效率折中，因为每增加一个 rubric 就需要额外的生成和验证调用。这里存在一个 inference cost vs. quality 的权衡，在生产环境中如何动态决定 rubric 数量值得进一步研究。

**冻结 judge 的局限性。** 论文刻意使用冻结的 VLM 来隔离 rubric 接口本身的贡献，但承认 fine-tune judge 很可能进一步提升 rubric 质量。当前的结果可以视为性能下界。一个自然的后续方向是将 rubric 生成作为 reward model 的训练信号——相当于在 rubric 空间而非标量空间做 reward modeling。

**二元奖励的表达能力。** RPO 使用 $\{+\lambda, -\gamma\}$ 的二元 reward 而非连续值，这在原理上会丢失偏好强度信息。论文的立场是，二元信号结合 rubric conditioning 已经足够稳定地传递梯度方向，而连续标量奖励反而更容易被 exploit。不过在两个候选质量接近时，二元信号可能产生较高方差的梯度估计。
