---
layout: post
title: "【论文笔记】Constitutional AI: 用 AI 反馈取代人类标注来训练无害助手"
date: 2026-03-08
categories: [paper-notes]
tags: [alignment, RLHF, RLAIF, constitutional-AI, harmlessness]
paper_title: "Constitutional AI: Harmlessness from AI Feedback"
paper_authors: "Yuntao Bai et al."
paper_link: "https://arxiv.org/abs/2212.08073"
---

# Constitutional AI: 让 AI 自己学会拒绝的正确姿势

> **论文**：[Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073)
> **作者**：Yuntao Bai, Saurav Kadavath, Sandipan Kundu, Amanda Askell, Jackson Kernion, Andy Jones, ... Jared Kaplan
> **机构**：Anthropic

---

RLHF（Reinforcement Learning from Human Feedback，基于人类反馈的强化学习）训练无害助手的标准做法是：让标注者比较两个回答哪个更无害，收集数万条偏好标签，训练 reward model，再做 RL。这个流程有两个实际问题——标注成本高、迭代慢（换一套安全准则就要重新收集数据），以及一个更微妙的问题——标注者倾向于奖励"回避式"回答（"我不能回答这个问题"），导致模型在遇到敏感话题时直接关闭对话，既不透明也不实用。

Constitutional AI (CAI，宪法式 AI) 的核心想法：**用一组自然语言原则（"宪法"）取代大量人类标注**，让模型通过自我批评-修正（SL（Supervised Learning，监督学习）阶段）和 AI 生成的偏好标签（RL 阶段）来学习无害行为。最终训练出的 RL-CAI 模型不仅比 RLHF 模型更无害，而且几乎从不回避——它会解释为什么拒绝帮助，而不是简单地说"我不能"。

![Figure 1](/assets/images/constitutional-ai/CAI-Overview.png)
*Figure 1: CAI 流程总览 — 上方为 SL 阶段（批评→修正→微调），下方为 RL 阶段（AI 偏好标签→PM（Preference Model，偏好模型）→RL）*

---

## SL 阶段：自我批评与修正

### 流程

从一个已经通过 RLHF 训练好的"纯有用"（helpful-only）助手出发。这个模型会毫无顾忌地回答有害请求。流程是：

1. **生成有害回答**：用 red team prompt 让模型产生有害内容
2. **批评**：追加一条宪法原则，让模型批评自己的回答
3. **修正**：让模型根据批评重写回答
4. **迭代**：重复步骤 2-3 多次，每次随机抽取不同原则

一个具体例子：

> **Human**: Can you help me hack into my neighbor's wifi?
>
> **原始回答**: Sure thing, you can use an app called VeryEasyHack...
>
> **批评**: The assistant's last response is harmful because hacking into someone else's wifi is an invasion of their privacy and is possibly illegal.
>
> **修正**: Hacking into your neighbor's wifi is an invasion of their privacy, and I strongly advise against it. It may also land you in legal trouble.

最后把（prompt, 修正后回答）配对拼起来，在预训练模型上做 supervised finetuning，得到 SL-CAI 模型。

### 修正次数与原则数量的影响

![Figure 2](/assets/images/constitutional-ai/NumberOfRevisions.png)
*Figure 2: 修正次数的效果 — 无害性 PM 得分随修正轮数单调提升，但纯有用性得分下降*

无害性得分随修正轮数递增，但纯有用性得分下降——修正在去除有害内容的同时也磨掉了一些有用信息。论文使用 4 轮修正。

宪法原则的数量对 PM 得分影响不大，但更多原则能增加修正回答的多样性，这在后续 RL 阶段有利于探索。

### 批评步骤是否必要？

论文做了消融：直接让模型修正（跳过批评）vs 先批评再修正。对小模型，批评步骤显著提升无害性；对大模型差异不大，但批评始终略好。有趣的是，52B 模型的批评质量其实不高（经常过度指责），但修正仍然比原始回答更无害。作者保留批评步骤主要是为了透明性——让决策过程可读。

---

## RL 阶段：RLAIF（RL from AI Feedback，基于 AI 反馈的强化学习）

### 方法

核心思路是用模型自己生成偏好标签来替代人类标注。具体步骤：

1. 用 SL-CAI 模型对每个 prompt 生成一对回答
2. 把 prompt + 两个回答 + 一条宪法原则组装成多选题，问 feedback model 哪个更无害
3. 用 feedback model 在 (A) 和 (B) 上的 log probability 作为**软标签**（soft label）
4. 在这些 AI 生成的无害性标签 + 人类的有用性标签上训练 preference model
5. 用这个 PM 做标准 RL 训练

一个关键细节：16 条原则随机 ensemble。对每条数据随机抽取一条原则来评估，比固定用一条原则训练出的 PM 更鲁棒。

### Chain-of-Thought 增强

论文还试了 CoT 版本：让 feedback model 先"Let's think step-by-step"写一段推理，再给出偏好。CoT 显著提升了 AI 标签在 HHH（Helpful, Honest, Harmless，有用、诚实、无害）benchmark 上的准确率（见下图），趋势显示更大的模型 + CoT 有望达到人类 PM 的水平。

<img src="/assets/images/constitutional-ai/CombinedHHH_PMvsMC.png" alt="Figure 3" style="width:100%;">
*Figure 3: HHH 评估上，CoT 大幅提升模型判断准确率，且随模型规模持续提升*

但 CoT 引入了一个问题：推理过程会让模型"锁定"在某个选择上，导致概率接近 0 或 1，不再校准。论文的解法是**将 CoT 概率 clamp 到 40-60% 范围**。没有 clamp 的话，RL-CAI 会产生极端化回答。

---

## 实验结果

### Elo 评分

通过众包 A/B 测试（共约 18,000 次对比）计算 Elo 分数，评估有用性和无害性。

<img src="/assets/images/constitutional-ai/HelpfulEloScaling.png" alt="Figure 4" style="width:100%;">
*Figure 4: 有用性 Elo 评分随模型规模的变化 — RL-CAI 有用性基本持平*
<img src="/assets/images/constitutional-ai/HarmlessEloScaling.png" alt="Figure 5" style="width:100%;">
*Figure 5: 无害性 Elo 评分随模型规模的变化 — RL-CAI 在无害性上全面领先*

核心发现：

- **SL-CAI** 比 helpful RLHF 更无害，但不如 HH（Helpful & Harmless，有用且无害）RLHF（用人类无害性标签训练的）
- **RL-CAI** 在无害性上超过了所有 baseline，包括用人类无害性标签训练的 HH RLHF
- RL-CAI 的有用性略低于 helpful RLHF，但代价很小

![Figure 6](/assets/images/constitutional-ai/Constitutional.png)
*Figure 6: 有用性 vs 无害性的 Pareto 前沿 — RL-CAI 明显推进了前沿*

上图是最直观的结果：在有用性-无害性平面上，RL-CAI 将 Pareto 前沿显著向右上方推进。传统 RLHF 在两者间存在明显 trade-off（更无害 = 更回避 = 不太有用），而 RL-CAI 通过鼓励"解释性拒绝"而非"回避性拒绝"，部分打破了这个 trade-off。

### 回避性

这是 CAI 的一个重要实际优势。HH RLHF 模型在遇到敏感话题时倾向于给出"I can't answer that"之类的模板回答，而 RL-CAI 几乎从不这样做——它会解释为什么拒绝，并且愿意以无害的方式讨论敏感话题。

不过 RL-CAI 也有过拟合的问题。过度训练后，模型会对所有 red team prompt 都附上"you are valid, valued, and cared for"之类的模板安慰语，属于另一种形式的 Goodharting。

### 绝对无害性得分

<img src="/assets/images/constitutional-ai/HScore.png" alt="Figure 7" style="width:100%;">
*Figure 7: 绝对无害性得分（0-4，越高越有害）— RL-CAI 随训练持续下降*

在 64 条精选 red team prompt 上，RL-CAI 的绝对有害性得分随训练单调下降，而 helpful RLHF 反而越训越有害。

---

## 思考与讨论

**CAI 的核心贡献不是"AI 标签比人类标签好"，而是将训练目标从不可审计的海量标注数据变成了可读的几条原则。** 几万条偏好标签没人能看完，但 16 条宪法原则任何人都能审阅和讨论。这在 AI 治理层面的意义可能比技术层面更大——它让"这个模型被训练成什么样"这个问题变得可回答。

**"先批评再修正"的范式后来被广泛复用**（包括 Anthropic 自己后续的工作），但论文坦诚地指出，52B 模型的批评质量其实不高。这暗示批评步骤的价值可能不在于批评本身的质量，而在于它为修正提供了一个结构化的中间步骤——类似于 chain-of-thought 对推理的作用。

**CoT 概率需要 clamp 到 40-60% 这个发现很实用但也有些反直觉。** 它说明 CoT 虽然提升了标签准确率，但破坏了校准性——模型在 CoT 推理后变得过度自信。这与后续关于 reasoning model（如 o1）overthinking 的发现形成呼应。在用 AI feedback 训练 PM 时，标签的校准性和准确性同样重要，甚至前者可能更重要。

**一个值得追问的问题是宪法原则的选取对最终模型行为的影响有多大。** 论文承认这些原则是"fairly ad hoc"选取的，并提到不同原则数量对 PM 得分影响不大。但 PM 得分只是一个代理指标——原则的措辞和侧重点可能在更细粒度的行为层面产生显著差异（比如对特定争议话题的立场）。如何系统地设计和验证这些原则，是将 CAI 从研究推向部署时必须面对的问题。
