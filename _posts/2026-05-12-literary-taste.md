---
layout: post
title: "【论文笔记】LiteraryTaste：面向个性化创意写作的阅读偏好数据集"
date: 2026-05-12
categories: [paper-notes]
tags: [NLP, personalization, preference-learning, creative-writing, reward-model]
paper_title: "LiteraryTaste: A Preference Dataset for Creative Writing Personalization"
paper_authors: "John Joon Young Chung et al."
paper_link: "https://arxiv.org/abs/2511.09310"
---

> **论文**：[LiteraryTaste: A Preference Dataset for Creative Writing Personalization](https://arxiv.org/abs/2511.09310)
> **作者**：John Joon Young Chung, Vishakh Padmakumar, Melissa Roemmele, Yi Wang, Yuqian Sun, Tiffany Wang, Shm Garanganao Almeda, Brett A. Halperin, Yuwen Lu, Max Kreminski
> **机构**：Midjourney, Stanford University, UC Berkeley, University of Washington, University of Notre Dame

## 快速概览

当前 LLM 在创意写作领域倾向于产出同质化文本，根本原因在于训练时使用了将所有标注者偏好视为单一整体的 monolithic reward model。本文提出 **LiteraryTaste** 数据集，从 60 位标注者处收集了两类偏好数据：

1. **Revealed preference**（揭示偏好）：每人对 100 对短文本片段做二选一偏好标注
2. **Stated preference**（陈述偏好）：通过问卷自报阅读习惯和审美偏好

核心发现：

- 人们在创意写作偏好上存在显著分歧（Fleiss' Kappa 仅 0.14）
- 微调 transformer encoder 个人化建模准确率可达 **75.8%**，即使只用 15 个样本也能达到约 70%
- 陈述偏好对推断揭示偏好的帮助有限——二者包含部分矛盾信息
- LLM few-shot prompting 在个人偏好建模上表现不佳（约 55-56%），远逊于简单的监督学习方法

## 数据集设计

### 揭示偏好：文本对比选择

标注者面对一对 150 词的文本片段，选择更喜欢的一个（或标记 "unsure"）。文本来源覆盖五类语料：

| 来源 | 特征 | 规模 |
|:---|:---|:---:|
| Project Gutenberg | 2000 年前的经典文学 | 1063 |
| Sterman et al. | Amazon Kindle 现代小说预览 | 1014 |
| r/WritingPrompts | 在线社区业余创作 | 1092 |
| Poetry Foundation | 文艺复兴/现代诗歌 | 308 |
| Tell-me-a-story | 专业作家协作短故事 | 123 |

此外加入了 LLM 生成文本（Claude Sonnet 和 GPT-4.1 各半）与人类文本的对比对，最终构成 2000 对文本，每对由 3 位标注者独立标注。

### 陈述偏好：阅读画像问卷

问卷涵盖四个维度：阅读频率（每周时长、每月读书量）、阅读动机（享受/学习/健康）、偏好体裁（虚构/非虚构类型）、以及文本品质偏好（如 "引发情感"、"语言丰富" 等 12 项四级量表评分）。

## 核心实验与结果

### RQ1：人们的文学品味是否存在分歧？

**结论：是的，且相当显著。**

- 揭示偏好的 Fleiss' Kappa 平均仅 0.14（slight agreement）
- 陈述偏好中，除阅读频率（alpha = 0.47）外，其余维度 Krippendorff's alpha 均低于 0.2

这意味着"好的写作"在不同读者眼中差异极大，用聚合偏好训练的 reward model 本质上是在抹平这些差异。

### RQ2：能否建模个人揭示偏好？

论文对比了多种方法，以 10-fold cross-validation 在 60 位标注者上评估：

![RQ2 个人偏好建模结果](/assets/images/literary-taste/fig4.png)
*各方法在个人偏好建模上的测试准确率对比。*

| 方法 | 测试准确率 |
|:---|:---:|
| Full-Finetuning (ModernBERT-large) | **75.8%** |
| Neural Network-All (嵌入+两层 MLP) | ~71% |
| Logistic Regression-All | ~62% |
| o4-mini-Synth (SynthesizeMe!) | ~61% |
| o4-mini / Sonnet-4 few-shot | ~55-56% |
| Decision Tree | ~54% |

关键方法说明：**Full-Finetuning** 将 ModernBERT-large 作为 reward model，使用 Bradley-Terry 式的 binary ranking loss：

$$\mathcal{L} = -\log(\sigma(r_\theta(x_c) - r_\theta(x_r)))$$

其中 $x_c$ 为被选择文本，$x_r$ 为被拒绝文本，$r_\theta$ 为模型打分函数。这个 loss 直接优化模型对 "chosen 分数高于 rejected" 的判别能力。

**样本效率分析**是本文一个重要的实践结论：

![样本量 vs 准确率](/assets/images/literary-taste/fig5.png)
*不同训练样本量下各方法的性能变化。*

Full-Finetuning 仅需 15 个偏好样本即可达到约 70% 准确率，说明基于预训练模型的个人偏好建模是**样本高效**的。性能在 90 样本时尚未饱和，更大数据集可能带来进一步提升。

### RQ3：聚合偏好建模更难还是更容易？

![聚合偏好建模结果](/assets/images/literary-taste/fig6.png)
*聚合偏好建模结果，同时对比了个人偏好建模的最佳方法。*

将 3 位标注者的偏好通过多数投票聚合后：

- Agg-Full-Finetuning 和 Agg-o4-mini-Zero（零样本 LLM）并列最佳，约 67.7%
- 但这低于个人偏好建模的 75.8%——**聚合偏好更难建模**
- LLM 零样本在聚合偏好上表现不错，说明 LLM 已经内化了某种"大众审美"

有趣的是，为 LLM 提供 few-shot 样本反而降低了聚合偏好的预测准确率，可能是少量样本引入了噪声，干扰了模型预训练阶段学到的通用偏好知识。

### RQ4：陈述偏好能否帮助推断揭示偏好？

这是论文中最具实用价值的问题之一：如果我们能通过问卷直接推断用户的实际偏好，就不需要收集大量标注数据。

结论令人遗憾——**陈述偏好的帮助非常有限**：

- 大多数跨标注者方法准确率接近随机（50%）
- 表现最好的是 Cross-LR-Weight（从陈述偏好预测 logistic regression 权重），跨标注者准确率 63.4%
- 但仍远不如只用个人揭示偏好数据训练的 Full-Finetuning（75.7%）

这暗示人们"说的"和"做的"之间存在gap——自我报告的审美偏好无法精确预测面对具体文本时的实际选择。

### RQ5：偏好维度分析

论文使用 Lloom（LLM 驱动的概念提取工具）从文本中提取了 13 个关键维度，包括：家庭关系、悬念与张力、身份认同与转变、感官意象、对话驱动、类型惯例、隐喻与拟人等。

![标注者偏好聚类](/assets/images/literary-taste/fig9.png)
*10 个标注者聚类的偏好向量热力图。上排为大聚类（偏好较温和），下排为小聚类（偏好强烈且极化）。*

通过层次聚类将 60 位标注者分为 10 组：

- **大聚类**（上排）：偏好方向温和，成员多，但仍各有侧重。如 Cluster 1 不偏好对话，Cluster 2 明确偏好对话驱动的叙事
- **小聚类**（下排）：偏好极端，如 Cluster 9（单人）对对话元素有压倒性偏好

另一个引人关注的发现：**大多数标注者更偏好 LLM 生成的文本**而非人类写作——60 位标注者中仅 6 位更偏好人类文本。这与 "LLM 写作质量不佳" 的直觉相矛盾，可能反映了 LLM 文本在表面流畅性和信息密度上的优势。

## 聚合偏好的方向性

![聚合偏好向量](/assets/images/literary-taste/fig11.png)
*所有标注者的聚合偏好向量，通过多数投票聚合后训练 logistic regression 得到的系数。*

从聚合偏好来看，读者整体上偏好：悬念与张力、身份认同主题、感官意象丰富的文本；不偏好：重复/碎片化结构、过于依赖类型惯例的写法。

## 讨论与启示

论文为构建个性化创意写作系统提供了实践指南：

1. **有 GPU 资源时**：收集揭示偏好 + 微调 transformer encoder，15 个样本即可启动，越多越好
2. **只能做推理时**：用 embedding model + 两层 neural network，需要约 90 个偏好样本
3. **只有 LLM API 时**：结合陈述偏好做 prompting，但不要期望高精度

从更宏观的视角看，这篇论文揭示了 RLHF/DPO 范式中被忽视的问题：单一 reward model 本质上是在不同人的品味之间做平均，这在创意写作这种高度主观的领域尤其成问题。未来的方向应该是**per-user reward model** 或条件化的偏好建模——而 LiteraryTaste 数据集为这方面的研究提供了基础设施。

局限性方面：数据集仅涉及 150 词短文本片段，无法反映对叙事弧线等长文本结构的偏好；60 位标注者的规模对跨用户泛化建模仍显不足；陈述偏好问卷设计可能不够全面。
