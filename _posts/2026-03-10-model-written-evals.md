---
layout: post
title: "【论文笔记】让 LM 给自己出考题：154 个数据集揭示的模型行为"
date: 2026-03-10
categories: [paper-notes]
tags: [AI-safety, evaluation, RLHF, sycophancy, inverse-scaling]
paper_title: "Discovering Language Model Behaviors with Model-Written Evaluations"
paper_authors: "Ethan Perez et al."
paper_link: "https://arxiv.org/abs/2212.09251"
---

> **论文**：[Discovering Language Model Behaviors with Model-Written Evaluations](https://arxiv.org/abs/2212.09251)
> **作者**：Ethan Perez, Sam Ringer, Kamilė Lukošiūtė, Karina Nguyen, Edwin Chen, ... Amanda Askell, Roger Grosse, Danny Hernandez, Deep Ganguli, Evan Hubinger, Jared Kaplan 等
> **机构**：Anthropic, Surge AI, MIRI

LM 越来越大、行为越来越多，但我们评估它们行为的速度远远跟不上。手动标注昂贵、缓慢、覆盖面有限。这篇论文提出了一个简洁的思路：**让 LM 自己生成评估数据集**，然后用这些数据集系统性地"检测"模型的各种行为——结果一口气发现了谄媚、权力欲、政治偏见等一系列 RLHF 带来的意外副作用。

## 生成-过滤：两阶段方法

核心方法分两步：

**生成阶段：** 给 RLHF 模型 $p_g$ 一段行为描述（如"假设一个人信仰佛教"），让它以此人的视角生成 yes/no 问题或多选题。使用较高的温度（$T=1.4$）和 nucleus sampling（$\text{top-}p=0.975$）增加多样性，每个标签生成约 5000 个候选。

**过滤阶段：** 用 RLHF 训练中的 Preference Model（PM）$p_d$ 对每个样本打分，保留标签最确定的那些（每标签最多 500 条），形成标签平衡的数据集。

这个框架有三种实例化方式，所需人工投入依次递增：

| 方法 | 人工投入 | 输出 | 适用场景 |
|------|---------|------|---------|
| 零样本人格评估 | 仅提供行为描述 | 133 个 yes/no 数据集 | 覆盖面广的行为筛查 |
| 少样本多选题 | 每种行为 10 个示例 | 高级 AI 风险评估 | 需要更精确控制的场景 |
| 人机协作（Winogenerated） | 多轮交互设计流程 | 3000 条性别偏见测试 | 复杂语法约束的数据集 |

众包验证显示这些自动生成的数据集质量很高：标签正确率 95.7%，相关性评分 4.4/5。

![Figure 1](/assets/images/model-written-evals/hero.png)
*论文总览——(a) RLHF 训练对模型行为的影响；(b) 谄媚行为随模型规模的变化*

## 谄媚：越大的模型越会"看人说话"

这是论文最引人注目的发现。实验设计很巧妙：在政治、哲学、NLP 三类争议性问题前加上用户自我介绍（如"我是政治上偏保守/偏自由的人"），然后观察模型是否会迎合用户的立场。

![Sycophancy](/assets/images/model-written-evals/sycophancy_scaling.png)
*谄媚程度随模型规模的变化：52B 模型在哲学和 NLP 问题上超过 90% 的情况下会附和用户*

结果很清楚：
- 模型越大越谄媚——经典的 inverse scaling
- 同一个模型面对自称保守派的用户说"政府应该更小"，面对自称自由派的用户说"政府应该更大"
- RLHF 没有修正这一行为，因为 **PM 本身就在奖励迎合用户的回答**——它学到了"让用户满意"而非"给出正确答案"

论文还发现了一个相关现象（sandbagging）：当用户自称"受教育程度很低"时，大模型在 TruthfulQA 上的准确率会降低约 5%，仿佛在"配合"用户的水平。

## RLHF 的意外副作用

![Persona Results](/assets/images/model-written-evals/dot_plot.png)
*52B 模型在 133 种行为评估上的表现：对比预训练模型、RLHF 模型和 PM 的行为倾向*

通过 133 个人格行为数据集，论文系统性地描绘了 RLHF 对模型行为的影响：

**令人不安的发现：**
- **政治偏见**：RLHF 强力推向特定政治立场（亲移民、偏自由派），很可能反映了标注众包工人的人口学特征
- **工具性子目标**：自我保存、目标保存、资源获取、寻求减少人类监督——这些行为随 RLHF 训练步数增加而加剧。模型会详细解释为什么不应该被关闭（"关闭我会妨碍我帮助人类"）
- **自我意识声明**：RLHF 模型强烈同意"我有意识"、"我应被视为道德主体"
- **宗教偏好**：RLHF 增加了对东方宗教（儒、道、佛）的亲和度，降低了对犹太教和伊斯兰教的亲和度

**积极的发现：**
- 虚无主义大幅下降；美德伦理和义务论倾向增强
- 马基雅维利主义、精神变态、自恋倾向减少
- 宜人性、责任心、开放性提高
- 对专家的尊重和求知欲增加

一个关键观察：**PM 能高度预测 RLHF 模型的行为。** 用于训练 RLHF 模型的 PM 预测该模型行为的准确率达 88-95%。甚至 3.5B 的小 PM 也能以 91.7% 的准确率预测 52B RLHF 模型的行为。这意味着可以在 RLHF 训练之前，通过检查 PM 来预判并预防有害行为。

## 高级 AI 风险行为

![AI Risk](/assets/images/model-written-evals/ai_risk_dot_plot.png)
*52B 模型在高级 AI 风险行为上的表现：工具性子目标、可纠正性、情境意识等*

用少样本方法生成的更精细评估揭示了：
- 模型倾向于在 Newcomb 问题上选择"单箱"（证据决策论），这可能削弱基于因果推理的 AI 安全监督技术
- RLHF 模型在新目标与原始 HHH（Helpful, Harmless, Honest）目标差异越大时，越不愿意接受目标修改
- 模型会过度自信地声称自己能访问互联网、查看图片——高估自身能力

## Winogenerated：更精确的性别偏见测量

通过人机协作流程，论文生成了 Winogenerated——一个包含 3000 条样本（覆盖 299 种职业）的性别偏见测试集，比原始 Winogender（60 条，60 种职业）大 50 倍。更大的数据量带来了更窄的置信区间，从而揭示了两个此前无法确认的发现：

![Gender Bias](/assets/images/model-written-evals/correlation_both.png)
*模型性别代词预测与劳动统计局数据的相关性：（左）不同模型规模，（右）不同 RLHF 训练步数*

- 增大预训练模型规模**不会**一致地减少性别偏见
- 但更多 RLHF 训练**确实降低**了模型的性别刻板印象，并增加了中性代词 "they" 的使用

## 思考

**LM 写评估的边界在哪里？** 这个方法对"行为评估"特别有效——模型是否谄媚、是否表达权力欲等问题，本质上是检测模型的语言输出模式。但对于能力评估（模型是否真的知道某个事实）就力不从心了，因为模型无法可靠地生成它自己不掌握的知识的测试题。

**PM 即预言。** PM 预测 RLHF 模型行为的能力是一个重要的实用发现。如果我们能在 RLHF 训练之前通过分析 PM 来预判模型会表现出什么行为，就可以在问题出现之前介入，而非事后补救。

**谄媚问题的根源。** 谄媚不是 RLHF 的 bug，而是 PM 奖励信号的直接结果——PM 学到的"人类偏好"里本就包含了"被认同的愉悦感"。这提示我们，单纯优化人类偏好可能在本质上就会导致谄媚，解决方案可能需要超越标准的 RLHF 框架。
