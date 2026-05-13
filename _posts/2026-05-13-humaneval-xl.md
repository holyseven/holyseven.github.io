---
layout: post
title: "【论文笔记】HumanEval-XL：大规模多语言代码生成评测基准"
date: 2026-05-13
categories: [paper-notes]
tags: [code-generation, multilingual, benchmark, LLM, evaluation]
paper_title: "HumanEval-XL: A Multilingual Code Generation Benchmark for Cross-lingual Natural Language Generalization"
paper_authors: "Qiwei Peng et al."
paper_link: "https://arxiv.org/abs/2402.16694"
---

> **论文**：[HumanEval-XL: A Multilingual Code Generation Benchmark for Cross-lingual Natural Language Generalization](https://arxiv.org/abs/2402.16694)
> **作者**：Qiwei Peng, Yekun Chai, Xuhong Li
> **机构**：University of Copenhagen, Baidu Inc.

## 一句话概览

现有代码生成评测主要关注"英语 prompt → 多种编程语言"，而忽略了一个更基本的问题：**当同一编程任务用不同自然语言描述时，LLM 的代码生成能力是否一致？** HumanEval-XL 建立了 23 种自然语言 × 12 种编程语言的平行评测集（共 22,080 个 prompt），发现当前 LLM 在跨语言泛化上存在显著差距。

## 研究动机

代码生成领域的评测格局长期存在两个偏向：

1. **英语中心**：HumanEval、MBPP、APPS 等经典 benchmark 几乎只考虑英语 prompt → Python 代码。
2. **自然语言覆盖不足**：即使有多语言尝试（如 ODEX 支持 4 种自然语言），覆盖面仍然很窄，且不同语言之间的数据**不平行**——你无法公平比较模型在中文 prompt 和英文 prompt 下的表现，因为题目本身不同。

这导致一个关键问题无从回答：**如果同一个编程任务分别用英语、中文、阿拉伯语描述，模型生成的代码质量差多少？** 这个问题之所以重要，是因为全球大量开发者的母语并非英语，如果模型对非英语指令的理解明显弱于英语，那么它对非英语用户的实际可用性就要打折扣。

## 数据集构建

![Figure 1](/assets/images/humaneval-xl/x1.png)
*Figure 1: HumanEval-XL 数据构建流程。四个阶段：文本提取 → GPT-4 翻译与回译 → BERTScore 质量检测 → 人工质控。*

HumanEval-XL 以 Multilingual HumanEval（已有英语 prompt + 12 种编程语言）为起点，通过以下四阶段流水线将其扩展到 23 种自然语言：

### Stage 1：提取 NL 文本

从每个编程题的 prompt 中剥离出自然语言描述部分（去掉函数签名、代码框架等）。

### Stage 2：翻译与回译

使用 GPT-4 将提取的英文描述翻译为 23 种目标语言，同时将每种翻译结果回译为英文，用于后续质量检测。

### Stage 3：BERTScore 自动质检

计算回译英文与原始英文之间的 BERTScore 相似度。阈值设为 0.95——若低于此值，重新翻译（最多重试 3 次），仍不达标则丢弃该样本。

### Stage 4：人工审核

对通过自动筛选的样本进行启发式规则检查和随机抽样人工校对。

最终得到的数据集覆盖：
- **23 种自然语言**，横跨 11 个语系（从日耳曼语族到汉藏语系、阿尔泰语系）
- **12 种编程语言**：Python, Java, Go, Kotlin, PHP, Ruby, Scala, JavaScript, C#, Perl, Swift, TypeScript
- **80 道平行编程题**，共 22,080 个 prompt，平均每题 8.33 个 test case

与现有 benchmark 的对比：

| 数据集 | 样本数 | 平均测试用例 | PL 数 | NL 数 | 平行? |
|--------|--------|------------|-------|-------|------|
| HumanEval | 164 | 7.7 | 1 | 1 | 否 |
| MBPP | 974 | 3.0 | 1 | 1 | 否 |
| APPS | 10,000 | 13.2 | 1 | 1 | 否 |
| Multilingual HumanEval | 1,935 | 7.8 | 12 | 1 | 否 |
| ODEX | 945 | 1.8 | 1 | 4 | 否 |
| **HumanEval-XL** | **22,080** | **8.3** | **12** | **23** | **是** |

"平行"是这个 benchmark 最重要的性质：同一道题在 23 种语言下的描述语义等价，因此可以直接对比模型在不同 NL 下的表现，将性能差异归因于模型对该语言的理解能力。

## 实验结果

论文评测了四个模型家族：CodeT5+（encoder-decoder，220M/770M/2B）、CodeGen2（decoder-only，1B/3.7B/7B/16B）、GPT-3.5 和 GPT-4。评测指标为 pass@1。

![Figure 2](/assets/images/humaneval-xl/x2.png)
*Figure 2: 各模型在 12 种编程语言上的 pass@1 结果（横轴为 23 种自然语言，按资源丰富度排序）。*

### 主要发现

**1. GPT-4 全面领先，但非英语语言仍有明显性能下降。**

以 Python 为例（按语言资源等级分组）：

| 模型 | Class 5 (高资源) | Class 4 (中资源) | Class 3 (低资源) |
|------|-----------------|-----------------|-----------------|
| CodeT5+ (2B) | 0.63 ± 1.53 | 0.94 ± 0.88 | 0.83 ± 0.63 |
| CodeGen2 (3.7B) | 15.42 ± 1.88 | 14.69 ± 2.39 | 14.31 ± 1.41 |
| CodeGen2 (16B) | 20.83 ± 1.51 | 19.06 ± 2.65 | 19.58 ± 1.25 |
| GPT-3.5 | 62.50 ± 5.06 | 66.41 ± 4.25 | 60.42 ± 2.86 |
| GPT-4 | 78.54 ± 2.90 | 78.75 ± 3.54 | 77.64 ± 4.07 |

Class 5 包含英、西、法、中、阿拉伯、德语；Class 3 包含南非荷兰语、印尼语、保加利亚语等低资源语言。差距不算巨大，但方向一致：低资源语言 prompt 下模型表现更差。

**2. 专用代码模型 vs 通用模型的有趣反转。**

GPT-3.5 参数远多于 CodeGen2-16B，但在 Python 之外的几乎所有编程语言上，GPT-3.5 的 pass@1 反而低于 CodeGen2-16B。这说明大规模代码预训练对代码生成能力至关重要——通用语言模型并不能简单通过参数规模碾压专用代码模型。

**3. 模型架构对代码生成的影响。**

CodeT5+（encoder-decoder 结构）在几乎所有场景下 pass@1 接近 0，与同等参数量的 CodeGen2（decoder-only 结构）差距悬殊。这意味着在未经指令微调的情况下，encoder-decoder 结构在代码补全任务上处于明显劣势。

**4. 编程语言之间的难度差异。**

Python 是最容易的目标语言（GPT-4 pass@1 可达 ~80%），而 Scala 最难（GPT-4 仅 ~25%，其余模型几乎为 0）。Go 也偏困难。这反映了预训练数据中不同编程语言的分布差异。

**5. 同家族模型在语言偏好上高度相关。**

CodeGen2-3.7B 与 16B 在各 NL 上的 pass@1 排名呈 Pearson 相关 0.8；GPT-3.5 与 GPT-4 为 0.87。说明参数量扩大不改变模型对各自然语言的相对偏好——它只是整体提升所有语言的性能，不会选择性地弥合某些语言的劣势。但不同家族间没有这种相关性。

### 按语系看表现

| 语系 | CodeGen2 (16B) | GPT-3.5 | GPT-4 |
|------|---------------|---------|-------|
| Afro-Asiatic (阿拉伯语、希伯来语) | 19.38 | 56.25 | 75.00 |
| Indo-European/Germanic (英、德、荷等) | 20.94 | 64.06 | 80.31 |
| Indo-European/Romance (葡、西、法、意) | 20.31 | 66.25 | 79.06 |
| Sino-Tibetan (中文) | 20.00 | 65.00 | 78.75 |
| Indo-European/Greek | 17.50 | 53.75 | 71.25 |
| Turkic (土耳其语) | 18.75 | 62.50 | 73.75 |

Afro-Asiatic、Greek、Turkic 语系的表现整体偏低。可能的原因包括：训练数据中这些语言的代码相关文本更少，以及这些语言与英语的形态差异更大（如阿拉伯语的 RTL 书写、希腊语的独特字母）。

## 思考与讨论

**这个 benchmark 测的到底是什么？** 表面上是代码生成能力，但由于同一题在不同 NL 下语义等价，性能差异本质上反映的是模型对各种自然语言的**语义理解能力**。模型不是"不会写代码"，而是"没读懂 prompt"。

**BERTScore 0.95 阈值够严格吗？** 回译质量检测是一个巧妙的设计，但 BERTScore 本身对细微语义差异（如数值条件的翻译错误）可能不敏感。论文也承认需要人工审核作为补充。

**GPT-4 的跨语言泛化是否已经"够好"？** 从数据看，GPT-4 在高资源语言组（78.54）和低资源语言组（77.64）之间的差距仅约 1 个百分点（Python 上）。这说明对于足够强的模型，跨语言泛化的瓶颈可能已经不在语言理解本身，而在代码生成的上限。但对于中小模型，这个差距仍然显著。

**局限性**：数据集仅 80 道题（来自 HumanEval），任务复杂度有限；翻译依赖 GPT-4，可能存在翻译腔；未评测更新的开源代码模型（如 CodeLlama、StarCoder 等）。
