---
layout: post
title: "【论文笔记】Tokenization 的诅咒：LLM 对子词切分的脆弱性"
date: 2026-05-13
categories: [paper-notes]
tags: [tokenization, LLM, robustness, BPE, evaluation]
paper_title: "Tokenization Falling Short: On Subword Robustness in Large Language Models"
paper_authors: "Yekun Chai et al."
paper_link: "https://arxiv.org/abs/2406.11687"
---

# Tokenization Falling Short: LLM 对子词切分的脆弱性

> **论文**：[Tokenization Falling Short: On Subword Robustness in Large Language Models](https://arxiv.org/abs/2406.11687)
> **作者**：Yekun Chai, Yewei Fang, Qiwei Peng, Xuhong Li
> **机构**：Baidu, ModelBest, University of Copenhagen
> **发表**：EMNLP 2024

---

## 快速概览

我们日常使用 LLM 时很少会注意到 tokenization 环节——它就像空气一样理所当然。但这篇论文系统地揭示了一个事实：**subword tokenization 本身就是 LLM 的能力瓶颈之一**。作者将这一系列问题统称为 "the curse of tokenization"（分词诅咒），具体表现为三个方面：

1. **对拼写错误极度敏感**：一个 typo 可能把一个 token 拆成完全不同的 subword 序列
2. **对 token 内部结构无感知**：模型不知道 "strawberry" 里有几个 r
3. **对文本长度缺乏意识**：无法可靠地数字符或数单词

论文通过 13 个任务对 Llama3、Mistral、GPT-4 等模型做了全面评估，并提出 BPE-dropout 作为缓解手段。

---

## 背景：为什么 Tokenization 会成为问题

现代 LLM 几乎都使用 Byte Pair Encoding (BPE) 或其变体做分词。BPE 的核心逻辑是反复合并训练语料中最高频的相邻字节对，最终形成一个固定大小的词表。模型输入的不是字符序列，而是词表中的 token ID 序列。

这个过程带来一个根本性矛盾：**token 是模型的最小处理单元，但人类的最小语义单元是字符**。当你问模型 "strawberry 里有几个 r"，模型看到的可能是 `[str, aw, berry]` 这样的 token 序列——它从未被训练去"拆开"一个 token 看里面的字母组成。

更麻烦的是，BPE 的切分结果对输入的微小扰动非常敏感。比如 "hello" 和 "helo"（少打一个 l），可能被切成完全不同的 token 序列，导致模型的理解产生不成比例的偏差。

---

## 实验设计：三个研究问题

论文围绕三个逐步深入的 Research Question 组织实验。

### RQ1：复杂问题求解（Anagram 和 LaTeX 理解）

**Anagram 任务**测试模型能否把打乱的字母重组为正确单词。例如输入 "nad"，期望输出 "and"。这直接考验模型是否能在字符层面操作 token 内容。

![Figure 1](/assets/images/tokenization-falling-short/x1.png)

上图展示了不同模型在 Word Unscrambling (WU) 和 Cycled Letters (CL) 任务上随 few-shot 数量变化的表现。关键观察：

- GPT-4 Turbo 在 WU 任务上远超其他模型（约 80% EM），说明参数规模确实有帮助
- 但所有模型在 CL 任务上表现都较差（即便 GPT-4 也只有约 40%）
- 增加 few-shot 示例对小模型帮助有限

**LaTeX 数学理解**任务要求模型识别 LaTeX 公式对应的数学定理。Llama3-70B 在 1-shot 下达到 79.25%，但随着示例增加反而波动下降——说明 few-shot 并非万能。

| 模型 | 0-shot | 1-shot | 2-shot | 3-shot |
|------|--------|--------|--------|--------|
| Llama3-8B | 41.51 | 45.28 | 45.28 | 35.85 |
| Llama3-70B | 62.26 | **79.25** | 69.81 | 71.70 |
| Mistral-7B | 47.20 | 43.40 | 37.70 | 37.70 |
| Mixtral-8x7B | 49.10 | 56.60 | 64.20 | 62.30 |

### RQ2：Token 结构探测

这是论文最有意思的部分。作者设计了两类探测任务来直接测试模型是否"理解" token 的内部结构。

**Intra-token 探测**（token 内部）包含四个子任务：

- **Character Count (CC)**：数某个字符在单词中出现几次（如 "undertake" 中 "e" 出现 2 次）
- **N-th Character (NC)**：输出单词的第 $n$ 个字符（如 "dual" 的第 4 个字符是 "l"）
- **N-th Character Reverse (NCR)**：从末尾数第 $n$ 个字符
- **Case Conversion (CCV)**：大小写转换

![Figure 3](/assets/images/tokenization-falling-short/x3.png)

结果很有启发性：

- CC 任务中，Llama3-8B 从 zero-shot 的 0% 飙升到 3-shot 的 81%——说明模型其实"可以学会"数字符，但需要示例激活这个能力
- NCR（反向字符定位）对所有模型都很难，GPT-4 最高也只有 52%
- 这说明模型的 token embedding 确实没有很好地编码字符级结构信息

**Inter-token 探测**（token 之间）测试模型找两个单词的公共子串和公共子序列：

- Common Substrings (CS)：找公共子串
- Longest Common Substrings (LCS)：找最长公共连续子串
- Longest Common Subsequences (LCSeq)：找最长公共子序列（不要求连续）

LCSeq 任务的表现普遍很差，这是因为它需要模型在字符层面做动态规划式的匹配——这对基于 subword 的模型来说几乎是不可能完成的任务。

### RQ3：拼写噪声鲁棒性

论文在 MMLU、TruthfulQA、GSM8K、HumanEval 四个 benchmark 上注入两个层级的拼写噪声：

**字符级扰动**：
- Permutation：在词内以 $n$-gram（$n$=2,3,5）为单位打乱字符顺序，概率 50%
- Noise：以 10% 概率对字符做插入、删除、替换

**Token 级扰动**：
- Permutation：以 $n$-gram 为单位打乱 token 顺序
- Noise：以 30% 概率对 token 做插入、删除、替换

![Figure 5](/assets/images/tokenization-falling-short/x5.png)

上图展示了 HumanEval 代码生成任务中字符级噪声的效果。可以看到，即使是人类可以轻松理解的轻微拼写扰动，也能让代码生成任务的性能大幅下降。

核心发现：

- **噪声比重排更具破坏力**：在所有 benchmark 上，noise injection（增删改字符）造成的性能下降远大于 permutation（仅重排）
- **字符级扰动比 token 级扰动更致命**：这说明模型对 token 边界内的字符信息非常依赖
- **大模型并不免疫**：GPT-4 在 reorder 扰动下表现稳定，但面对 noise injection 同样有明显退化

---

## 缓解方案：BPE-Dropout

论文的第 6 节提出了一个相对简单但有效的缓解方案——BPE-dropout（Provilkov et al., 2020）。

BPE-dropout 的思路是：在训练时，随机跳过一部分 BPE merge 操作。如果正常 BPE 会把 "un" + "do" 合并为 "undo"，那么在 dropout 率为 $p$ 的情况下，有 $p$ 的概率这次合并不发生，模型就会看到 "un" 和 "do" 两个独立 token。这样做的好处是让模型在训练过程中见到同一个词的多种切分方式，从而获得对 token 内部结构的更好理解。

作者在 Mistral-7B 上用 111k 合成数据做了 5 个 epoch 的 post-training，对比了不同 dropout 率的效果。

![Figure 4](/assets/images/tokenization-falling-short/x4.png)

上图展示了 BPE-dropout 在不同 dropout 率 $p$ 下的表现（虚线为 baseline，即 $p=0$）。关键结论：

| Dropout 率 | 效果 |
|-----------|------|
| $p=0$（baseline） | CS、CC 等简单任务表现不错，但复杂任务一般 |
| $p=0.2$ | **整体最优**，在 CCV、NC、LCS 等任务上稳定超越 baseline |
| $p=0.4$ | 多数任务仍有提升，但边际收益递减 |
| $p=0.6, 0.8$ | 性能下降，可能是训练不充分（仅 5 个 epoch） |

适度的 BPE-dropout（$p=0.2$）通过引入分词多样性，有效提升了模型对 token 结构的感知能力。更高的 dropout 率虽然理论上提供更多多样性，但也需要更多训练计算来收敛。

---

## 思考与讨论

**Tokenization 是 LLM 的"原罪"吗？** 从这篇论文的实验来看，subword tokenization 确实在根本层面限制了模型对字符级信息的处理能力。但这并不意味着要抛弃 BPE——论文也指出大规模参数可以部分缓解问题，而且 BPE-dropout 这样的正则化手段成本很低。

**Tokenization-free 方向值得关注。** 论文在 Related Work 中提到了将文本渲染为图片然后做视觉建模的 tokenization-free 方法。这个方向虽然计算成本更高，但天然具备字符级感知能力和多语言泛化能力，可能是长期解决方案。

**实际应用启示：** 如果你的应用场景涉及用户输入（容易有 typo）、字符级操作（如验证码、密码检查）、或精确的文本格式要求，需要意识到当前 LLM 在这些场景下的脆弱性，并考虑在前处理阶段加入拼写纠正或在 fine-tuning 时使用 BPE-dropout。

---

## 总结

这篇论文的价值在于系统性地量化了一个大家隐约知道但缺乏严谨评估的问题。通过 13 个精心设计的任务，作者清晰地展示了 "分词诅咒" 的三个维度（长度无感知、大小写混淆、内部结构盲区），并验证了 BPE-dropout 作为低成本缓解手段的有效性。对于从事 LLM 开发和评估的研究者来说，这些发现提醒我们：在追求更大模型、更多数据的同时，tokenization 这个看似已经"解决"的预处理步骤仍然值得重新审视。
