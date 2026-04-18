---
layout: post
title: "【技术探索】基于 Cascaded Selective Evaluation 的新方法"
date: 2026-04-07
categories: [tech-exploration]
tags: [LLM-as-judge, selective-prediction, evaluation, experiments]
origin_paper: "Trust or Escalate: LLM Judges with Provable Guarantees for Human Agreement"
origin_note: "2026-03-07-cascaded-selective-eval"
---

# 基于 Cascaded Selective Evaluation 的新方法探索

> **出发点**：[Trust or Escalate 论文笔记](2026-03-07-cascaded-selective-eval.md)

---

## 核心想法

<!-- 用一两段话描述你的新技术思路，和原论文的区别/改进点 -->
也不算新方法，是将few-shot的思路应用到主观任务上，few-shot使用人工构建的示例。分三个实验阶段：

1. 搜集完人工构建的示例，few-shot中只使用判别结果，不使用具体理由。

> 这个方式，使用一些弱模型效果还不错（除去gemini-3.1-pro和claude-4.6-opus），以gemini-3.1-flash-lite和deepseek-v3.2效果最好。但是强模型效果不好

> 这就导致了一些反直觉的发现：使用弱模型进行判别，比使用强模型效果还好。

> 分析后发现，gemini-3.1-pro和claude-4.6-opus很自信，多次请求较稳定；对结果的判断不容易受few-shot的影响，给的分析能自洽且合理。

> 这就导向了，ground truth并不准确（的确，使用的是专家标注的那1136条，以高质判断为主要准则）

> 另外一个点就是，few-shot效果不明显 -> 目前想的是加入人工判别的理由进去（不过这个是不是要进行人工二次矫正？）

2. 支线：不使用few-shot，给定准则进行判定。

> 结论没变，还是弱模型比强模型效果更好。

> 所以要变的，应该是使用一个更准确的ground truth。

3. 想先验证的一个点，是不是V1版本的prompt写的不好，没有要求模型要仔细查看给的示例，按照示例的逻辑来进行判断。(相关的代码可以查看run_judge_eval.py和prompt_v2.py)  

你跟我讨论下，你有什么想法吗？

---

## 方法设计

<!-- 描述你的方法的具体设计，可以包括公式、流程图等 -->



---

## 实验记录

### 实验 1：

**日期**：2026-04-07

**目标**：

**设置**：
- 数据集：
- 模型：
- 基线：
- 超参数：

**结果**：

| 指标 | 基线 | 本方法 |
|------|------|--------|
|      |      |        |

**观察**：

---

## 遇到的问题与分析

### 问题 1：

**现象**：

**可能原因**：

**尝试的解决方案**：

**结论**：

---

## TODO

- [ ]
- [ ]
- [ ]

---

## 阶段性总结

<!-- 定期更新，记录当前进展和下一步方向 -->


