---
layout: post
title: "【论文笔记】从偏导数出发重新推导 Transformer 中的 Token 归因"
date: 2026-05-16
categories: [paper-notes]
tags: [explainability, Transformer, attention, ViT, BERT, CLIP]
paper_title: "Beyond Intuition: Rethinking Token Attributions inside Transformers"
paper_authors: "Jiamin Chen et al."
paper_link: "https://openreview.net/forum?id=rm0zIzlhcX"
---

> **论文**：[Beyond Intuition: Rethinking Token Attributions inside Transformers](https://openreview.net/forum?id=rm0zIzlhcX)
> **作者**：Jiamin Chen, Xuhong Li, Lei Yu, Dejing Dou, Haoyi Xiong
> **机构**：北航, 百度
> **发表**：Transactions on Machine Learning Research (02/2023)

## 快速概览

已有的 Transformer 解释方法（如 Attention Rollout、Generic Attribution）或多或少依赖直觉式设计——比如假设线性投影矩阵是单位阵、直接丢弃负梯度等。这些隐含的近似在模型变深（ViT-Large）或采用全局池化（ViT-MAE）时会引发误差累积，导致解释结果失真。本文从 loss 对输入 token 的偏导数出发，通过链式法则将归因过程拆分为 Attention Perception 和 Reasoning Feedback 两个子问题，分别给出显式的数学推导，最终在 BERT、ViT、CLIP 等多种 Transformer 上取得比已有方法更好的 faithfulness 评估结果。

## 问题出发点：直觉式方法的误差累积

![Generic Attribution 与本文方法在不同 ViT 变体上的对比](/assets/images/token-attribution-transformer/fig1.png)
*Figure 1: Generic Attribution 在 ViT-B 上表现尚可，但在更深的 ViT-L 和采用全局池化的 ViT-MAE 上出现明显退化。本文方法在三种变体上保持一致。*

Transformer 的解释方法需要回答一个核心问题：每个输入 token 对最终预测贡献了多少？已有方法大致分为两类：

- 基于注意力的方法：如 Attention Rollout（Abnar & Zuidema, 2020）假设各层注意力可以线性叠乘传播，Generic Attribution（Chefer et al., 2021b）在 Rollout 基础上加入梯度信息
- 基于梯度的方法：如 GradCAM、LRP 等

这些方法的共同问题是：它们在设计过程中引入了一些未经验证的近似（比如 Rollout 假设 $W^{(l)} = I$，即线性投影矩阵为单位阵），这些近似在浅层模型上误差可控，但当模型变深时误差会逐层累积。Figure 1 直观展示了这一点。

## 形式化框架：从偏导数到两阶段分解

![方法总览](/assets/images/token-attribution-transformer/fig2.png)
*Figure 2: 左侧是 ViT 的标准结构，右侧是本文的解释框架。从偏导数出发，用链式法则拆分为 Token Attributions = Attention Perception $\odot$ Reasoning Feedback。*

### 基本思路

给定输入 $X$（$N$ 个 token），预测类别 $c$ 的 loss 为 $\mathcal{L}^c(X)$。目标是计算每个 token $i$ 的归因分数 $\frac{\partial \mathcal{L}^c(X)}{\partial t_i}$，其中 $t_i$ 是输入向量在一组基向量（basis）$\mathbb{B} = \{\tilde{X}_{\text{CLS}}, \tilde{X}_1, \ldots, \tilde{X}_N\}$ 上的坐标。

通过链式法则，将这个偏导数拆分为两部分：

$$\frac{\partial \mathcal{L}^c(X)}{\partial t_i} = \frac{\partial \mathcal{L}^c(X)}{\partial \mathbf{Z}_{\text{CLS}}^{(L)}} \cdot \frac{\partial \mathbf{Z}_{\text{CLS}}^{(L)}}{\partial t_i}$$

其中 $\mathbf{Z}_{\text{CLS}}^{(L)}$ 是最后一个 attention block 输出的 [CLS] token 向量。这个拆分对应两个子问题：

- Attention Perception $\mathbf{P}^{(L)}$：描述输入 token 如何通过 $L$ 层 attention block 交互融合，最终形成 [CLS] 向量的过程。对应 $\frac{\partial \mathbf{Z}^{(L)}}{\partial t_i}$。
- Reasoning Feedback $\mathbf{F}^c$：描述 [CLS] 向量被预测头使用来产生类别 $c$ 预测的过程。对应 $\frac{\partial \mathcal{L}^c(X)}{\partial \mathbf{Z}_{\text{CLS}}^{(L)}}$。

最终的归因表达式为二者的 Hadamard 乘积：

$$\mathcal{T} = \left(\mathbf{P}^{(L)} \odot \mathbf{F}^c\right)_{\mathcal{P}}$$

下标 $\mathcal{P}$ 由池化策略决定——[CLS] 池化取第 0 行对应的第 1 到 $N$ 列，全局池化则对所有 token 行求和。

### Attention Perception：递推注意力图

这是本文的核心技术贡献。目标是建立第 $l$ 层 attention block 输入 $\mathbf{Z}^{(l-1)}$ 和输出 $\mathbf{Z}^{(l)}$ 之间的关系。

**Approximation 1**（线性化）：忽略 LayerNorm 和 ReLU/GeLU 等非线性成分对 token 交互的影响（它们主要作用在单个 token 维度上），将 attention block 近似为线性变换：

$$\mathbf{Z}^{(l)} \approx (\mathbf{A}^{(l)} \mathbf{Z}^{(l-1)} W^{(l)} + \mathbf{Z}^{(l-1)}) W_{\text{MLP}}^{(l)}$$

其中 $W^{(l)} = W_v^{(l)} W_{\text{proj}}^{(l)}$，$W_{\text{MLP}}^{(l)}$ 是两层 MLP 的合并权重。注意这里保留了 $W^{(l)}$（即线性投影矩阵），而不是像 Rollout 那样假设它等于单位阵 $I$。

**Approximation 2**（递推展开）：定义修正后的注意力图 $\bar{\mathbf{A}}^{(l)}$，使得 $\mathbf{A}^{(l)} \mathbf{Z}^{(l-1)} W^{(l)} + \mathbf{Z}^{(l-1)} \approx (\bar{\mathbf{A}}^{(l)} + I) \mathbf{Z}^{(l-1)}$。从最后一层递推到第一层，得到：

$$\mathbf{P}^{(L)} = (\bar{\mathbf{A}}^{(L)} + I) \cdot \mathbf{P}^{(L-1)} = (\bar{\mathbf{A}}^{(L)} + I)(\bar{\mathbf{A}}^{(L-1)} + I) \cdots (\bar{\mathbf{A}}^{(1)} + I)$$

这个递推形式与 Rollout 的 $(\mathbf{A}^{(L)} + I) \cdots (\mathbf{A}^{(1)} + I)$ 结构相同，但用 $\bar{\mathbf{A}}^{(l)}$ 替换了原始注意力 $\mathbf{A}^{(l)}$。论文给出了 $\bar{\mathbf{A}}^{(l)}$ 的两种计算方式：

**Token-wise 注意力图**：这个近似要解决的核心矛盾是：attention block 的输出包含两条路径——attention 路径经过了线性投影 $W^{(l)}$，residual 路径没有。要做逐层递推，需要把整个式子写成 $(\text{某个矩阵}) \cdot \mathbf{Z}^{(l-1)}$ 的形式。Residual 那项天然满足（就是 $I \cdot \mathbf{Z}^{(l-1)}$），但 attention 路径因为右乘了 $W^{(l)}$，没法直接提取出一个左乘的系数矩阵。

Rollout 的做法是假设 $W^{(l)} = I$，直接丢掉这个投影。Token-wise 的做法是：既然 $W^{(l)}$ 把每个 token 向量 $\mathbf{Z}_j^{(l-1)}$ 变换成了 $\mathbf{Z}_j^{(l-1)} W^{(l)}$，方向变了、长度也变了，那至少把长度变化这个信息保留下来——用范数比 $\frac{\|\mathbf{Z}_j^{(l-1)} W^{(l)}\|}{\|\mathbf{Z}_j^{(l-1)}\|}$ 去缩放对应的注意力权重：

$$(\bar{\mathbf{A}}_{\text{token}}^{(l)})_{i,j} = \mathbf{A}_{i,j}^{(l)} \cdot \frac{\|(\mathbf{Z}^{(l-1)} W^{(l)})_{j,\cdot}\|}{\|\mathbf{Z}_{j,\cdot}^{(l-1)}\|}$$

直觉上：token $i$ 关注 token $j$ 时，原始注意力权重是 $\mathbf{A}_{i,j}^{(l)}$，但 $j$ 经过投影后可能被放大或缩小了，实际贡献不等于注意力权重暗示的那么多。Token-wise 把这个放大/缩小系数乘回注意力权重里，使得修正后的 $\bar{\mathbf{A}}^{(l)}$ 能更准确地反映每个 token 的实际贡献量。代价是丢掉了 $W^{(l)}$ 带来的方向变化信息（投影不只改变长度，还会旋转向量方向），这是近似误差的来源。但实验表明这个误差比直接令 $W^{(l)}=I$ 小了约 $20\times$，说明在实践中，投影矩阵对归因的影响主要通过缩放各 token 的"信号强度"来体现，方向变化的影响相对次要。

**Head-wise 注意力图**：将各 head 按重要性加权求和。Head 重要性 $I_h$ 通过注意力与其梯度的内积来衡量，最终以归一化权重 $\theta_h$ 合并各 head 的注意力：

$$\bar{\mathbf{A}}_{\text{head}}^{(l)} = \sum_{h=1}^{H} \theta_h \mathbf{A}_h^{(l)}$$

### 近似误差对比

![Approximation Test](/assets/images/token-attribution-transformer/fig3.png)
*Figure 3: 在 ViT-Base 各层上，token-wise 近似与 Attention Rollout 的逐层误差对比（log10 尺度）。随着层数增加，Rollout 的误差快速增长（从 $10^5$ 到 $10^{8.5}$），而 token-wise 方法的误差基本保持在 $10^5$ 量级。*

在 ImageNet 验证集上随机选取 5000 张图片，逐层比较近似值与真实输出 $\mathbf{Z}_{\text{true}}^{(l)}$ 的平方误差。Token-wise 方法比 Attention Rollout 平均准确 $20\times$，且误差方差随层数增加保持稳定。这验证了保留线性投影矩阵 $W^{(l)}$ 的必要性。

### Reasoning Feedback：集成梯度降噪

直接计算 $\frac{\partial \mathcal{L}^c(X)}{\partial \mathbf{A}^{(L)}}$ 可能包含对无关 token 的噪声梯度。论文采用 Integrated Gradient（Sundararajan et al., 2017），对最后一层注意力图的梯度沿从全零基线到输入的线性路径做积分：

$$\mathbf{F}^c = \text{ReLU}\left(\frac{1}{K} \sum_{k=1}^{K} \frac{\partial \mathcal{L}^c(\frac{k}{K} X)}{\partial \mathbf{A}^{(L)}}\right) \xrightarrow{K \to \infty} \text{ReLU}\left(\int_{\alpha=0}^{1} \frac{\partial \mathcal{L}^c(\alpha X)}{\partial \mathbf{A}^{(L)}} d\alpha\right)$$

ReLU 只保留正梯度部分，过滤掉抑制性的归因信号。

## 实验验证

实验覆盖了三个维度：不同模态（BERT/ViT/CLIP）、不同模型大小（ViT-Base/Large）、不同池化策略（[CLS]/全局池化）。评估指标包括扰动测试（perturbation test）、分割测试（segmentation test）和语言推理测试（language reasoning test）。

### 扰动测试（Perturbation Test）

按归因分数从高到低（正向）或从低到高（负向）逐步遮盖 token，观察分类准确率的变化。正向扰动的 AUC 越低、负向 AUC 越高，说明归因越准确。

| 模型 | 方向 | 类别 | RAM | Rollout | CAM | PLRP | GA | TA | Ours-H | Ours-T |
|:---|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| ViT-Base | Pos. | Pred. | 24.51 | 20.45 | 34.61 | 20.18 | 20.95 | 17.37 | 15.97 | **15.89** |
| ViT-Base | Neg. | Pred. | 45.65 | 53.86 | 41.95 | 50.62 | 48.95 | 54.53 | 57.13 | **57.97** |
| ViT-Large | Pos. | Pred. | 27.71 | 21.91 | 45.66 | 30.19 | 19.21 | 20.41 | 18.01 | **17.99** |
| ViT-Large | Neg. | Pred. | 40.99 | 53.44 | 47.58 | 37.14 | 54.72 | 52.67 | 55.86 | **56.44** |
| ViT-MAE | Pos. | Pred. | 38.56 | 38.20 | 56.79 | 26.59 | 33.11 | 34.00 | **20.57** | 20.72 |
| ViT-MAE | Neg. | Pred. | 40.79 | 52.03 | 24.80 | 55.34 | 57.67 | 56.92 | **65.33** | 64.48 |

在 ViT-Base 上，本文方法与 TA（Transformer Attribution）差距不大。但在 ViT-Large（24 层、1024 维）和 ViT-MAE（全局池化）上，优势明显扩大。特别是 ViT-MAE 的正向扰动，Ours-H 的 20.57 比 GA 的 33.11 低了近 13 个百分点。这与前面的理论分析一致：模型越深、结构越不同于标准 ViT-Base，直觉式方法的误差累积越严重。

### 分割测试（Segmentation Test）

将归因结果与 ImageNet-Segmentation 的语义分割 ground truth 比较，评估像素级/mIoU/mAP/mF1。

| 模型 | mIoU | Pixel Acc. | mAP | mF1 |
|:---|:---:|:---:|:---:|:---:|
| ViT-Base (Ours-T) | **66.32** | **82.15** | **88.04** | **45.72** |
| ViT-Base (TA) | 61.92 | 79.68 | 85.99 | 40.10 |
| ViT-Large (Ours-T) | **61.65** | **78.92** | **86.33** | **43.38** |
| ViT-MAE (Ours-T) | 62.36 | **79.63** | 86.21 | **44.08** |

Token-wise 方法在所有设置下均为最优。mIoU 提升幅度从 ViT-Base 的 +4.4 到 ViT-Large 的 +5.3（相比 TA），说明更精确的逐层近似在语义定位上有实际价值。

### 不同模型变体的可视化

![不同 ViT 变体在同一张图上的可视化对比](/assets/images/token-attribution-transformer/fig8.png)
*Figure 8: 同一张"tiger cat"图片，在 ViT-Large（第 1 行）、ViT-Base（第 2 行）、ViT-MAE（第 3 行）上的可视化。GradCAM、GA、TA 等方法在 ViT-Large 上出现明显的语义偏移，本文方法保持稳定。*

这张对比图能直观说明误差累积的后果。在 ViT-Base 上，GradCAM 和 GA 尚能捕捉猫的轮廓；但在 ViT-Large 上，它们的热力图偏移到了无关区域。本文方法在三种架构上都能稳定地聚焦在猫的身体区域。

### 消融实验

| Perception | Feedback | mIoU | mAP | mF1 | Pos. | Neg. |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| $\times$ | $\int \mathbf{G}^{(L)}$ | 48.76 | 77.93 | 38.44 | 18.56 | 52.12 |
| Head-wise | $\times$ | 53.24 | 83.66 | 42.26 | 20.20 | 52.79 |
| Head-wise | $\mathbf{G}^{(L)}$ | 57.75 | 84.49 | 43.01 | 17.33 | 55.27 |
| Head-wise | $\int \mathbf{G}^{(L)}$ | 60.74 | 86.18 | 44.45 | 15.97 | 57.13 |
| Token-wise | $\int \mathbf{G}^{(L)}$ | **66.32** | **88.04** | **45.72** | **15.89** | **57.97** |

从消融结果看：（1）Attention Perception 和 Reasoning Feedback 两部分各自都有独立贡献；（2）集成梯度（$\int \mathbf{G}^{(L)}$）比直接梯度（$\mathbf{G}^{(L)}$）好约 3 个 mIoU 点，说明降噪确实有效；（3）Token-wise 近似比 Head-wise 好约 5.6 个 mIoU 点，对应前面 $20\times$ 近似误差优势在下游指标上的体现。

## 思考与讨论

**Rollout 是本文框架的特例。** 当令 $W^{(l)} = I$（即忽略线性投影），本文的递推关系退化为 $\bar{\mathbf{A}}^{(l)} = \mathbf{A}^{(l)} + I$，正好就是 Attention Rollout。这提供了一个统一视角：Rollout 并非"错误"的方法，而是本文框架在最强近似假设下的退化形式。当这个假设不成立时（深层模型、大投影矩阵），误差自然放大。

**Token-wise vs. Head-wise 的取舍。** 两种方法的量化差异不大（扰动测试差 <1%），都优于所有 baseline。Token-wise 在分割测试上更好，但实现时需要提取中间层的 $\mathbf{Z}^{(l-1)} W^{(l)}$，这些不是模型的标准输出，需要额外的 hook。Head-wise 只需要注意力权重和梯度，对现有推理框架的侵入性更小。在实际工程中，如果对精度要求不高，head-wise 可能是更实用的选择。

**方法的适用边界。** 论文只处理了 Transformer Encoder 结构。对于 Encoder-Decoder（如机器翻译）或 Decoder-only（如 GPT 系列）的归因问题，cross-attention 和自回归 mask 会改变注意力的传播方式，递推关系需要重新推导。这是作者在结论中提到的未来方向。
