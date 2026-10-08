---
layout: post
title: "【论文笔记】Agentic RL 为什么把 critic 请回来了：SAO 与 CompactionRL"
date: 2026-08-14
categories: [paper-notes]
tags: [agentic-RL, PPO, GRPO, value-model, context-compaction]
paper_title: "Single-Rollout Asynchronous Optimization for Agentic Reinforcement Learning / CompactionRL: Reinforcement Learning with Context Compaction for Long-Horizon Agents"
paper_authors: "Zhenyu Hou, Yujiang Li et al."
paper_link: "https://arxiv.org/abs/2607.07508"
---

> **论文一**：[Single-Rollout Asynchronous Optimization for Agentic Reinforcement Learning](https://arxiv.org/abs/2607.07508)
> **作者**：Zhenyu Hou\*, Yujiang Li\*, Jie Tang, Yuxiao Dong（\* 共同一作）
> **机构**：清华大学；部分工作在 Z.AI 实习期间完成
> **发表**：arXiv:2607.07508v1，2026-07-08

> **论文二**：[CompactionRL: Reinforcement Learning with Context Compaction for Long-Horizon Agents](https://arxiv.org/abs/2607.05378)
> **作者**：Yujiang Li\*, Zhenyu Hou\*, Yi Jing\*, Jie Tang, Yuxiao Dong（\* 共同一作）
> **机构**：清华大学；部分工作在 Z.AI 实习期间完成
> **发表**：arXiv:2607.05378v1，2026-07-06

## 两篇论文，同一个动作

这两篇论文相隔两天挂上 arXiv，作者几乎是同一批人，结尾都写着"已部署到 GLM-5.2（750B-A40B）的 agentic RL pipeline"。放在一起读，会发现它们在做同一件事的两个侧面：**把 GRPO 的 group 结构拆掉，换回带 value model 的 PPO**。

这个方向和过去两年的主流是反的。自 DeepSeekMath 提出 GRPO 之后，"扔掉 critic"几乎成了 LLM RL 的默认选择：一个 prompt 采 $$G$$ 条回答，用组内奖励均值当 baseline，省下一个和 policy 同等规模的 value model，同时避开了 value 学习本身的不稳定。数学推理、代码这类可验证任务上，这条路线的确好用。

但两篇论文各自遇到一个 GRPO 结构上装不下的场景：

- **SAO** 的场景是**异步训练**。异步 RL 的卖点是 rollout 一生成完就立刻喂给训练，不等整批。可 GRPO 的 group 必须凑齐 $$G$$ 条才能算组内均值——这等于在异步系统里手动加回一道同步屏障，而且要等的是组里最慢的那条。agentic 任务的轨迹长度差异极大，这道屏障的代价特别高。
- **CompactionRL** 的场景是**上下文压缩**。当 agent 的历史撑满上下文窗口时，把前面的历史总结成一段摘要、再从摘要继续跑。这样一条 rollout 会被切成若干段，段数还是变量。GRPO 假设"一个 prompt 对应固定 $$G$$ 个样本"，在这里根本不成立。

两篇都得出同一个结论：既然 group baseline 用不了，就得有一个 state-dependent 的 baseline，也就是 critic。于是问题从"如何避免训练 value model"变成了"如何把 value model 训好"。这两篇论文的技术含量，大部分落在后面这个问题上。

下面先补一点 PPO / GRPO 的分工，再分别看两篇。

## 预备：min 里面装了什么

两篇论文的公式都从同一个 clipped surrogate 出发：

$$\mathbb{E}\left[\frac{1}{|y|}\sum_{t=1}^{|y|}\min\left(r_{t}(\theta)\hat{A}_{t},\ \text{clip}(r_{t}(\theta),1-\epsilon,1+\epsilon)\hat{A}_{t}\right)\right]$$

其中 $$r_{t}(\theta) = \frac{\pi_\theta(y_{t} \mid q, y_{<t})}{\pi_{\theta_\text{old}}(y_{t} \mid q, y_{<t})}$$ 是当前策略与采样时策略在第 $$t$$ 个 token 上的概率比，$$\hat{A}_{t}$$ 是优势估计，$$\epsilon$$ 控制信任区域的宽度。直觉是：优势为正的 token 想提高概率，但提高得太多（$$r_{t}$$ 超过 $$1+\epsilon$$）就把梯度截断，防止一步跨太远。

PPO 与 GRPO 的分界只在 $$\hat{A}_{t}$$ 怎么来。PPO 训一个 value 网络 $$V_\phi$$，用 GAE 算 token 级优势：

$$\hat{A}_{t}^{\text{GAE}}=\sum_{l=0}^{|y|-t-1}(\gamma\lambda)^{l}\delta_{t+l},\qquad \delta_{t}=r_{t}+\gamma V_{\phi}(s_{t+1})-V_{\phi}(s_{t})$$

$$\delta_{t}$$ 是单步的 TD 残差——"实际拿到的即时奖励加上后一状态的估值，比当前状态的估值高出多少"。GAE 把这些残差按 $$(\gamma\lambda)^l$$ 指数加权累加，$$\lambda$$ 越接近 1 越像蒙特卡洛回报（偏差小、方差大），越接近 0 越依赖 $$V_\phi$$（偏差大、方差小）。代价是要额外维护一份和 policy 同规模的参数，显存翻倍。

GRPO 把这一整套换成一行：同一个 prompt 采 $$G$$ 条，用组内奖励的均值（和标准差）归一化，整条序列共享一个标量优势。省掉了 $$V_\phi$$，但也失去了 token 级的信号，并且引入了"必须有一组样本"的结构性依赖。

这个依赖就是两篇论文各自撞上的墙。

## 论文一：SAO —— 异步训练下的单条 rollout

![SAO 与 GRPO 的数据流对比](/assets/images/glm-agentic-rl/sao-overview.png)
*Figure 2：上半是 GRPO，下半是 SAO。方框里的数字是轨迹生成完成的顺序。GRPO 那一行里，1、2、7、9 号轨迹属于同一个 group（蓝色），必须等到 9 号也跑完才能进 Training，此时 3、4、5、6、8（灰色）已经跑完在旁边干等——"waiting for Group"。SAO 那一行里，9、8、…、3、2、1 按完成顺序逐条进 Training，不存在等待。右侧两张小图是信任区域示意：横轴是优势 $$A$$，纵轴是 $$\pi_\theta/\pi_\text{rollout}$$，红色虚线标出 $$1+\epsilon_{h}$$ 和 $$1-\epsilon_{l}$$，落在带外的 token 直接丢掉。*

### 异步带来的两个麻烦

同步 RL 的流程是：用一个固定的策略快照采满一批 rollout，然后在这批数据上做若干轮优化。agentic 和 coding 任务的轨迹长度分布是长尾的，短的几轮结束，长的能跑几百轮工具调用，于是整个集群大部分时间在等最慢的那条轨迹。异步 RL 让生成和训练并行，rollout 到达即消费。

代价是两个：

**第一，policy lag 变得不可控。** 一条轨迹在生成过程中，rollout engine 可能已经被更新过好几次——轨迹的前半段来自 $$\pi_{\theta_\text{old}}^{(1)}$$，后半段来自 $$\pi_{\theta_\text{old}}^{(3)}$$。标准的 decoupled PPO 要维护三个策略：当前策略 $$\pi_\theta$$、采样时的旧策略 $$\pi_{\theta_\text{old}}$$、以及推理引擎实际用的 $$\pi_\text{rollout}$$，用 $$\frac{\pi_\theta}{\pi_{\theta_\text{old}}}$$ 修正 staleness，用 $$\frac{\pi_{\theta_\text{old}}}{\pi_\text{rollout}}$$ 修正训练框架与推理框架的数值差异。异步下要精确记住每个 token 是哪个版本生成的，就得留一整串 checkpoint $$\{\pi_{\theta_\text{old}}^{(1)}, \dots, \pi_{\theta_\text{old}}^{(N)}\}$$，工程上不现实。

**第二，group 采样和异步互相打架。** 除了前面说的同步屏障，还有一层更根本的不兼容：在线学习或者复杂 agentic 环境里，环境对一个 prompt 往往只给一条轨迹的反馈，凑不出 group。

### DIS：不追旧策略，只设一条硬边界

SAO 对第一个麻烦的处理相当直接——**不追了**。直接把 $$\pi_\text{rollout}$$ 当作行为策略，比值就是

$$r_{t}(\theta)=\exp\left(\log\pi_{\theta}(a_{t}|s_{t})-\log\pi_{\text{rollout}}(a_{t}|s_{t})\right)$$

分母的 log 概率在生成阶段由推理引擎顺手输出，不需要额外做一次前向。$$\pi_{\theta_\text{old}}$$ 整个被丢掉。论文的理由是：反正那个"最新的旧策略"本身就是错的近似（轨迹的不同部分来自不同版本），用一个错的近似不如干脆不用，换来的是省掉一次 old-policy 推理。

丢掉修正项当然会放大 off-policy 偏差，所以第二步是收紧信任区域。标准 PPO 的 clip 只在"优势为正且 $$r_{t} > 1+\epsilon_{h}$$"或"优势为负且 $$r_{t} < 1-\epsilon_{l}$$"时才起作用，另外两种组合是放行的。SAO 不看优势符号，只要 $$r_{t}$$ 落在 $$[1-\epsilon_\ell, 1+\epsilon_{h}]$$ 之外就把这个 token 从梯度里**完全抹掉**（mask，不是截断到边界值）：

$$f(x;\epsilon_{\ell},\epsilon_{h})=\begin{cases}x,&\text{if }1-\epsilon_{\ell}<x<1+\epsilon_{h}\\ 0,&\text{otherwise}\end{cases}$$

目标函数写成

$$L(\theta)=\hat{\mathbb{E}}_{t}\left[f(r_{t}(\theta),\epsilon_{l},\epsilon_{h})\hat{A}_{t}\log\pi_{\theta}(a_{t}|s_{t})\right]$$

注意这个形式和 PPO 的 $$\min(\cdot,\cdot)$$ 不一样：没有取小的悲观项，$$f(r_{t})$$ 就是一个乘在 $$\hat{A}_{t} \log \pi_\theta$$ 前面的系数，带外为 0、带内等于 $$r_{t}$$。本质上是"带硬门控的重要性加权 REINFORCE"。论文说这与 IcePop 机制相似，区别是进一步去掉了 $$\pi_{\theta_\text{old}}$$。

作者把这套叫 DIS（Direct double-sided Importance Sampling）。实际超参在 TIR（数学 + Python 工具）任务上是 $$\epsilon_\text{low}=0.3$$、$$\epsilon_\text{high}=5.0$$，即 $$r_{t}$$ 低于 0.7 或高于 6.0 的 token 被丢弃；coding 任务上是 $$\epsilon_\text{low}=0.8$$、$$\epsilon_\text{high}=3.0$$，即区间 $$[0.2, 4.0]$$。可以看出这个"双侧"极不对称：真正卡得紧的是下界，也就是"当前策略认为这个 token 比采样时更不可能"的情况。

![Clip ratio 对比](/assets/images/glm-agentic-rl/sao-clip-ratio.png)
*Figure 4(c)：横轴训练步数，纵轴被 clip / mask 掉的 token 比例。紫色是 SAO（带 DIS），从 0.001 起伏后在中后段冲到 0.006 又回落；蓝色是 vanilla VAPO（不带 DIS），几乎贴着 0。作者的解读是：VAPO 的 clip ratio 接近零说明它其实没有在拦任何东西，发散的 off-policy 更新被原样放进梯度，训练在约 90 步崩掉。注意纵轴量级——即便 SAO 的峰值也只有 0.6% 的 token 被丢，这个"激进"是相对而言的。*

### 单条 rollout：把方差问题转交给 value model

处理完稳定性，第二个麻烦是 group。SAO 的做法是把 group size 直接设成 1——**一个 prompt 只采一条轨迹，生成完立刻进训练**。这样"等最慢的"这个问题从根上消失了，同时也天然兼容那些只能给单条反馈的环境。

代价很清楚：单条 rollout 没有组内均值可以当 baseline，梯度估计的方差会显著上升，退化成 REINFORCE 的处境。降方差就得靠一个足够准的 $$V_\phi$$。论文 3.2 节的四个设计全部围绕这一点：

**（1）critic 更新比 policy 更频繁。** 作者定位单条 rollout RL 的主要不稳定来源是 policy 和 value 的互相依赖：$$V_\phi$$ 不准 → $$\hat{A}_{t}$$ 有噪声 → policy 更新出错 → 分布漂移 → $$V_\phi$$ 更不准。解法是解耦两者的更新频率，policy 每更新一次，value 更新 $$K$$ 次，实验里 $$K=2$$。让 value 先追上当前 policy，再用它算优势。

**（2）value model 冻结 attention。** 前期实验里作者发现 value model 的梯度范数明显大于 policy，进一步分解后定位到 full attention 层，MoE 层相对稳定。于是 RL 阶段冻住 $$V_\phi$$ 的 attention 参数，只训 MoE 投影。论文给的假设是预训练的 attention 权重已经具备足够的语义定位能力，只训 MoE 相当于一种正则。

**（3）Skip-Observation token 级 GAE。** 这一条是专门为 agentic 轨迹设计的。agentic 轨迹的结构是 $$T=[a_{0}, o_{0}, a_{1}, o_{1}, \dots]$$，$$a_{i}$$ 是模型生成的动作，$$o_{i}$$ 是环境返回的观察。标准 GAE 逐 token 算相邻状态的价值差，但从动作末尾 $$a_{i,\text{end}}$$ 跨到观察开头 $$o_{i,\text{start}}$$ 这一步是"断裂"的——$$o_{i}$$ 不是模型生成的，让 $$V_\phi$$ 去预测一个外部环境状态的价值，得到的只是噪声。

SAO 改写 Bellman target，直接跳过观察 token，把当前动作的最后一个 token 接到下一个动作的第一个 token：

$$\hat{A}(a_{i,N})=\delta+\gamma\lambda\hat{A}(a_{i+1,0}),\qquad \delta=r_{t}+\gamma V(a_{i+1,0})-V(a_{i,N})$$

其中 $$a_{i,N}$$ 是第 $$i$$ 个动作的末 token，$$a_{i+1,0}$$ 是下一个动作的首 token。这样优势估计只依赖模型自己的输出，环境反馈的随机性被过滤掉。

**（4）扩大 value pretraining 的数据量。** 作者说 value 估计的"冷启动"是一个主要瓶颈，显著加大 value 预训练语料的规模才能让前面几条机制从训练早期就生效。这一条论文只给了定性说明，没有给数据量的消融曲线。

另外还有一个容易漏掉的超参：SAO 用的是 length-adaptive GAE，$$\lambda_\text{policy}=1-\frac{1}{\alpha l}$$、$$\alpha=1.5$$，$$l$$ 是回答长度——轨迹越长 $$\lambda$$ 越接近 1。critic 侧则固定 $$\lambda_\text{critic}=1$$。

### 实验：能稳定训一千步

训练设置：数学 + Python 工具（TIR）任务上，先用 GPT-OSS-120B 生成的 TIR 数据把 Qwen3-30B-A3B-Thinking-2507 微调 3 个 epoch，用这个 SFT 模型同时初始化 policy 和 value；coding 任务直接从 Qwen3-30B-A3B-Thinking-2507 开始。batch size 128，group size 1，最大长度 128k。GRPO 对照组是 16 个 prompt × 8 条 rollout = 128，总 batch size 对齐。SWE-Bench Verified 用 OpenHands 作 scaffold，最多 300 轮交互。

![SAO 主结果柱状图](/assets/images/glm-agentic-rl/sao-main-bar.png)
*Figure 1：五个 benchmark 上 Baseline（白）、GRPO（浅蓝）、SAO（深蓝）的对比。前四个是数学推理（带 Python 工具），baseline 是 Qwen3-30B-A3B 的 SFT 模型；SWE-Bench Verified 是 coding，baseline 是 Qwen3-30B-A3B。SAO 在五个上都最高，数学四项的提升幅度（AIME 80.4→97.3、BeyondAIME 53.3→74.8）远大于 coding（23.0→29.8）。*

数学推理的完整结果（Pass@1 准确率 %，AIME/HMMT/IMOAnswerBench 取 16 次评测均值，BeyondAIME 取 4 次）：

| 模型 | AIME2025 | BeyondAIME | HMMT Nov 2025 | IMOAnswerBench |
|---|---|---|---|---|
| Claude-Sonnet-4.5 | 87.0 | 62.0 | 81.7 | 65.8 |
| GPT-5 High | 94.6 | 74.0 | 89.2 | 76.0 |
| GLM-4.7 | 95.7 | – | 93.5 | 82.0 |
| Qwen3-30B-A3B（w/ python） | 14.6 | 10.5 | 17.3 | 7.8 |
| Qwen3-30B-A3B（w/o python） | 85.0 | 63.0 | 76.7 | 55.3 |
| SFT（w/ python） | 80.4 | 53.3 | 75.2 | 53.3 |
| GRPO（w/ python） | 84.2 | 54.8 | 76.0 | 55.8 |
| **SAO** | **97.3** | **74.8** | **88.3** | **74.0** |
| – SAO（仅 DIS） | 94.2 | 71.5 | 86.7 | 71.3 |
| – GRPO（+ DIS） | 93.5 | 70.8 | 84.0 | 70.0 |

这张表有几处值得留意。原始 Qwen3-30B-A3B 带 Python 工具时只有 14.6，不带工具反而 85.0——说明它本来不会用工具，工具调用反而干扰了推理，SFT 阶段（80.4）主要是在补这个能力。所以从 SFT 的 80.4 到 SAO 的 97.3 这 +16.9 分，是在一个刚学会用工具的起点上做 RL 拿到的。

表格下半部分的三行对照更能说明各部分的贡献：GRPO + DIS 拿到 93.5，SAO 只用 DIS（应该理解为 PPO + DIS + 单条 rollout）拿到 94.2，完整 SAO 拿到 97.3。也就是说 DIS 这一项把 GRPO 从 84.2 拉到 93.5（+9.3，主要是让它不崩），而"单条 rollout + 完整 value 设计"再贡献约 +3.8。

SWE-Bench Verified 上的差距小得多：

| 模型 | Accuracy (%) |
|---|---|
| Qwen3-30B-A3B | 23.0 |
| + GRPO（w/ DIS） | 27.0 |
| + SAO | 29.8 |

![训练曲线 AIME](/assets/images/glm-agentic-rl/sao-curve-aime.png)
![训练曲线 BeyondAIME](/assets/images/glm-agentic-rl/sao-curve-beyondaime.png)
*Figure 3（两个 panel）：横轴训练步数（0–1000），纵轴评测准确率。紫色 SAO、深蓝 GRPO (w/ DIS)、浅蓝 Vanilla GRPO。浅蓝线在 160 步左右垂直坠落出图——这就是论文说的 vanilla GRPO 崩溃，表格里报的是它崩溃前的最后有效分数。紫色和深蓝在前 400 步基本重叠，之后开始分叉，到 900 步附近 SAO 稳定高出 GRPO (w/ DIS) 约 2 分。*

这张图是全文最关键的证据，也是解读上最需要小心的地方：**DIS 负责"不崩"，单条 rollout 负责"后期还能涨"**。400 步之前两者没有区别，说明单条 rollout 的收益不是立刻显现的，而是在 value model 逐渐训准之后才兑现。反过来说，如果只训几百步，这套复杂的 value 工程收益有限。

### 消融：每一条都在贡献，但幅度不均

| 配置 | AIME2025 | BeyondAIME |
|---|---|---|
| SAO | 97.3 | 74.8 |
| SAO w/o Faster value（critic 每批只更新 1 次） | 95.0 | 69.8 |
| SAO w/o Frozen attention（value 全参数训练） | 90.6 | 74.5 |
| Vanilla VAPO（不带 DIS） | 91.3 | 69.0 |
| Running mean baseline | 79.8 | 55.3 |

两个观察：

一是 **Running mean baseline 掉得最狠**（79.8 / 55.3）。这个 baseline 的做法是为每个 prompt 维护最近 8 次奖励的滑动窗口，用均值当 baseline——一个不需要参数化 value model 的廉价替代。它比 SFT 起点（80.4 / 53.3）几乎没有提升，作者用这个结果论证"训好的 value model 是必要的"。这是全篇最有说服力的对照。

二是**冻结 attention 的效果在两个 benchmark 上方向不一致**：去掉它 AIME 从 97.3 掉到 90.6（-6.7），但 BeyondAIME 只从 74.8 掉到 74.5（-0.3，基本在噪声内）。同理 faster value 在 BeyondAIME 上掉 5.0 而 AIME 上掉 2.3。两个设计的收益并不像论文正文表述得那样一致。

![Explained Variance](/assets/images/glm-agentic-rl/sao-explained-variance.png)
*Figure 4(a)：纵轴是 explained variance $$EV=1-\frac{\text{Var}(R-V(s))}{\text{Var}(R)}$$，衡量 $$V_\phi$$ 的预测和真实回报的吻合程度，1 表示完美。紫色是 SAO（critic 每批更新 2 次），蓝色是只更新 1 次。前 400 步两条线缠在一起（0.22 上升到 0.42），400 步之后紫色明显走高，880 步时约 0.60 对 0.52。这个分叉点和 Figure 3 里 SAO 与 GRPO 的分叉点是同一个位置，支持了"后期收益来自 value 变准"的解释。*

![Critic 梯度范数](/assets/images/glm-agentic-rl/sao-critic-gradnorm.png)
*Figure 4(b)：纵轴 critic 梯度范数。紫色是全参数训练 value model，从 5.6 一路爬到 10.5；蓝色是 SAO 的冻结 attention，稳定在 3–5 之间。这张图直接支撑了"不稳定来自 attention 层"的定位。*

### 在线学习模拟：写作风格漂移

这一节是 SAO 最有想象力的实验，也是最能体现"单条 rollout"独特价值的地方——**在线环境里通常没有 group 可采**。

任务设计：一个写作任务，奖励信号是"用户偏好的语言风格"，训练中途换两次偏好，依次偏好 cute（可爱）、chuunibyou（中二）、classical（古典）。奖励用 GLM-4.7 当 judge，$$r = r_\text{quality} \times r_\text{style}$$，两项都是 0/1，所以最终奖励也是 0/1——质量和风格都达标才给分。系统提示要求模型从四个候选风格里选一个，前两阶段候选是 {Academic, Cute, Chuunibyou}，最后一阶段换成 {Classical, Cute, Chuunibyou}。

![在线学习风格切换](/assets/images/glm-agentic-rl/sao-online-styles.png)
*Figure 5(a)：横轴训练步数（0–420），纵轴是留出测试集上四种风格各自的出现准确率。灰色竖条标出偏好切换的时刻（约 155–185 步、290–320 步）。蓝色 Academic 从 28% 一路降到接近 0（它从来不是奖励目标）；青色 Cute 在第一阶段爬到 68%，切换后垂直坠到 0；紫色 Chuunibyou 在第二阶段接棒冲到 78%，第二次切换后同样坠落；橙色 Classical 在最后阶段从 3% 涨到 66%。三次接力都发生在灰条附近的几十步内。*

![在线学习奖励曲线](/assets/images/glm-agentic-rl/sao-online-reward.png)
*Figure 5(b)：横轴训练步数，纵轴训练奖励。深蓝 SAO、青色 Running mean（窗口 128 的历史奖励均值当 baseline）。第一阶段 Running mean 反而更快（0.74 对 0.60）；但两次风格切换后，SAO 的恢复明显更快更高——第二阶段 SAO 回到 0.73 而 Running mean 只到 0.52，第三阶段 SAO 0.69 对 0.59。*

作者的解释是：Running mean 的滑动窗口有惯性，切换后窗口里还塞着旧分布的奖励，baseline 暂时是偏的；而 $$V_\phi$$ 是 state-dependent 的，能跟着输入状态动态调整。这个论证在逻辑上说得通，不过第一阶段 Running mean 领先 0.14 这件事论文没有解释——静态环境下这个廉价 baseline 收敛更快，只有在分布切换时才吃亏。

### 附录里被否掉的一条路：step-level value

附录 A.1 试了另一种降方差思路：既然 token 级 value 方差大，那把一整个对话轮次（step）当作一个动作，一个 step 共享一个优势。step value 有两种取法——所有 token value 的平均（Step Average），或者只取最后一个 token（Last-Token，理由是末 token 汇聚了整个 step 的信息）。同时 length-adaptive GAE 的 $$\lambda$$ 也改成按 step 数而非 token 数缩放：$$\lambda_\text{policy}=1-\frac{1}{\alpha \cdot \text{step number}}$$。

| 粒度（400 步对齐） | AIME2025 | BeyondAIME |
|---|---|---|
| Step-level（Average） | 85.8 | 60.5 |
| Step-level（Last-Token） | 87.3 | 62.8 |
| Token-level | 89.8 | 66.8 |

![Step vs token 训练奖励](/assets/images/glm-agentic-rl/sao-step-vs-token.png)
*Figure 6：横轴训练步数（0–400），纵轴训练奖励。浅蓝 SAO（token 级）在约 100 步后拉开，最终到 0.54；紫色 Step-level(Average) 与深蓝 Step-level(Last-Token) 纠缠在 0.49 附近。*

结论是 token 级更好，作者归因于更细粒度的监督信号对捕捉推理轨迹里的逻辑转折是必要的。这条否定结果值得记下来——它和第二篇论文的做法形成对照：CompactionRL 也用 token 级，且把 token 级 loss 归一化列为最关键的组件。

## 论文二：CompactionRL —— 把"压缩"变成可训练的动作

### 问题：上下文窗口是长程 agent 的硬墙

SWE-bench、Terminal-Bench 这类任务，agent 要反复推理、调用工具、读环境反馈、修改计划。累积的历史（工具输出、中间推理、报错信息、半成品补丁）会撑满上下文窗口。加长上下文能缓解但不解决：成本高，而且长序列上的有效利用率会下降。

上下文压缩（context compaction）是现成的解法，Claude Code 这类产品里已经是标配：历史快满时，把前面的内容总结成一段摘要，然后从"摘要 + 最近几轮"重新开始。问题是这件事以前只被当作**推理期的启发式**或者外部记忆操作。

而在 RL 训练里，压缩的地位要更根本：**摘要一旦替换掉原始历史，它就决定了后续所有动作能看到什么信息**。任务能不能做成，不只取决于执行策略，还取决于压缩策略。

![有无压缩的对比](/assets/images/glm-agentic-rl/crl-compaction-illustration.png)
*Figure 1（左）：上半"Inference w/o compaction"——prompt 后面 step 1、2、…、N 塞进一个 context window，撞到 "Context budget exhausted" 的红线就 Stopped。下半"Inference with compaction"——同样撞到红线，但先生成一个 Summary（橙色块），然后开一个 New context window，从摘要继续跑到 End，Completed。*

作者先用一个受控实验证明摘要质量确实是瓶颈：固定执行 agent 为 GLM-4.7-Flash，只换负责生成摘要的模型。

| 执行 agent | 摘要 agent | SWE-Verified Acc. | 每条轨迹平均摘要次数 |
|---|---|---|---|
| GLM-4.7-Flash | Qwen3.5-27B | 55.5 | 1.010 |
| GLM-4.7-Flash | GLM-4.7-Flash | 50.5 | 1.075 |
| GLM-4.7-Flash | Qwen3-30B-A3B | 49.0 | 1.126 |

只换摘要模型，准确率从 49.0 到 55.5，差 6.5 个点；而且好的摘要器触发的压缩次数还略少一些。有意思的是 GLM-4.7-Flash 给自己做摘要（50.5）反而不如 Qwen3.5-27B 给它做（55.5）——摘要能力和执行能力并不同源。这个观察直接引出了论文的动机：**把摘要生成也训起来**。

### 为什么 GRPO 在这里更不适用

压缩改变了 RL 数据的采样结构。GRPO 假设一个 prompt 对应固定 $$G$$ 条完整 rollout，用组内奖励归一化算优势。一旦轨迹被压缩切段，这个假设有两种破法：

- 如果把每个压缩段当作独立的优化样本，那么 $$G$$ 条 rollout 产生的不是 $$G$$ 个样本，而是 $$\sum_{g=1}^{G} K_{g}$$ 个段（$$K_{g}$$ 是第 $$g$$ 条 rollout 的段数）。因为同一条 rollout 的所有段共享同一个最终奖励，**压缩次数多的 rollout 会在 group 统计里被重复计入更多次，拿到不成比例的权重**。
- 如果坚持只在完整 rollout 层面做归一化，那就拿不到段级优势，无法独立优化执行段和摘要段。

所以 CompactionRL 也选了 PPO：value function 给出的优势估计不依赖固定大小的奖励组，能适应变长的段数，也支持一个 prompt 只采一条 rollout 的情形（实验里 group size 确实就是 1）。

这里和 SAO 的逻辑完全一致——都是"group 结构装不下我的采样方式，所以回到 critic"。

### 方法：三个部件

![CompactionRL 总览](/assets/images/glm-agentic-rl/crl-overview.png)
*Figure 2：上半是 CompactionRL。蓝色块是 assistant 轮次、灰色是环境观察、橙色是摘要、紫色 R 是最终奖励。左起 Execution Segment 1 跑到红色虚线（"Remaining Context < $$T_\text{comp}$$"）时生成 Summary 1；接着从"$$S_{1}$$ + Recent turns"（虚线框，即 Resume context）开始 Execution Segment 2；再次触发生成 Summary 2，继续 Segment 3，最后 Verify 得到奖励 R。底部的点划线表示 R 被共享给所有段（Shared Reward），包括两个摘要段。顶部的黑色弧线是 cross-trajectory GAE：把后续段的 token 数 $$N_{>s}$$ 传回给前面的段做折扣修正。下半"RL(w/o compaction)"作为对照，整条 rollout 就是一个 Single Execution Segment。*

**（1）压缩进入 rollout 收集。** 交互历史记作

$$h_{t}=(s,u,z_{1},\ldots,z_{t}),\qquad z_{i}=(a_{i},o_{i})$$

$$s$$ 是 system prompt，$$u$$ 是原始用户指令，$$a_{i}$$ 是第 $$i$$ 步的 assistant 回复，$$o_{i}$$ 是对应的环境观察。$$z_{i}$$ 被当作原子步——工具调用和它的返回不会被压缩切开。

设 $$C$$ 是上下文预算，压缩在剩余预算低于阈值 $$T_\text{comp}$$ 时触发：

$$C-|h_{t}|<T_{\mathrm{comp}}$$

触发后，在当前历史后面接一段固定的摘要指令 $$q_\text{sum}$$，从策略采样出摘要：

$$S_{t}\sim\pi_{\theta}(\cdot\mid h_{t}\oplus q_{\mathrm{sum}})$$

$$q_\text{sum}$$ 要求模型保留继续任务所必需的信息：原始目标、已完成的动作、重要观察、未解决的报错、当前状态、可能的下一步。然后重建上下文：

$$\bar{h}_{t}=(s)\oplus u_{\mathrm{resume}}(S_{t})\oplus(z_{t-k+1},\ldots,z_{t})$$

$$u_\text{resume}(S_{t})$$ 是把摘要包进去的固定模板，末尾保留最近 $$k$$ 步原文（默认 $$k=2$$，装不下时减小）。摘要负责长程信息，最近几步保留精确原文。

**（2）执行段和摘要段共享同一个奖励。** 一条带压缩的 rollout 被自然切成段序列 $$\tau=(\sigma_{1},\dots,\sigma_{K})$$，每段要么是执行段，要么是摘要段。关键点：**摘要器不是外部模块，摘要 token 由同一个可训练策略采样，并且计入 RL 目标**。整条 rollout 的最终任务奖励 $$R(\tau)$$ 分配给所有可训练段。

作者明确说不引入单独的"摘要质量奖励"，理由是手工设计的摘要指标未必反映哪些细节对解题真正有用。这个选择很干净——摘要好不好，由下游任务成不成来判定。

**（3）token 级 loss 归一化。** 概率比按 token 定义

$$\rho_{s,i}(\theta)=\frac{\pi_{\theta}(y_{s,i}\mid x_{s,i})}{\pi_{\theta_{\mathrm{old}}}(y_{s,i}\mid x_{s,i})}$$

损失在整个 batch 的可优化 token 集合 $$\mathcal{M}$$ 上取平均：

$$\mathcal{L}_{\pi}=-\frac{1}{|\mathcal{M}|}\sum_{(s,i)\in\mathcal{M}}\min\left(\rho_{s,i}(\theta)\widehat{A}_{s,i},\ \mathrm{clip}\left(\rho_{s,i}(\theta),1-\epsilon,1+\epsilon\right)\widehat{A}_{s,i}\right)$$

注意分母是 $$\lvert\mathcal{M}\rvert$$（batch 内所有 token 数）而不是样本数。这一点是针对压缩的直接修正：如果按段平均，压缩次数多的 rollout 会因为段数多而贡献更大的权重，即使它和别的 rollout 拿到同样的奖励。按 token 归一化消掉了段数偏差，每个可训练 token 权重相同。

**（4）Cross-trajectory GAE。** 段被独立优化会带来时间信用分配的问题：如果在每一段末尾都放上共享的终局奖励，那么对靠前的段来说，任务成功看起来比实际更"近"，会过度奖励那些发生在任务完成很久之前的动作或摘要。

修正办法是先算段内的局部 GAE：

$$A^{\mathrm{loc}}_{s,i}=\sum_{\ell=0}^{n_{s}-i}(\gamma\lambda)^{\ell}\delta_{s,i+\ell},\qquad\delta_{s,i}=r_{s,i}+\gamma V_{\phi}(x_{s,i+1})-V_{\phi}(x_{s,i})$$

再用后续段的 token 总数 $$N_{>s}=\sum_{j>s}n_{j}$$ 做位置折扣：

$$\widehat{A}_{s,i}=(\gamma\lambda)^{N_{>s}}A^{\mathrm{loc}}_{s,i}$$

这样第 $$s$$ 段第 $$i$$ 个 token 的奖励项总折扣是 $$(\gamma\lambda)^{N_{>s}+n_{s}-i}$$，正好等于它在"拼接后的完整轨迹"里到终局的距离。换句话说，这个修正让分段优化在折扣意义上等价于对整条轨迹做 GAE。

值得注意的是这个修正只是**近似**——它对齐了奖励项的折扣距离，但段与段之间的 value bootstrap 并没有真正接上（每段的 $$V_\phi(x_{n_{s}+1})$$ 仍按终止状态处理）。论文在 Limitations 里也承认了这点。

### 实验：压缩评测下的一致提升

训练设置：两个规模的模型，GLM-4.7-Flash（30B-A3B）和 GLM-4.5-Air-SFT（106B-A30B，由 GLM-4.7 生成的轨迹微调 GLM-4.5-Air 得到）。critic 从同一个 checkpoint 初始化，RL 之前先做 50 步 value 预训练。训练数据用开源的 SWE-Dev，框架用 slime（开源异步 RL 框架）。

batch size 128，**group size 1**，上下文预算 30B 用 64k、106B 用 80k。policy 学习率 $$2\times10^{-6}$$，critic $$3\times10^{-6}$$，每批 2 次 value 更新 + 1 次 policy 更新（和 SAO 的 $$K=2$$ 一致）。同样用 length-adaptive GAE，$$\lambda=1-\frac{1}{\alpha l}$$、$$\alpha=1.5$$。单次回复上限 10,240 token，剩余预算低于 10,240 时触发压缩，每条 rollout 最多压 3 次。

评测在 Harbor 环境用 Terminus-KIRA scaffold，SWE-bench Verified 随机采 200 题、Terminal-Bench 2.0 用全集，最多 250 轮交互、最多 3 次压缩，报 2 次评测均值。

![CompactionRL 主结果](/assets/images/glm-agentic-rl/crl-main-bar.png)
*Figure 1（右）：两个 benchmark × 两个模型规模。浅色是 base 模型，深色是 + CompactionRL，柱子上方标出提升幅度。SWE-Verified：30B 上 +5.5、106B 上 +7.0；Terminal-Bench 2.0：30B 上 +6.8、106B 上 +3.1。四组都是压缩评测设定下的对比。*

主结果表把"单窗口评测"和"压缩评测"分开报，这是这篇论文最重要的信息：

| 模型 | Peak Len. | SWE 单窗口(×1) | SWE 压缩(×4) | TB2.0 单窗口(×1) | TB2.0 压缩(×4) |
|---|---|---|---|---|---|
| GLM-4.7-Flash (30B-A3B) | 64k | 47.5 | 50.5 | 14.6 | 13.4 |
| + RL（无压缩） | 64k | 50.0 | 48.0 | 16.9 | 12.4 |
| + CompactionRL | 64k | 43.7 | **56.0** | 16.9 | **20.2** |
| GLM-4.5-Air (106B-A30B) | 80k | 57.8 | 59.8 | 17.9 | 21.4 |
| + RL（无压缩） | 80k | 58.3 | 62.5 | 20.2 | 23.6 |
| + CompactionRL | 80k | 57.3 | **66.8** | 21.4 | **24.5** |

"单窗口(×1)"关掉压缩，只给一个 peak-length 窗口；"压缩(×4)"允许最多 3 次压缩，等效预算是 4 倍 peak length。参考值：GPT-5 mini（400k 上下文）在 SWE-Verified 上 72.0、TB2.0 上 31.9；不过论文注明公开 baseline 用的是 SWE-bench Verified 全集且 scaffold 可能不同，只作参考。

两个方向的结论都很清楚：

- **压缩评测下 CompactionRL 全胜**，且优于"标准 RL（无压缩）"。标准 RL 能提升单窗口执行（30B 上 47.5→50.0），但这个收益不能稳定迁移到压缩推理，甚至反而变差（50.5→48.0，TB2.0 上 13.4→12.4）。
- **CompactionRL 的单窗口性能不一定提升**，30B 上甚至从 47.5 掉到 43.7。作者的解释是关掉压缩会造成训练-测试不匹配，超长率上升。这个代价论文没有回避，Limitations 里明确写了。

### 消融：摘要训练贡献了多少

| 系统 | 训练预算 | 摘要训练 | SWE 单窗口 | SWE 压缩 | SWE Long | TB2.0 单窗口 | TB2.0 压缩 | TB2.0 Long |
|---|---|---|---|---|---|---|---|---|
| GLM-4.7-Flash (30B) | – | – | 47.5 | 50.5 | 53.5 | 14.6 | 13.4 | 16.9 |
| + RL（无压缩）-64k | 64k | ✗ | 50.0 | 48.0 | 48.5 | 16.9 | 12.4 | 12.4 |
| + RL（无压缩）-128k | 128k | ✗ | 48.3 | 52.5 | 59.0 | 11.8 | 23.6 | 14.6 |
| + CompactionRL（无摘要训练） | 64k×4 | ✗ | 52.5 | 54.5 | 50.2 | 9.0 | 12.4 | 11.2 |
| + CompactionRL | 64k×4 | ✓ | 43.7 | **56.0** | 49.0 | 16.9 | **20.2** | 16.9 |
| GLM-4.5-Air (106B) | – | – | 57.8 | 59.8 | 59.5 | 17.9 | 21.4 | 20.8 |
| + RL（无压缩）-80k | 80k | ✗ | 58.3 | 62.5 | 61.8 | 20.2 | 23.6 | 21.1 |
| + RL（无压缩）-160k | 160k | ✗ | 63.0 | 64.5 | 64.0 | 20.8 | 23.0 | 23.6 |
| + CompactionRL（无摘要训练） | 80k×4 | ✗ | 54.5 | 64.5 | 61.6 | 17.6 | 21.5 | 20.2 |
| + CompactionRL | 80k×4 | ✓ | 57.3 | **66.8** | 62.1 | 21.4 | **24.5** | 22.5 |

"Long"列是给一个更大的非压缩窗口（30B 用 128k、106B 用 160k）作为参照，相当于"假设你有足够长的上下文"。两个观察：

- 保持 peak 上下文很短的前提下，压缩训练在压缩评测上能和"更长上下文训练"打平甚至超过（106B：66.8 对 64.5；30B：56.0 对 52.5）。这是论文想要的核心论断——**用可训练的压缩换取有效训练视野，而不必抬高 peak 上下文**。
- 摘要是否计入 loss，在压缩评测上一致有正收益：30B 54.5→56.0（+1.5）、106B 64.5→66.8（+2.3）；TB2.0 上 12.4→20.2（+7.8）、21.5→24.5（+3.0）。TB2.0 30B 那一格的 +7.8 是全表最大的单项差，但 TB2.0 只有百来道题、报的是 2 次均值，这个数字的置信区间应该不窄。

第二组消融拆开两个优化部件（GLM-4.5-Air-SFT，全部在 80k×4 压缩设定下评测）：

| 系统 | SWE-bench Verified | Terminal-Bench 2.0 |
|---|---|---|
| GLM-4.5-Air | 59.8 | 21.4 |
| + CompactionRL | 66.8 | 24.5 |
| – w/o token-level loss | 60.0 | 21.3 |
| – w/o cross-trajectory GAE | 63.0 | 22.5 |

去掉 token 级归一化，性能几乎退回 base（60.0 / 21.3），也就是说**这一个部件承担了全部提升的大部分**。去掉 cross-trajectory GAE 掉到 63.0 / 22.5，损失约一半。作者的解读是修正变长段数和段长引起的优化偏差尤其关键——这和直觉一致：段数偏差是一个系统性的权重错误，而 GAE 的折扣修正只是精度问题。

### 行为分析：学到的不是"跑更久"

![平均压缩次数](/assets/images/glm-agentic-rl/crl-behavior-compaction-count.png)
![平均工具调用次数](/assets/images/glm-agentic-rl/crl-behavior-tool-calls.png)
![压缩任务准确率](/assets/images/glm-agentic-rl/crl-behavior-compacted-acc.png)
*Figure 3：GLM-4.5-Air 在 80k×4 压缩评测下的行为对比，五根柱子依次是 base（GLM-4.5-Air-SFT）、RL-80k、RL-160k、CompRL(无摘要训练)、CompRL。(a) 每条轨迹平均压缩次数：0.47 / 0.21 / 0.26 / 0.58 / 0.36。(b) 平均工具调用次数：83.4 / 48.2 / 63.5 / 90.8 / 60.7。(c) 在触发了压缩的那些任务上的 Pass@1：35.4 / 29.0 / 45.9 / 42.4 / **47.7**。*

这三张图回答了一个自然的质疑：CompactionRL 的提升是不是仅仅因为它被允许跑更长的轨迹？

答案是不是。CompactionRL 的压缩次数（0.36）和工具调用数（60.7）都比 base（0.47 / 83.4）和无摘要训练的变体（0.58 / 90.8）**更少**，只比完全不带压缩训练的 RL-80k/160k 多。也就是说它既用上了压缩带来的延长视野，交互效率还更高。作者的解释是训练过的摘要更好地保留了任务相关信息，减少了压缩后的重复探索——这个因果链条合理，但严格说来图里只能读出相关性。

(c) 是更针对性的证据：只在**触发了压缩的任务子集**上比 Pass@1，CompactionRL 最高（47.7）。这个子集本来就是压缩质量最吃紧的地方。

![摘要长度](/assets/images/glm-agentic-rl/crl-summary-length.png)
![每轮推理 token 数](/assets/images/glm-agentic-rl/crl-reasoning-tokens.png)
![策略熵](/assets/images/glm-agentic-rl/crl-entropy.png)
*Figure 4：GLM-4.5-Air CompactionRL 的训练动态，横轴都是训练步数（0–80）。(a) 摘要长度：深蓝 CompactionRL 从 2080 涨到 2350，浅蓝无摘要训练的变体从 2060 跌到 1650 附近——摘要计入 loss 与否，长度走向完全相反。(b) 每轮推理 token：CompactionRL 从 90 涨到 193，另两条（无摘要训练、无压缩）都降到 60–78。(c) 策略熵：三条都在涨，但 CompactionRL（0.38→0.43）涨得最慢，无压缩 RL 涨到 0.57。*

(a) 这张图是最直观的：**摘要不计入 loss 时，摘要会自己变短**。原因不难想——摘要 token 只是被动生成，没有任何梯度鼓励它详细，模型倾向于走捷径。计入 loss 后摘要越来越长、越来越具体，作者的说法是这样能保留更多"实现相关的上下文和续接状态"，而不只是记录高层进度。

(b) 推理 token 增加，作者解释为"学到的压缩有效扩大了可用上下文窗口，从而腾出更多推理预算"。这个因果方向需要一点推敲：也可能是同一个优化过程同时让摘要和推理都变长。(c) 熵涨得慢，论文只说"更受控的策略优化"，没有展开。

## 放在一起看：一条共同的技术路线

把两篇的技术选择列在一起，重合度相当高：

| 维度 | SAO | CompactionRL |
|---|---|---|
| 触发动机 | 异步训练下 group 是同步屏障 | 压缩把 rollout 切成变长段数 |
| 优化框架 | PPO + value model | PPO + value model |
| group size | 1 | 1 |
| 优势粒度 | token 级 GAE | token 级 GAE |
| GAE 变体 | Skip-Observation（跳过环境观察 token） | Cross-trajectory（跨压缩段折扣修正） |
| loss 归一化 | — | token 级（消融里最关键的一项） |
| critic : policy 更新比 | 2 : 1 | 2 : 1 |
| value 初始化 | 从 SFT checkpoint + 扩大 value 预训练 | 从同一 checkpoint + 50 步 value 预训练 |
| length-adaptive GAE | $$\alpha=1.5$$ | $$\alpha=1.5$$ |
| 主要 benchmark | AIME/BeyondAIME/HMMT/IMOAnswerBench + SWE-Bench Verified | SWE-bench Verified + Terminal-Bench 2.0 |
| 声明的落地 | GLM-5.2 agentic RL pipeline | GLM-5.2 RL pipeline |

两篇合起来讲了一个完整的故事：**GRPO 的 group 结构是一种对采样方式的硬约束，一旦 agentic 场景要求更灵活的采样（异步到达、变长分段、单条环境反馈），这个约束就得让位，代价是必须把 value model 训好**。而"训好 value model"这件事被两篇论文拆成了一批具体可操作的手段：critic 更新更频繁、冻结 attention、扩大 value 预训练、按任务结构改写 GAE 的连接方式、按 token 而非样本归一化 loss。

有意思的是这几乎是把 2023 年 RLHF 时代的 PPO 工程经验，在 agentic 场景里重新做了一遍——只不过这次要处理的结构复杂度高得多：环境观察不是模型生成的、轨迹会被压缩切段、rollout 异步到达。

一个自然的猜想是：两篇的技术在 GLM-5.2 的 pipeline 里应该是同时开启的（异步 + 单条 rollout + 压缩训练），毕竟长程 coding agent 同时需要这三件事。但两篇论文都没有报告组合起来的结果，各自的实验也在不同 backbone 上（Qwen3-30B-A3B vs GLM-4.7-Flash / GLM-4.5-Air）。

## 讨论

**value model 的成本被低估了吗。** 两篇都把"训好 critic"当作方法的前提，但都没有正面讨论这件事的开销。一个和 policy 同规模的 value model 意味着显存翻倍；SAO 还额外需要"扩大 value 预训练语料"（论文没说规模），CompactionRL 需要 50 步 value 预训练；critic 每批更新 2 次，等于 value 侧的计算量是 policy 的两倍。相对于 GRPO，这套方案的总成本增加了多少？如果把同样的算力给 GRPO（比如更大的 batch 或更多步数），差距还有多少？这是最想看到但两篇都没给的对照。异步带来的吞吐收益和 value model 带来的成本，孰大孰小，取决于具体集群配置，值得一个 wall-clock 的对比。

**冻结 attention 这个发现能推广吗。** SAO 观察到 value model 的梯度不稳定主要来自 full attention 层而非 MoE 层，于是冻住 attention。这个定位是经验性的，论文给的假设（"预训练 attention 已有足够语义定位能力"）没有进一步验证。一个自然的问题是：这个现象是 MoE 架构特有的，还是 dense 模型也成立？如果换成 dense value model，"只训 FFN"是否同样有效？另外冻结 attention 在 AIME 上贡献 6.7 分而在 BeyondAIME 上只有 0.3 分，这个不一致本身也提示这条规律可能对任务分布敏感。

**摘要不给专门奖励，是优点也是限制。** CompactionRL 明确拒绝设计摘要质量奖励，让摘要好坏完全由下游任务成败来判定。这个选择避开了指标设计的主观性，而且实验证明有效（摘要变长变详细）。但它也意味着信号非常稀疏：一次任务成功要归因到 3 次压缩里的哪一次摘要？cross-trajectory GAE 就是在处理这个问题，但论文也承认它只是近似。如果进一步做，一个方向是引入过程性信号——比如摘要之后的若干步里，模型是否重复探索了摘要本该记住的东西——这类信号可以自动检测，不需要人工标注摘要质量。

**训练-测试不匹配是个实际问题。** CompactionRL 的单窗口性能下降（30B 上 47.5→43.7）不是小事。实际部署时，一个 agent 未必总会触发压缩——短任务在一个窗口内就结束了。如果模型在这类任务上变差，整体收益就要重新算。论文的 Limitations 提到了这点。一个自然的想法是训练时混合有压缩和无压缩的轨迹，让模型两种模式都保持；不过这会让 batch 的结构更复杂。

**基准的统计强度。** 两篇的关键结论都建立在几个百分点的差距上。SAO 报了评测次数（AIME/HMMT/IMOAnswerBench 16 次、BeyondAIME 4 次），CompactionRL 报了 2 次均值，但两篇都没有给误差棒或置信区间。AIME2025 只有 30 道题，Terminal-Bench 2.0 也是百题量级，SWE-bench Verified 在 CompactionRL 里还只取了 200 题的随机子集。以 CompactionRL 的 66.8 对 64.5（消融里摘要训练的贡献）为例，200 题上 2.3 个点约等于 4.6 道题的差别，2 次评测的均值很难说这个差距稳定。

## AI 犀利评判

**一、"单条 rollout 更有利于泛化"这个论断没有被真正验证。** SAO 的摘要和引言里反复出现"reduce off-policy effects and improve generalization"的表述，但论文给的证据只是最终 benchmark 分数更高。分数高可以有很多原因：value model 训得更久、超参调得更好、DIS 的 clip 边界更适合。真正支持"单条 rollout 降低了 off-policy 程度"的直接测量——比如轨迹生成期间跨越的模型版本数、token 级 $$r_{t}$$ 的分布对比——一个都没有。Figure 3 显示 SAO 和 GRPO(w/DIS) 前 400 步完全重叠，反而说明单条 rollout 本身在早期没有任何优势；后期的分叉更像是 value model 收敛程度的差异（Figure 4a 的 EV 曲线在同一位置分叉），而这是"critic 训得好"的功劳，不是"单条 rollout"的功劳。把两者混在一个方法名下报告，让读者无法区分。

**二、SAO 的核心增益其实来自 DIS，而 DIS 是一个纯粹的稳定性补丁。** 拆开表 1：GRPO 84.2 → GRPO+DIS 93.5，这 +9.3 分几乎全部来自"让训练不在 160 步崩掉"。而 DIS 做的事——丢掉 $$\pi_{\theta_\text{old}}$$、双侧硬 mask——从方法论上说是减法，不是新机制，论文自己也承认与 IcePop 相似。剩下的 +3.8（93.5 → 97.3）才是"单条 rollout + value 工程"的贡献，而这部分是由四个设计（faster value、frozen attention、skip-observation GAE、扩大 value 预训练）共同堆出来的，其中"扩大 value 预训练"连消融都没有。以论文标题的 framing（Single-Rollout 是主角）对照实际贡献分布，主次是倒置的。

**三、消融表的内部矛盾没有被处理。** SAO 表 4 里，去掉 frozen attention 在 AIME 上掉 6.7 分，在 BeyondAIME 上掉 0.3 分；去掉 faster value 在 BeyondAIME 上掉 5.0 分，在 AIME 上掉 2.3 分。两个组件的效果在两个 benchmark 上强弱完全对调。论文正文的结论句是"all examined variants exhibit a performance decline relative to the proposed SAO, validating the necessity of each design choice"——这是把方向一致当成了强度一致。更诚实的读法是：这些差异有相当一部分落在评测噪声里，而论文没有给误差棒让读者判断。另外表 3 和表 4 的 caption 完全相同（一字不差），内容却不同，这是投稿稿件的编辑疏漏。

**四、CompactionRL 的消融暴露了一个 framing 问题。** 表 4 显示去掉 token 级 loss 归一化后，性能从 66.8 掉回 60.0，而 base 是 59.8——**几乎全部提升都来自这一个部件**。但 token 级 loss 归一化是什么？它是修正"段数多的 rollout 被重复计权"这个 bug。也就是说：如果不做这个修正，把压缩段当独立样本训练不仅没用，还基本等于白训。这意味着论文的真实贡献更接近"我们指出并修好了分段训练的一个权重 bug"，而不是标题暗示的"我们提出了带压缩的 RL 框架"。cross-trajectory GAE 作为论文形式上最漂亮的推导（那个 $$(\gamma\lambda)^{N_{>s}}$$ 折扣），实际只值 3.8 分中的一半左右，而且作者自己承认它只是近似——段间的 value bootstrap 并没有真正接上。

**五、baseline 的选择在两篇里都偏软。** SAO 拿 vanilla GRPO 做主对照，而 vanilla GRPO 在 160 步就崩了——用一个会崩的 baseline 证明自己稳定，说服力有限。更公平的对照应该是"GRPO + DIS + 同等 value 工程投入"，但 GRPO 不需要 value model，这个对照不存在，于是真正的问题变成"同等算力下 GRPO 能走多远"，论文没答。CompactionRL 的情况类似：它的 "RL (w/o compaction)" 对照在压缩评测下天然吃亏（训练时没见过压缩历史，测试时却要处理），train-test 不匹配的方向对 CompactionRL 有利。作者确实报了反向的单窗口评测（CompactionRL 在那里掉分），这一点值得肯定，但主结果的呈现顺序和加粗仍然只强调有利的那一半。

**六、"已部署到 GLM-5.2"是一个无法核验的信号。** 两篇论文都在摘要结尾放了这句话（SAO：750B-A40B；CompactionRL：同）。这句话在评审意义上不提供任何信息——没有 GLM-5.2 的对照实验，没有说明这两套方法在其中各自开启了哪部分、贡献了多少。它的作用是给方法背书（"这是真在大模型上用的东西"），但读者无法验证。考虑到两篇论文的实验规模都停在 30B/106B，而声明落地的是 750B，中间的 scaling 跨度也没有任何证据支撑。

**七、有一个反直觉的数据点两篇都没解释。** CompactionRL 表 1 里，GLM-4.7-Flash 给自己做摘要（50.5）显著不如 Qwen3.5-27B 给它做摘要（55.5）——差 5 个点。这个观察其实比论文的主线更有意思：它暗示摘要能力和执行能力可能需要不同的优化目标，甚至可能存在"自己总结自己"的系统性盲区（模型倾向于省略它认为显然但实际关键的信息）。论文只把这张表当作"摘要质量重要"的引子就带过了。同样，SAO 的在线学习实验里 Running mean 在第一阶段反超 SAO 0.14，论文完全没提这件事。

**总体判断**：两篇都是扎实的工程论文，把"agentic 场景下 GRPO 的 group 结构不够用、需要回到 critic"这个判断落成了一批可复现的具体做法，DIS 的简化和 token 级 loss 归一化这两点尤其实用。但两篇的 framing 都把功劳分配给了标题里最新颖的部件（Single-Rollout、带压缩的 RL 框架），而消融显示真正扛住提升的是更朴素的东西（稳定性补丁、权重归一化修正）；加上缺少误差棒、缺少同算力对照、缺少 GLM-5.2 落地的任何证据，论文的说服力明显低于其叙事的自信程度。
