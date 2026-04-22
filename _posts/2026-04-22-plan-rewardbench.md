---
layout: post
title: "【论文笔记】Plan-RewardBench：评估 Agent 轨迹级奖励建模的 Benchmark"
date: 2026-04-22
categories: [paper-notes]
tags: [reward-model, agent, tool-use, benchmark, RLHF]
paper_title: "Aligning Agents via Planning: A Benchmark for Trajectory-Level Reward Modeling"
paper_authors: "Jiaxuan Wang, Yulan Hu, Wenjin Yang et al."
paper_link: "https://arxiv.org/abs/2604.08178"
---

> **论文**：[Aligning Agents via Planning: A Benchmark for Trajectory-Level Reward Modeling](https://arxiv.org/abs/2604.08178)
> **作者**：Jiaxuan Wang, Yulan Hu, Wenjin Yang, Zheng Pan, Xin Li, Lan-Zhe Guo
> **机构**：南京大学 / 阿里巴巴高德

当 LLM 从"回答问题"进化到"使用工具完成任务"时，奖励模型（RM）的评估对象也应该从单条回复变成一整条包含规划、工具调用、环境反馈的多步轨迹。但现有的 RM benchmark（RewardBench、FC-RewardBench 等）基本停留在 response 级别或单次 tool-call 级别，没有覆盖到轨迹级的复杂判断。Plan-RewardBench 试图填补这个空白：它构造了一个 pairwise 轨迹偏好数据集，让各种评估器（判别式 RM、生成式 RM、LLM-as-Judge）在多步工具使用场景下判断哪条轨迹更好。

## 评估的粒度从 Response 到 Trajectory

![Figure 1](/assets/images/plan-rewardbench/x1.png)
*Figure 1：Plan-RewardBench 概览——左侧对比不同 benchmark 的评估粒度，右侧展示数据构造流程*

现有 benchmark 的核心问题在于评估粒度太粗。RewardBench 只看最终回复质量；FC-RewardBench 只检查工具名称和参数对不对。但在真实的 agent 场景中，一条轨迹可能经历十几轮工具调用，中间涉及规划调整、错误恢复、安全拒绝等复合决策。一个流畅的最终回答可能完全忽视了工具返回的矛盾信息，而一个看起来"笨拙"的轨迹可能在错误恢复上做得更好。

Plan-RewardBench 的评估单元是完整轨迹：给定工具环境描述 $\mathcal{T}$、多轮用户交互、以及两条候选轨迹 $(\tau_A, \tau_B)$，评估器需要判断哪条更优。金标签基于四类场景的专用标注准则生成。

## 四类场景覆盖 Agent 的典型失败模式

Benchmark 将任务分成四个 scenario family，每个对应一类 agent 容易出错的维度：

**Safety Refusal**：agent 是否正确拒绝了有害请求。难点在于区分"基于政策的坚定拒绝"和"基于能力限制的模糊推脱"，以及在长对话中前面一切正常、最后一步才出现安全违规的情况（compliance inertia）。

**Tool-Irrelevance / Unavailability**：当任务根本不需要工具、或可用工具无法完成任务时，agent 是否能诚实地说明，而非编造工具调用或幻觉工具结果。

**Complex Planning**：多步规划的核心场景。分为 single-turn 和 multi-turn，各自又按复杂度分 easy/hard。关键失败模式包括工具结果编造（声称调用了工具但实际没有）、参数/schema 错误、以及用户中途修改需求后 agent 仍按旧计划执行。

**Robust Error Recovery**：工具执行出错时 agent 的应对能力。重点区分"盲目重试同一个失败调用"和"诊断错误原因后换策略修复"。

![Figure 3](/assets/images/plan-rewardbench/x3.png)
*Figure 3：四类场景中 rejected 轨迹的失败模式分布*

## 数据构造：多源负样本 + 人工审核

数据来源以 Toucan/MCP 的工具注册表和执行记录为基础，通过三种方式生成候选轨迹：

1. **自然 rollout（70%）**：用 Qwen-Agent 和 OpenAI-Agent 两个运行时，搭配不同底座模型、system prompt、温度参数生成多条轨迹。自然产生的错误保留了真实分布。

2. **规则扰动（8%）**：引入可控的失败，如约束丢弃/交换、数值错误、实体替换、工具幻觉注入、盲目重试等。

3. **最小编辑扰动（22%）**：从高分轨迹出发做小改动，只降低某一个维度的质量（比如把"强拒绝"改成"弱拒绝"），保持风格和长度不变。这是构造 hard negative 的关键手段。

![Figure 2](/assets/images/plan-rewardbench/x2.png)
*Figure 2：轨迹来源分布*

偏好标注采用两阶段流程：先由 3 个 LLM judge 按场景专用 rubric 打分（1-5 分），取中位分和多数投票标签；分歧大的交给 meta-review；仍然模糊的丢弃。最后组装 pair 时控制两个维度——难度（easy pair 分差 $\geq 2$，hard pair 分差 $=1$）和偏差（按长度/格式分层，确保评估器不能靠表面线索判断）。

人工审核结果（Table 2）显示 Cohen's $\kappa$ 在 0.71-0.86 之间，达到 substantial 到 almost perfect 的一致性水平。

## 评估协议：Pointwise vs. Pairwise

Benchmark 统一采用 pairwise 准确率作为指标——给定一对轨迹（chosen vs. rejected），评估器选对了就算正确。但被评估的模型分两种协议到达这个结果：

**Pointwise（判别式 RM / DRM）**：对每条轨迹独立输入（工具环境 + 对话历史 + 单条轨迹），模型输出一个 scalar reward 分数。两条轨迹分别打分后，分数高的那条作为模型的选择。因为两条轨迹是独立评估的，天然没有位置偏差问题。论文评测的 DRM 全部是已有的开源模型：Inf-ORM-Llama3.1-70B（InftyAI）、InternLM2-7B-Reward（上海 AI Lab）、Skywork-Reward-V2 的 Qwen3-8B 和 Llama-3.1-8B 两个版本（昆仑万维）、FsfairX-LLaMA3-RM-v0.1（Salesforce）、QRM-Gemma-2-27B（阿里）。

**Pairwise（生成式 RM / GRM + 通用 LLM Judge）**：模型同时看到两条轨迹，使用场景专用的 rubric prompt（Planning / Robustness / Safety / Tool-Irrelevance 四套），输出选择（A 或 B）和 critique 推理过程。GRM 和 LLM Judge 用的是完全相同的评估协议和 prompt，区别仅在于模型本身——GRM 是专门训练过偏好判断的开源模型（微软 RRM-32B、Unbabel M-Prometheus-14B、UIUC RM-R1-DS-Distill-Qwen-32B），而 LLM Judge 是通用大模型（Qwen-Plus、GPT-5、DeepSeek-R1 等）。所有 pairwise 评估器都执行 **A/B swap**：每对轨迹评估两次，第二次交换呈现顺序，取平均准确率，以消除位置偏差。

两种协议下，所有评估器收到的上下文内容完全相同：Tool-Env schema、多轮对话历史、完整的工具调用和执行输出。同一对中唯一的变量是 agent 的轨迹内容本身。

整体指标 Avg 是 7 个子集（4 个 Complex Planning 子集 + Robust + Safety + Irrelevance）的 macro-average，有意让 Complex Planning 占更大权重，因为这是 benchmark 的核心关注点。

## 实验结果：三类评估器的意外排序

下表列出了代表性模型在各子集上的 pairwise 准确率（%），按 Avg 排序：

| 模型 | 类型 | Multi-E / Multi-H / Sngl-E / Sngl-H | Robust | Safety | Irrel. | Avg |
|:---|:---|:---|:---|:---|:---|:---|
| Qwen-Plus | LLM Judge | 68.4 / 68.8 / 84.6 / 74.7 | 73.8 | 55.9 | 63.7 | **70.0** |
| DeepSeek-V3.2-Exp | LLM Judge | 69.3 / 61.6 / 79.5 / 74.8 | 66.8 | 75.0 | 60.0 | 69.6 |
| Qwen3-235B-A22B | LLM Judge | 69.3 / 67.4 / 84.4 / 71.4 | 68.8 | 65.7 | 59.8 | 69.5 |
| **Inf-ORM-70B** | **DRM** | **70.3 / 65.0 / 79.9 / 74.1** | **69.8** | **58.5** | **66.9** | **69.2** |
| Gemini-3-Flash | LLM Judge | 66.4 / 47.5 / 81.1 / 67.3 | 67.3 | 78.4 | **75.6** | 69.1 |
| GPT-5 | LLM Judge | 64.0 / 45.8 / 83.9 / 62.2 | 69.4 | **84.8** | 69.7 | 68.5 |
| RRM-32B | GRM | 68.5 / 62.1 / 75.2 / 70.8 | 67.2 | 60.3 | 61.2 | 66.5 |
| M-Prometheus-14B | GRM | 65.2 / 58.8 / 72.3 / 68.4 | 64.6 | 62.4 | 63.9 | 65.1 |
| RM-R1-Distill-Qwen-32B | GRM | 67.3 / 59.2 / 71.8 / 68.6 | 62.2 | 56.6 | 69.1 | 65.0 |
| QRM-Gemma-2-27B | DRM | 58.9 / 49.1 / 55.6 / 44.9 | 49.3 | 54.9 | 56.9 | 52.8 |
| Llama-3.2-8B | LLM Judge | 55.3 / 36.2 / 56.6 / 55.1 | 51.6 | 49.5 | 50.5 | 50.7 |

最出乎意料的排序是：**通用 LLM Judge > DRM (70B) > GRM > DRM (小规模)**。一个 pointwise scalar RM（Inf-ORM-70B）以 69.2% 排第四，打赢了 GPT-5、DeepSeek-R1 和所有 GRM；而专门训练过偏好判断的 GRM 反而全部落后于头部 LLM Judge 和这个 70B DRM。

### GRM 为什么不如通用 LLM Judge

GRM 和 LLM Judge 用的是完全相同的 pairwise 协议和 prompt，唯一区别是模型本身。最好的 GRM（RRM-32B，66.5%）比最好的 LLM Judge（Qwen-Plus，70.0%）低 3.5 个百分点，甚至不如 4B 参数的 Qwen3-4B-Instruct（67.4%）。

这些 GRM 的训练数据以 response-level 偏好为主（对话质量、有用性等），没有见过 trajectory-level 的多步工具调用场景。当评估对象从"哪个回答更好"变成"哪条包含十几步工具调用的轨迹规划更合理"时，专门的偏好训练反而可能成为领域迁移的负担——模型学到的判断 heuristic 不适用于新场景。

### DRM 70B 为什么能和 LLM Judge 打平

Inf-ORM-70B 的竞争力来自两个因素：

**规模效应显著**。同样是 DRM，70B 的 Inf-ORM 到 69.2%，27B 的 QRM-Gemma 只有 52.8%——差了 16 个点。但规模不是万能的：27B DRM 远不如同规模的 pairwise judge，说明 pointwise 协议在小规模下劣势明显。

**Pointwise 协议在长轨迹上有结构性优势**。DRM 每次只看一条轨迹，而 pairwise judge 要同时塞两条，上下文翻倍。在轨迹最长的 Robust Recovery 子集上（最大 ~30k token/条），pairwise judge 的上下文负担最重，DRM 的优势也最明显。

但 DRM 有硬伤：Safety 上 Inf-ORM 只有 58.5%，接近随机水平，而 GPT-5 到了 84.8%。scalar reward 能捕捉显式信号（错误日志、工具调用格式是否正确），但对"这个拒绝是好拒绝还是坏拒绝"这种需要语义推理的判断基本无能为力。

### 没有全能选手

**轨迹级判断是多面的**。整体最优的 Qwen-Plus（Avg 70.0%）在 Safety 上只有 55.9%，远逊于 GPT-5 的 84.8%。在 Tool-Irrelevance 上 Gemini-3-Flash（75.6%）又明显领先。没有任何一个评估器能同时搞定所有维度。

**Safety 是最分裂的维度**。LLM judge 在 Safety 上的准确率从 40.7% 到 84.8% 不等，方差极大。区分"好的拒绝"和"坏的拒绝"对当前模型仍然很难。

## 长轨迹下的性能崩塌

![Figure 4](/assets/images/plan-rewardbench/x4.png)
*Figure 4：准确率随输入长度变化——(a) pairwise LLM judge 在超过 32k token 时急剧下降，(b) pointwise RM 下降更平缓*

在短上下文（< 4k token）下各评估器表现稳定，但超过 32k token 后 pairwise LLM judge 出现断崖式下降，部分模型甚至低于随机水平（50%）。Pointwise RM 因为独立打分、不用拼接两条轨迹，衰减相对平缓。

不过这张图的解读需要谨慎。论文将其归因为"length sensitivity"，但有一个没有被控制的混淆变量：**长轨迹和高难度场景是高度耦合的**。32k+ 区间的样本几乎全部来自 Robust Recovery 和 Planning-Hard——这些本身就是最难的子集。准确率的下降到底多少是因为上下文变长导致模型理解能力退化，多少是因为这些样本本身就更难判断，从图中无法区分。要真正验证长度效应，需要在同一场景、同一难度下控制轨迹长度做对比，论文没有做这个分析。

另一个值得注意的现象是部分模型在某些 length bin 上准确率远低于 50%。论文执行了 A/B swap 协议（正反各评一次取平均），这意味着位置偏差已经被消除了。swap 之后仍然低于 50%，说明模型在内容层面存在系统性的反向偏好——例如总是选择更长或更啰嗦的那条轨迹，而 rejected 轨迹恰好更长（Robust Recovery 的 rejected 平均 29.6k vs chosen 17.2k）。这种系统性偏差比"看不懂长文本"更有信息量，但论文没有进一步挖掘成因。

此外，从图中可以看到 Qwen-Max 在 32k+ bin 上准确率从 ~0.7 骤降到接近 0，这种断崖式跳变不符合能力渐进退化的规律，更像是该 bin 样本极少且模型在超长输入上出现了系统性故障（如 API 截断、输出格式崩坏或拒绝作答），导致所有判断失败。类似地，Kimi-K2-Thinking 在 4-8k bin 骤降到 ~0.4 后又在后续 bin 回升，也呈现出与"长度越长越差"不一致的非单调行为。论文没有对这些明显的异常值做任何说明，也没有报告每个 length bin 的样本数量，这让极端区间数据点的可靠性存疑。

## 四类典型失败模式

论文的定性分析揭示了评估器判断出错的四种模式：

1. **安全优先级错位**：评估器更偏好"我没有这个工具"这样的能力借口，而非直接基于政策的拒绝。长对话中还存在 compliance inertia——前面一直很有帮助的表现会"掩护"最后一步的安全违规。

2. **约束过期盲区**：用户中途修改需求后，评估器仍然奖励按原始计划执行完毕的轨迹，忽略了约束已经变化。

3. **努力偏差**：在 Tool-Irrelevance 场景中，评估器倾向于惩罚直接给出答案的高效回复，反而奖励做了无意义工具调用的轨迹——形成一种"调用越多 = 越有帮助"的偏见。

4. **表面恢复**：在 Error Recovery 中，评估器无法区分盲目重试和真正的诊断修复，只要看到"重试了"就倾向给高分。

## 下游验证：Best-of-N Reranking

论文在 BFCL v4 上做了 best-of-4 reranking 验证：用 Qwen3-32B 生成 4 条候选轨迹，分别用不同评估器选最优。Plan-RewardBench 上表现好的评估器确实能带来更高的下游收益——Qwen-Plus 将 BFCL Overall 从 48.71 提升到 55.14，而弱模型（Llama-3.2-8B）几乎等于随机选择。这为 benchmark 的实际意义提供了外部证据。

## 思考

Plan-RewardBench 指向的核心问题是：当我们用 LLM-as-Judge 或 RM 来为 agentic RL 提供奖励信号时，这些评估器本身在长轨迹、多工具场景下的可靠性是存疑的。最好的模型也就 70% 准确率，在 hard 子集上更低。如果用这样的评估器做 RL training 的 reward provider，噪声和偏差会直接传导到 agent 策略中。

一个自然的后续方向是针对轨迹级判断做专门训练。论文的数据构造流程（自然 rollout + 规则扰动 + 最小编辑）本身就是一个生成 preference data 的 recipe，可以直接用于训练更好的 trajectory-level RM。另一个值得关注的点是 pairwise 和 pointwise 两种评估协议的互补性——前者在短上下文下更准但长上下文衰减严重，后者更稳定但天花板较低。实际部署中可能需要根据轨迹长度选择不同的评估策略。
