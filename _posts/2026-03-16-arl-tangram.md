---
layout: post
title: "【论文笔记】ARL-Tangram: Agentic RL 训练中的外部资源效率优化"
date: 2026-03-16
categories: [paper-notes]
tags: [reinforcement-learning, resource-management, systems, LLM, scheduling]
paper_title: "ARL-Tangram: Resource Efficiency in Agentic Reinforcement Learning"
paper_authors: "Bangjun Xiao, Yihao Zhao, Xiangwei Deng et al."
paper_link: "https://arxiv.org/abs/2603.13019"
---

# ARL-Tangram: 把"七巧板"思路用到 Agentic RL 的资源调度上

> **论文**：[ARL-Tangram: Resource Efficiency in Agentic Reinforcement Learning](https://arxiv.org/abs/2603.13019)
> **作者**：Bangjun Xiao, Yihao Zhao, Xiangwei Deng, Shihua Yu, Yuxing Xiang, Huaqiu Liu, Qiying Wang, Liang Zhao, Hailin Zhang, Xuanzhe Liu, Xin Jin, Fuli Luo
> **机构**：Peking University, LLM-Core Xiaomi
> **发表**：arXiv 2026.03

---

Agentic RL（让 LLM 在训练中调用外部工具和环境）正在成为后训练阶段的重要范式——AI Coding（SWE-bench 类任务）、DeepSearch（联网搜索 + LLM-as-judge 奖励）等场景已经大量使用。但一个被低估的问题是：**这些外部资源（sandbox 环境、reward model GPU、搜索 API 配额）的利用率极低**。这篇来自北大和小米的工作量化了这个浪费——在 AI Coding 任务中，环境平均只在 47% 的 trajectory 生命周期内被实际使用——然后提出 ARL-Tangram 系统，通过将资源管理粒度从 trajectory/task 级别下沉到 **action 级别**（即单次工具调用），配合弹性调度和异构资源管理器，实现了最高 **4.3× 的平均 action 完成时间降低**、**1.5× 的训练 step 加速**和 **71.2% 的外部资源节省**。该系统已部署于小米 MiMo 系列模型的 RL 训练中。

---

## 问题：两层资源过度分配

### Agentic RL 的 Rollout 模式

Agentic RL 的 rollout 不同于传统 RLHF。在 ReAct 范式下，LLM 在每一步生成一个 action（如执行代码、调用搜索），然后等待环境返回 observation，再生成下一步。一个 trajectory 可能包含多轮交互，期间需要 sandbox 环境（CPU 容器）、reward 模型（GPU 服务）、或外部 API（配额限制）。

关键矛盾：**资源只在 action 执行时被真正使用，而在 LLM 生成 token 的过程中完全空闲**。

### 第一层：Trajectory 级别的过度分配

现有框架（如 VeRL、OpenRLHF）在 rollout 开始时为每个 trajectory 分配独占的外部资源，直到 rollout 结束才释放。论文实测了 AI Coding 场景：

- 环境利用率仅约 **47%**——超过一半的时间里，分配的 sandbox 容器在空等 LLM 出 token
- GPU reward 服务的 SM 利用率平均振荡在 **3% 以下**，呈现极端的脉冲式负载

这意味着你需要为 1280 个并发 trajectory 预留 1280 个 sandbox 实例，但实际同时在执行 action 的可能只有 600 个。

### 第二层：Task 级别的过度分配

在实际训练中，一次 RL step 可能同时涉及多种外部服务（如 MOPD 任务中的 9 个 teacher model）。每个服务独立部署、各自占 GPU，但被调用的时间点高度不均匀——某些 teacher model 在某些 trajectory 中根本不会被调用。

更糟的是，不同 RL 任务之间的外部服务也无法共享。DeepSearch 和 MOPD 各自维护独立的 GPU 资源池，即使一方空闲时另一方也无法借用。

---

## 核心思路：Action 级别的"拆解与池化"

ARL-Tangram 的命名来自七巧板（Tangram）：把原本绑定在 trajectory 或 task 上的整块资源拆解成细粒度的碎片，然后在一个统一的资源池中动态拼装。

**关键抽象变换**：将资源管理的基本单元从 trajectory（生命周期数分钟）降到 action（生命周期数秒甚至毫秒）。Action 执行完毕后资源立即回收入池，供其他 trajectory 或 task 的 action 使用。

这带来三个好处：

1. **跨 trajectory 共享**：不同 trajectory 的 action 分时复用同一组资源
2. **跨 task 共享**：不同 RL 任务的异构服务可以在同一资源池中"腾挪"
3. **弹性分配**：空闲资源可以临时分配给正在执行的 action 加速完成

---

## 系统架构

ARL-Tangram 作为 RL 训练框架（如 VeRL）和外部资源之间的中间层，工作流程分五步：

1. **Action 提交**：Rollout 过程中 LLM 生成 action 后，将请求提交给 ARL-Tangram
2. **统一 Action 建模**：将异构的 action 请求转化为标准化表示（包括资源消耗向量和弹性特征）
3. **弹性调度**：根据当前资源可用情况，决定调度顺序和资源分配量
4. **资源分配与执行**：通过对应的 Resource Manager 分配资源并执行 action
5. **结果回传**：执行结果返回 RL 框架继续生成

系统包含三个核心组件：

### 组件一：统一 Action 建模

每个 action 被建模为一个**向量化资源消耗**：

$$C_i = (c_{i,0}, c_{i,1}, \ldots, c_{i,k-1})$$

其中每个维度对应一种资源类型（CPU 核心、GPU 卡、内存、API 配额等）。

对于**弹性 action**（可以通过增加资源量来加速的 action），还建模其弹性函数。执行时长与分配资源量 $m$ 的关系为：

$$\text{getDur}(m) = \frac{T_{\text{ori}}}{E(m) \times m}$$

其中 $T_{\text{ori}}$ 是单资源单元下的基准执行时长，$E(m)$ 是弹性比率（$0 < E(m) \leq 1$），反映并行效率的衰减。例如 tensor parallelism 下 $E(m)$ 随 $m$ 递减（通信开销增大），但总的 $E(m) \times m$ 仍然递增。

资源消耗的两种基本模式：
- **并发型**（CPU 容器）：一个容器占用固定 CPU 核心数，同时运行多个容器就乘以容器数
- **配额型**（API 调用）：受限于 requests-per-second 或并发连接数

### 组件二：弹性调度算法

调度目标是**最小化所有 action 的总完成时间**（ACT，Action Completion Time）：

$$\min \sum_i \text{ACT}_i = \sum_i (T_i^q + T_i)$$

其中 $T_i^q$ 是排队时间，$T_i$ 是执行时间。

算法是一个基于贪心驱逐的启发式方法：

**Step 1：初始候选选择**
从等待队列中按 FCFS 顺序选出前 $n$ 个满足资源约束的 action 作为候选集。FCFS 顺序很重要——它防止某些 action 被持续饿死，这在 agentic RL 中尤其关键（某些 trajectory 的后续 action 依赖前序 action 的结果）。

**Step 2：按弹性资源分割**
将候选集按关键弹性资源类型分类。论文假设每个 action 至多有一种关键弹性资源（实际场景中合理——一个 action 要么主要消耗 CPU，要么主要消耗 GPU）。

**Step 3：贪心驱逐**
对每种弹性资源类型的候选子集，迭代执行：
- 尝试驱逐最后一个候选 action（将其释放的资源分配给剩余候选）
- 计算驱逐前后的目标函数近似值
- 如果驱逐后目标改善，则接受驱逐；否则终止

目标函数近似计算分两部分：
- 对已调度的候选 action，精确计算 ACT
- 对队列中剩余 action，用一个基于"完成时间堆"的估计方法近似

**复杂度**：以动态规划为主项，$O(kn^2m^2)$，其中 $k$ 为资源类型数、$n$ 为候选数、$m$ 为剩余资源单元数。考虑到 action 调度在微秒量级内需要完成，论文通过可配置的 depth 参数控制搜索空间。

### 组件三：异构资源管理器

不同资源类型有本质不同的管理方式，系统为 CPU 和 GPU 分别设计了专门的管理器。

#### CPU Manager：Allocate-on-Execution（AOE）

传统做法是为每个 trajectory 启动一个独立容器，容器占用固定 CPU 核心。ARL-Tangram 的 CPU Manager 做法不同：

- 容器预先创建但 **CPU 配额为零**——容器状态保持但不消耗计算资源
- 当 action 需要执行时，通过修改 Docker 的 **cgroup 配置**动态分配 CPU 核心
- 执行完毕后立即回收 CPU 配额

关键实现细节：
- **弹性并行度**（Elastic DoP）：对于支持并行执行的 action（如编译、测试），可以动态调整并行进程数
- **NUMA 感知的核心选择**：在同一节点内优先分配同一 NUMA 域的核心，减少跨域通信开销

#### GPU Manager：Evict-on-Execution（EOE）

GPU 资源管理更复杂——模型加载到 GPU 需要数十秒，不能像 CPU 那样即用即分。EOE 策略：

- 所有 GPU 服务预先部署，状态备份到 **CPU 内存**
- GPU 服务分为"驻留态"（在 GPU 上可直接执行）和"驱逐态"（仅在 CPU 内存中保存 checkpoint）
- 当 action 到来时：如果目标服务在 GPU 上 → 直接执行；否则 → 从 CPU 内存恢复到 GPU → 执行
- 采用 **LRU 驱逐策略**减少服务抖动

GPU 分配的核心挑战是**拓扑感知**。多卡推理需要高速互联（NVLink），论文设计了 **Multi-Level Cell 结构**：

- 将 GPU 组织为 $\{1, 2, 4, 8\}$ 张卡的 chunk
- 用动态规划求解最优分配方案，目标是在满足拓扑约束的前提下最大化资源利用率
- DP 状态用位图表示每层 chunk 的占用情况，避免组合爆炸

---

## 实验评估

### 实验配置

| 集群 | 配置 |
|:---|:---|
| GPU 训练集群 | 最多 48 节点，每节点 8 × NVIDIA Hopper GPU，NVLink + RDMA |
| CPU 资源集群 | 15 节点，每节点 256 AMD CPU 核心，2.4 TB 内存 |
| GPU 资源集群 | 5 节点，每节点 8 × 高端 NVIDIA GPU，3 TB CPU 内存 |

RL 框架为 VeRL，支持 sequence-level 和异步 rollout。

### 三个 Workload

| Workload | 模型 | Batch Size | 特点 |
|:---|:---:|:---:|:---|
| AI Coding | Qwen3-32B | 1280 | Sandbox 环境交互（GRPO 算法），类似 SWE-bench |
| DeepSearch | Qwen3-32B | 2048 | 联网搜索 + GPT-OSS 奖励计算（GRPO 算法） |
| MOPD | MiMo-V2-Flash | 2048 | 多任务 RL + 9 个 teacher model 计算 log-prob |

### 端到端性能

**AI Coding**（CPU 密集型）：

- 平均 ACT 降低 **4.3×**
- 环境交互速度提升 **9.0×**，reward 计算提升 **2.8×**
- 训练 step 时长缩短 **1.4×**

**DeepSearch**（API 密集型）：

- 训练 step 时长缩短 **1.5×**

**MOPD**（GPU 密集型）：

- 9 个 teacher model 在基线需要各自 4 GPU（共 36 GPU），ARL-Tangram 只需 **10 GPU**（71.2% 节省）且延迟相当
- 当 MOPD + DeepSearch 混合运行时优势更明显——跨 task 的 GPU 资源池化大幅减少空闲

### ACT 时间分解

| Workload (Batch) | 执行时长 | 排队时长 | 系统开销 |
|:---|:---:|:---:|:---:|
| AI Coding (1280) | 0.975s | 0.015s | 0.024s |
| AI Coding (1536) | 1.206s | 0.428s | 0.036s |
| MOPD (2048) | 0.621s | 0.081s | 0.201s |
| MOPD (3072) | 0.705s | 15.05s | 0.240s |

可以看出：
- 在资源充足时（batch 1280/2048），排队时长几乎可忽略
- 在资源紧张时（batch 3072），MOPD 的排队时长暴增到 15 秒，此时弹性调度的价值凸显

### 可扩展性

**CPU 可扩展性**：随 batch size 从 256 增到 1536，相比 Kubernetes baseline 的 ACT 改善从 **3.1×** 增长到 **27.7×**。越大的 batch 越能从资源池化中获益。

**GPU 可扩展性**：batch size 2048 时，相比 SGLang baseline 的改善达 **18.1×**。关键是 EOE 策略让多个 reward 服务分时复用少量 GPU。

### 系统开销

- CPU 密集 workload：< **3%** 的系统开销（cgroup 操作非常轻量）
- GPU 密集 workload：约 **25%** 的开销（主要来自 GPU ↔ CPU 内存的模型状态恢复）

### 消融实验：弹性调度 vs 固定并行度

论文比较了弹性调度与两种固定 DoP 策略：

- **DoP=4**（低并行度）：在资源充足时浪费、在 action 密集期排队严重
- **DoP=16**（高并行度）：在资源紧张时更多 action 被迫排队
- **弹性调度**：在 action 稀疏期自动扩大单 action 资源分配加速完成，在 action 密集期自动压缩以提高并发

弹性调度在整个训练过程中持续优于固定策略，体现了"按需分配"的优势。

---

## 思考与讨论

**从"独占资源"到"按需拼装"的范式转变**。这篇论文的核心洞察很朴素：agentic RL 的外部资源调用是脉冲式的，但现有系统按"持续占用"来管理。ARL-Tangram 所做的，本质上就是把操作系统的时分复用思想搬到了 RL 训练场景中。Container 的 cgroup 操作、GPU 的 checkpoint/restore，都是成熟的基础设施能力，论文的贡献在于识别到这些能力在 agentic RL 场景下的系统性应用机会。

**Tangram 命名的精妙之处**。七巧板的核心是用固定的几何碎片拼出无数形状。ARL-Tangram 用固定的资源池拼装出适配不同 action 需求的资源组合，action 执行完毕资源碎片回池——这个类比非常贴切。

**GPU 的 EOE 策略是一个有趣的工程 trade-off**。25% 的恢复开销看似不低，但考虑到替代方案是为每个 reward 服务独占 GPU（71.2% 的资源浪费），这笔帐算得过来。论文没有详细讨论的一个问题是：随着模型规模增大，GPU ↔ CPU 的状态迁移时间会否成为瓶颈？对于十亿级 reward model，3 TB CPU 内存可能不够缓存所有服务的状态。

**FCFS 调度策略的选择值得注意**。论文明确选择了 FCFS 而非 SJF（最短作业优先），理由是防止长 action 被饿死。这在 agentic RL 中尤其重要——如果某个 trajectory 的一个 action 被长时间延迟，会导致整个 trajectory 的后续 action 全部阻塞，进而影响该 trajectory 的 reward 信号质量。这个设计决策体现了对 RL 训练语义的理解，而非简单套用调度理论的最优解。

**与 ByteScale 的对比视角**。有趣的是，这篇论文和之前读的 ByteScale 论文都在解决 RL 训练中的资源效率问题，但切入角度完全不同：ByteScale 优化的是 **LLM 训练内部**的 GPU 间并行（DP/CP 维度），ARL-Tangram 优化的是 **训练外部**的异构资源调用。两者是互补而非竞争关系——你可以在 GPU 集群上用 ByteScale 做高效的 rollout generation，同时用 ARL-Tangram 管理 sandbox 和 reward model 的资源。

**弹性调度的局限性**。论文假设每个 action 至多有一种关键弹性资源，这简化了调度问题但可能不适用于未来更复杂的场景——比如一个 action 同时需要弹性 CPU（编译）和弹性 GPU（推理），此时多维弹性的联合优化可能更有意义。此外，弹性函数 $E(m)$ 需要离线 profiling 获取，如果工作负载特征频繁变化，profiling 的成本和准确性也是潜在问题。

**生产部署的说服力**。论文明确指出系统已用于小米 MiMo 系列模型的 RL 训练，而非仅在实验环境验证。这增加了可信度——生产环境中的 corner case 远多于实验环境，能上线说明系统的鲁棒性经过了实际考验。
