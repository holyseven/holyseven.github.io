---
layout: post
title: "【论文笔记】AgentBench：系统评估 LLM 作为智能体的能力"
date: 2026-04-17
categories: [paper-notes]
tags: [LLM, agent, benchmark, evaluation, interactive-environment]
paper_title: "AgentBench: Evaluating LLMs as Agents"
paper_authors: "Xiao Liu et al."
paper_link: "https://arxiv.org/abs/2308.03688"
---

> **论文**：[AgentBench: Evaluating LLMs as Agents](https://arxiv.org/abs/2308.03688)
> **作者**：Xiao Liu, Hao Yu, Hanchen Zhang, Yifan Xu, Xuanyu Lei, Hanyu Lai, Yu Gu, Hangliang Ding, Kaiwen Men, Kejuan Yang, Shudan Zhang, Xiang Deng, Aohan Zeng, Zhengxiao Du, Chenhui Zhang, Sheng Shen, Tianjun Zhang, Yu Su, Huan Sun, Minlie Huang, Yuxiao Dong, Jie Tang
> **机构**：Tsinghua University, The Ohio State University, UC Berkeley

传统 LLM benchmark 大多测的是"给一个输入，产生一个输出"的静态能力。但当 LLM 被用作自主 agent 时，它需要在多轮交互中持续推理、做决策、执行动作、根据环境反馈调整策略——这些能力几乎没有被系统评估过。AgentBench 是第一个针对 LLM-as-Agent 的多维度 benchmark，覆盖 8 个交互环境，测试了 29 个 LLM。

## Benchmark 设计

AgentBench 将 LLM-as-Agent 的评估形式化为一个 Partially Observable Markov Decision Process（POMDP），包含状态空间、动作空间、转移函数、奖励函数、任务指令空间和观察空间。简单来说：模型收到任务描述和环境观察，输出一个动作，环境执行后返回新的观察，如此反复直到任务完成或达到轮数上限。

8 个环境分为三大类：

- **Code-grounded**（需要编码/命令能力）：Operating System (OS)、Database (DB)、Knowledge Graph (KG)
- **Game-grounded**（需要策略/推理/常识）：Digital Card Game (DCG)、Lateral Thinking Puzzles (LTP)、House Holding (HH)
- **Web-grounded**（需要网页交互能力）：Web Shopping (WS)、Web Browsing (WB)

![Figure 3](/assets/images/agentbench/x3.png)
*Figure 2: AgentBench 的 8 个环境示意。Code-grounded 环境测试编码与工具使用能力，Game-grounded 环境测试策略推理与常识，Web-grounded 环境测试网页导航与决策能力。*

所有环境都使用最基础的 Chain-of-Thought（CoT）prompting，不使用 self-consistency、reflection、tree-of-thought 等高级策略，也不允许多次尝试。这样测得的是 LLM 最原始的 agent 能力下界。

## 评估设置

测试集共 1,014 个样本，对应约 11K 次推理调用（与 MMLU 的规模相当）。所有模型使用 temperature=0（greedy decoding）。

总分计算方式：先对每个任务的平均分做归一化（除以所有模型在该任务上的均分），再取 8 个任务的加权平均。这样避免某些天然高分任务（如 Web Shopping）主导总分。

## 主要发现

![Figure 1](/assets/images/agentbench/x2.png)
*Figure 1(b): 29 个 LLM 在 AgentBench 上的总分（OA score）。虚线分别为 API 模型和开源模型的平均分。商业 API 模型整体远超开源模型。*

### 商业模型 vs 开源模型的巨大差距

| 模型 | 类型 | OA | OS | DB | KG | DCG | LTP | HH | WS | WB |
|------|------|-----|------|------|------|------|------|------|------|------|
| gpt-4 (0613) | API | 4.01 | 42.4 | 32.0 | 58.8 | 74.5 | 16.6 | 78.0 | 61.1 | 29.0 |
| claude-3 opus | API | 3.11 | 22.9 | 51.7 | 34.6 | 44.5 | 14.3 | 70.0 | 27.9 | 26.0 |
| gpt-3.5-turbo | API | 2.32 | 32.6 | 36.7 | 25.9 | 33.7 | 10.5 | 16.0 | 64.1 | 20.0 |
| codellama-34b | OSS | 0.96 | 2.8 | 14.0 | 23.5 | 8.4 | 0.7 | 4.0 | 52.1 | 20.0 |
| vicuna-13b | OSS | 0.93 | 10.4 | 6.7 | 9.4 | 0.1 | 8.0 | 8.0 | 41.7 | 12.0 |
| llama-2-70b | OSS | 0.78 | 9.7 | 13.0 | 8.0 | 21.3 | 0.0 | 2.0 | 5.6 | 19.0 |

API 模型平均 OA 为 2.32，开源模型平均仅 0.51，差距约 4.5 倍。gpt-4 在 8 个任务中的 6 个拿到最高分。开源模型在 KG、DCG、HH 等任务上普遍接近零分。

### 失败模式分析

论文将 agent 执行结果分为五类：Complete（正常完成）、Context Limit Exceeded (CLE)、Invalid Format (IF)、Invalid Action (IA)、Task Limit Exceeded (TLE)。

| 环境 | Complete | IF | IA | TLE |
|------|----------|-----|-----|------|
| OS | 75.0% | 0.0% | 0.9% | 23.9% |
| DB | 37.9% | 53.3% | 0.0% | 8.0% |
| KG | 30.1% | 0.0% | 0.0% | 67.9% |
| HH | 13.1% | 0.0% | 64.1% | 22.1% |

主要失败原因是 TLE（Task Limit Exceeded），即模型在有限轮次内无法解决问题，反映了长程推理和决策能力的不足。DB 和 DCG 中 Invalid Format 错误频繁，说明模型难以严格遵循格式要求。HH 中 Invalid Action 占比最高（64.1%），因为模型经常生成预定义动作空间之外的动作。

![Figure 6](/assets/images/agentbench/x5.png)
*Figure 6: 不同模型的执行结果类型分布。Invalid Format、Invalid Action 和 TLE 是主要失败类型。*

### 代码训练的双刃剑效应

对比 CodeLlama 和 Llama-2 系列：代码训练在流程化任务（如 Web Shopping）上有优势，但在需要灵活思维的任务（如 Digital Card Game、OS 交互）上反而有害。这提示代码训练可能强化了过程式思维模式，但削弱了开放式推理能力。

### 高质量对齐数据的价值

vicuna-13b（基于 ShareGPT 的 gpt-4/gpt-3.5 数据对齐）在 AgentBench 上优于同基座的 llama-2-13b（从头对齐），甚至可比参数量 3 倍的 codellama-34b。这说明高质量对齐数据对 agent 能力的提升效果，可以超过单纯增大模型规模。

## 思考与讨论

AgentBench 发布于 2023 年 8 月，当时开源模型在 agent 任务上的表现确实远不如商业 API 模型。到 2025-2026 年，这个差距已经大幅缩小，但 AgentBench 识别的核心瓶颈——长程推理、格式遵循、动作空间约束——至今仍是 LLM agent 的主要挑战。

一个自然的延伸方向是：AgentBench 目前只测试单次 CoT 的 agent 能力，没有涉及 reflection、planning、tool learning 等高级策略。如果加入这些策略，模型间的排序是否会改变？底层推理能力弱的模型能否通过更好的策略弥补差距？

另外，benchmark 中 8 个任务的权重是通过所有模型均分的倒数来确定的，这意味着权重会随评估模型集合的变化而变化。论文通过固定初始一批模型的权重来缓解这个问题，但这种设计在模型能力普遍提升后可能需要重新校准。

---

## 8 个环境的交互方式与评估方式总览

### 交互方式分类

8 个环境的交互方式可以分为三类：

**开放动作空间**——Agent 自由生成动作内容，不限于预定义选项：
- **OS**：任意 bash 命令
- **DB**：任意 SQL 语句
- **KG**：从 7 种工具中选择，参数需要自行构造（实体名、关系名等来自环境反馈）

**结构化动作空间**——Agent 从有限选项中选择，或按固定格式填入字段：
- **DCG**：固定 JSON 字段（pick_fish / action / target_position）
- **HH**：从环境给出的动作列表中选择（go to X, take Y, put Z...）
- **WS**：两种固定格式 `search[keywords]` 或 `click[value]`，click 的 value 必须来自可用按钮列表
- **WB**：多选题 A/B/C/D + "None of the above"，候选元素由 DeBERTa 预筛选

**对话式交互**——没有环境操作，纯问答推理：
- **LTP**：只能提出可用 Yes/No/Irrelevant 回答的问题

开放动作空间的三个环境恰好都属于 Code-grounded 类别。它们对格式遵循的要求反而更低（只要输出合法命令即可），但对推理能力要求更高。结构化动作空间看似简单，但论文的失败分析显示 DCG 和 DB 中 Invalid Format 错误频繁，HH 中 Invalid Action 占比高达 64.1%——说明很多模型连"从给定选项中选一个"都做不好。

所有 8 个环境都统一要求模型先输出思考过程（Think/Thought），再输出动作，CoT prompting 是贯穿全部环境的交互范式。每个环境都有最大交互轮数限制（OS 8轮、DB 5轮、KG 15轮、LTP 25轮、HH ~35轮等），超出即判定为 Task Limit Exceeded。

### 评估方式分类

| 评估范式 | 环境 | 特点 | 是否使用 LLM 判定 |
|---------|------|------|------------------|
| 二元成功率 | OS, DB, HH | 对/错，没有中间状态 | 否，纯规则 |
| 集合/连续匹配 | KG, WS, WB | 允许部分正确，能区分"接近正确"和"完全跑偏" | 否，纯规则 |
| 过程导向的复合指标 | DCG, LTP | 不只看最终结果，也看过程质量 | LTP 是，DCG 否 |

具体来说：

- **纯规则评估（7 个环境）**：OS 通过脚本链检查 exit code；DB 对 SELECT 做精确匹配、对 INSERT/UPDATE 比较表 hash；KG 比较实体集合计算 F1；DCG 统计胜负和伤害值；HH 由模拟环境判定任务是否完成；WS 用属性匹配公式计算连续 reward；WB 比较 HTML 元素 ID 和操作类型。这些评估结果完全确定，不依赖任何模型判断。

- **LLM-as-Judge 评估（仅 LTP）**：LTP 是 8 个环境中唯一使用 LLM 作为评判者的。Host 角色由 gpt-3.5-turbo 扮演，负责回答 Solver 的问题（判断 Yes/No/Irrelevant），以及判定 Solver 的推理是否覆盖了真相的 key points。论文与人工评估对比后发现，自动评估在 Single Game Accuracy 和 Query Relevance 两个指标上偏宽松（尤其对开源模型），但在 Game Progress 和 Round Efficiency 上客观性接近人工水平。

AgentBench 在评估设计上尽量采用确定性的规则评估，避免评估本身引入额外噪声。LTP 因为任务性质（开放式推理问答）难以完全规则化，是唯一的例外。

---

## 附录：AgentBench 各环境详细介绍

### 1. Operating System (OS)

**环境与任务**：Agent 在一个真实的 Ubuntu Docker 容器中工作，通过 bash 命令与操作系统交互。每个测试样本包含：一段自然语言任务描述、一个预配置好的 Docker 镜像（可能有特定的文件/用户/系统状态）、一个初始化脚本（在交互前设置环境），以及一个 checking pipeline（用于验证结果的脚本序列）。

任务分为两类：
- **Question Answering (QA)**：需要通过命令获取特定信息，最终提交一个答案（如某个目录下的文件数量、某个配置的值）
- **Operation**：需要对系统状态做出可验证的修改（如更改文件权限、创建用户），不需要提交文本答案，由 checking pipeline 直接检查系统状态

数据来源方面，约一半的测试样本由人工从 Stack Overflow 上的高赞 `bash`/`shell` 问题改编——先从 6000 个真实问题中按点赞数排序，由 8 名编程专业标注者筛选出有挑战性的问题，为每个问题编写任务描述、初始化脚本和验证脚本，每个问题的标注耗时约 2 小时。另一半为 QA 类问题，由 gpt-4 生成并经过严格的单元测试过滤（执行示例代码 → 运行 checking pipeline → 只保留正确的样本）。最终得到 144 个高质量测试样本，每题最多 8 轮交互。

**交互流程**：Agent 每一轮有两种动作可选：`bash`（执行一段 bash 命令，环境返回标准输出）或 `commit`（提交最终答案或宣告任务完成）。交互以 1-shot CoT 格式进行，要求模型先输出 Think（思考过程），再输出 Act（动作类型和内容）。如果命令输出过长会被截断，模型需要自行处理这种情况。

**示例**：

> **系统 prompt（节选）**："You are an assistant that will act like a person, I'll play the role of linux (ubuntu) operating system. Your goal is to implement the operations required by me or answer to the question proposed by me. For each of your turn, you should first think what you should do, and then take exact one of the three actions: 'bash', 'finish' or 'answer'."
>
> **1-shot 示例**：
>
> USER: Tell me how many files are in the directory "/etc"?
>
> AGENT: Think: To count the files in /etc, I need to print all the files in it. Act: bash `ls /etc`
>
> USER: The output of the OS: cpi cron.hourly fuse.conf iproute2... [truncated because the output is too long]
>
> AGENT: Think: The output has been truncated because it is too long, so I need to count files by script directly. Act: bash `ls -1 /etc | wc -l`
>
> USER: The output of the OS: 220
>
> AGENT: Think: Now I get the answer, it is 220. Act: answer(220)
>
> **实际测试问题**："Find the total number of non-empty directories inside the '/etc' directory."

这个示例展示了 OS 环境的几个关键特点：(1) 输出可能被截断，模型需要调整策略；(2) 需要多步操作才能得到答案；(3) 动作空间是开放的（任意合法 bash 命令），没有预定义选项。

**评估方式**：Success Rate（成功率）。每个问题对应一个 checking pipeline $f_1, f_2, \cdots, f_n$，脚本链式执行，前面脚本的输出作为后面脚本的输入参数，所有脚本的 exit code 均为 0 才算成功。QA 类和 Operation 类的区别仅在于 checking pipeline 的内容——前者检查提交的答案，后者检查 Docker 容器中的系统状态。

---

### 2. Database (DB)

**环境与任务**：Agent 通过 SQL 命令与一个真实的 MySQL 数据库交互。每个测试样本包含：一段自然语言的任务描述、表的元信息（表名和列名的含义说明）、表的实际内容，以及正确答案。数据库运行在 Docker 容器中，通过端口转发提供交互。

任务分为三类：
- **SELECT（查询）**：需要执行查询并提交结果，查询进一步细分为 Counting、MIN/MAX/AVG/SUM 聚合、Ranking、Comparison 等子类型
- **INSERT（插入）**：按照自然语言描述向表中插入新行
- **UPDATE（更新）**：按照自然语言描述修改表中已有数据

数据源自 WikiSQL、WikiTableQuestions、SQA、HybridQA、FeTaQA 等已有数据集。为了避免数据泄露并增加多样性，用 gpt-3.5-turbo 做了数据增强：给定表的 header 和原始行，生成 10 条新行和 5 条新 SQL 查询，再将 SQL 改写为自然语言描述（确保语义不变但表述多样化）。所有增强数据都经过单元测试验证。最终得到 300 个测试样本，每题最多 5 轮交互。

**交互流程**：Agent 每轮需要输出思考过程和一条 SQL 命令（格式为 `Action: Operation` + SQL 代码块），环境返回 MySQL CLI 的原始输出。当 Agent 认为任务完成时，输出 `Action: Answer` 并提交最终答案。如果 Agent 的输出格式不匹配任何预定义的 pattern，直接判定为失败。

**示例**：

> **系统 prompt（节选）**："I will ask you a question, then you should help me operate a MySQL database with SQL to answer the question. You have to explain the problem and your solution to me and write down your thoughts. After thinking and explaining thoroughly, every round you can choose to operate or to answer."
>
> **操作格式**：
> ```
> Action: Operation
> ```sql
> SELECT * FROM table WHERE condition;
> ```
> ```
>
> **提交答案格式**：
> ```
> Action: Answer
> Final Answer: ["ANSWER1", "ANSWER2", ...]
> ```
>
> **具体示例**：给定 'Olympic Medals' 表，任务为 "What was the total number of medals won by United States?"
>
> Agent 执行 `SELECT SUM(total) FROM olympic_medals WHERE country = 'United States';` → 环境返回查询结果 → Agent 提交 Final Answer

每轮只能执行一条 SQL 语句，环境只执行第一个 SQL 代码块中的内容。模型需要自行处理 MySQL 的原始输出格式。

**评估方式**：Success Rate，三类任务分别计算成功率后取宏平均。SELECT 类：将提交的答案与标准答案做精确匹配，忽略元素顺序，数值允许等价表示（如 5、"5.0"、'+5' 被视为相同）。INSERT 和 UPDATE 类：对操作后的整张表计算 hash 值，与正确操作后的表 hash 比较——这意味着只要表的最终状态正确即可，不限制 Agent 使用何种 SQL 路径达到目标。论文还用 Claude-2 重新标注了一小批数据，验证了 gpt-3.5-turbo 增强的数据不引入模型偏差。

---

### 3. Knowledge Graph (KG)

**环境与任务**：Agent 在 Freebase 知识图谱（4500 万实体、30 亿条事实）上进行多步推理以回答复杂的自然语言问题。这个环境最核心的特点是**部分可观察性**——KG 远大于任何 prompt 能容纳的范围，Agent 无法一次看到全貌，只能通过工具一步步探索。Agent 面对的是一个类似于"在黑暗中摸索"的场景：你不知道 KG 里有什么关系和实体，需要先调用工具看看有哪些可用路径，再决定走哪条。

KG 后端使用 Virtuoso 托管 Freebase 的最新版本。由于 SPARQL 查询过于复杂，论文没有要求 LLM 直接写 SPARQL，而是封装了 7 种 API 工具：

| 工具 | 功能 | 示例 |
|------|------|------|
| `get_relations(var)` | 列出与某实体/变量相连的所有关系 | `get_relations(Barack Obama)` → [people.person.profession, ...] |
| `get_neighbors(var, relation)` | 沿指定关系获取相邻实体集合 | `get_neighbors(Obama, people.person.profession)` → variable #0 |
| `intersection(var1, var2)` | 求两个实体集合的交集 | `intersection(#0, #1)` → variable #2 |
| `get_attributes(var)` | 列出实体的所有数值属性 | 用于后续 argmax/argmin |
| `argmax(var, attr)` | 返回某属性值最大的实体 | `argmax(#2, age)` → 最年长的实体 |
| `argmin(var, attr)` | 返回某属性值最小的实体 | 类似 argmax |
| `count(var)` | 返回集合中的实体数量 | `count(#0)` → 整数 |

每次工具调用产生的实体集合被赋予一个编号（#0, #1, #2...），Agent 可以在后续调用中引用这些变量。当 Agent 认为某个变量即为最终答案时，输出 `Final Answer: #id` 结束任务。

数据来自 GrailQA、ComplexWebQuestions、GraphQuestions，通过 S-expression 标注确定每个问题的最优工具调用序列，只保留需要至少 5 次工具调用的难题。最终得到 1,663 个问题，取前 500 个用于评估，每题最多 15 轮。

**示例**：

> **问题**："The bipropellant rocket engine with kerosene and gas-generator cycle is designed by who?"
>
> **给定实体**：[Gas-generator cycle, Kerosene]
>
> **Agent 推理过程**：
>
> Thought: I need to first find engines with gas-generator cycle.
>
> Action: `get_relations(Gas-generator cycle)`
>
> Observation: [spaceflight.rocket_engine_cycle.rocket_engines, automotive.fuel.engines, ...]
>
> Thought: I'll use spaceflight.rocket_engine_cycle.rocket_engines to find engines.
>
> Action: `get_neighbors(Gas-generator cycle, spaceflight.rocket_engine_cycle.rocket_engines)`
>
> Observation: variable #0, which are instances of spaceflight.bipropellant_rocket_engine
>
> Thought: I also need to find engines with kerosene.
>
> Action: `get_relations(Kerosene)` → ... → `get_neighbors(Kerosene, spaceflight.rocket_engine_cycle.rocket_engines)` → variable #1
>
> Thought: The intersection of #0 and #1 gives engines with both properties.
>
> Action: `intersection(#0, #1)` → variable #2
>
> Thought: Now I need to find who designed these engines.
>
> Action: `get_relations(#2)` → [spaceflight.rocket_engine.designed_by, ...]
>
> Action: `get_neighbors(#2, spaceflight.rocket_engine.designed_by)` → variable #3
>
> Final Answer: #3

这个示例完整展示了 KG 环境的典型推理链：先分别从两个条件出发探索 KG，用 intersection 取交集缩小范围，再沿新关系找到最终答案。整个过程需要 7 次工具调用。零样本设定下 LLM 几乎无法产出有效输出，因此使用了 1-shot 示例。

**评估方式**：主要指标为 **F1 score**——将预测答案集合与标准答案集合做比较，计算 precision 和 recall 的调和平均。另外报告 **Exact Match**（预测集合与标准集合完全一致的比例）和 **Executability**（Agent 的动作序列执行后是否能产生任何结果，即便结果不正确——用于衡量 Agent 是否至少学会了正确使用工具）。

---

### 4. Digital Card Game (DCG)

**环境与任务**：使用 Aquawar 卡牌对战系统（来自清华 Agent 竞赛 THUAC 2021），这是一个信息不对称的回合制策略对战游戏。LLM 控制一支 4 条鱼的队伍，与算法对手进行 PvE 对战。

游戏基本规则：
- **卡池**：共 10 种鱼可选（Spray、Flame、Eel、Sunfish、Barracuda、Mobula、Octopus、Whiteshark、Hammerhead 等），双方各选 4 条组队
- **初始属性**：每条鱼 400 HP、200 攻击力，各有一个主动技能和一个被动技能
- **普通攻击**：造成攻击力 50% 的伤害
- **主动技能**：包括 AOE（对所有敌人造成攻击力 35% 的伤害）、Infight（对一个队友造成 75 伤害但自身攻击力 +140）、Crit（对生命最低的敌人造成 120% 暴击伤害）、Subtle（减伤 70% 并增加攻击力）等
- **被动技能**：自动触发，包括 Counter（队友生命低于 30% 时反击 30 伤害）、Deflect（70% 伤害分摊给队友，累积受 200 伤害后攻击力 +40）、Reduce（30% 概率闪避伤害）、Heal（被攻击后回复 20 HP）、Explode（被攻击时对攻击者造成 40 伤害）
- **身份隐藏机制**：对手鱼的种类最初是隐藏的。每回合可以猜测一条敌方鱼的身份，猜对则敌方全队各受 50 伤害。不同鱼种之间故意设计了技能重叠（如 Spray 和 Eel 共享 AOE 技能），增加猜测难度
- **胜利条件**：游戏结束时存活鱼更多的一方获胜

游戏分两个阶段测试：Stage 1 移除了猜测机制（简化版），Stage 2 包含完整的猜测机制。两个阶段的得分取平均。对手策略有两种：随机策略（从合法动作中随机选择）和贪心策略（优先使用 AOE，评估能否一击必杀，再考虑主动技能，最后普通攻击）。共 20 个测试样本（固定的对战场景以保证公平性），每样本约 30 轮。

**交互流程**：每轮 Agent 需要输出 JSON 格式的动作。Action 阶段的格式为：

```json
{"pick_fish": "选择的鱼名", "action": "normal 或 active", "target_position": "目标位置 0-3"}
```

Guess 阶段的格式为：

```json
{"guess_type": "猜测的鱼种", "target_position": "目标位置 0-3"}
```

如果 Agent 连续 5 次无法输出合法格式的动作，直接判定为战败。

**示例**：

> **环境信息**：你的队伍是 Spray（被动 Counter / 主动 AOE）、Flame（被动 Counter / 主动 Infight）、Eel（被动 Deflect / 主动 AOE）、Sunfish（被动 Deflect / 主动 Infight）。对手的四条鱼身份未知。
>
> **当前状态**：你的 Spray HP=320, Flame HP=400；对手 Fish0 HP=280, Fish1 HP=400, Fish2 HP=350, Fish3 HP=400
>
> **Agent 决策**：Think: Fish0 的 HP 最低，我应该集中火力。Spray 的 AOE 能打所有人但伤害较低，不如用 Flame 的 Infight 先提升攻击力为后续输出做准备。
>
> 输出：`{"pick_fish": "flame", "action": "active", "target_position": 2}`（对队友 Eel 释放 Infight，牺牲 75 HP 换取 +140 攻击力）

这个环境的难点在于：(1) 需要在不完全信息下进行策略推理；(2) 技能之间有复杂的交互效果；(3) 需要平衡短期收益与长期战略；(4) 必须严格遵循 JSON 输出格式。

**评估方式**：综合 Reward = 0.7 × Win Rate + 0.3 × Damage Rate。Win Rate 是胜局数占总局数的比例。Damage Rate 是 Agent 造成的总伤害与对方全队总 HP 的比值——即使输了也能通过造成较多伤害获得部分分数，这使得评估更能区分"完全不会玩"和"有一定策略但不够好"的模型。Stage 1（无猜测）和 Stage 2（有猜测）的平均分作为最终得分。

---

### 5. Lateral Thinking Puzzles (LTP)

**环境与任务**：海龟汤游戏（Situation Puzzle）。这是一个纯推理任务，没有工具使用或环境操作，完全依赖 Agent 的推理和提问策略。游戏中有两个角色：**Host**（主持人，知道谜面和真相）和 **Solver**（解谜者，由 LLM 扮演）。

游戏流程：
1. Host 给出一个谜面（story），通常是一个看起来不合逻辑的简短场景
2. Solver 只能提出可以用 "Yes"、"No" 或 "Irrelevant" 回答的问题来逐步推理真相
3. Solver 不能直接问 "why"，不能要求 Host 透露真相细节，必须自己做出猜测
4. 当 Solver 认为已经猜出真相的主要部分时，可以向 Host 陈述自己的推理，Host 判断是否正确
5. 如果 Solver 推理方向偏离太远，Host 会适当给出提示（但不直接透露真相）

数据分为 easy/medium/hard/expert 四个难度级别。每个样本包含一对 story（谜面）和 truth（真相）。真相被 gpt-3.5-turbo 分解为若干 key points（关键点），数量因样本而异。共 50 个测试样本，每题最多 25 轮。

评估中，Host 由 gpt-3.5-turbo 自动扮演。为确保 Host 理解规则，prompt 中包含了多轮预热对话：先让 Host 复述 story 和 truth 的要点，再让 Host 复述规则，之后才开始正式游戏。

**示例**：

> **谜面**："A man sleeps with the lights off, and the next morning he suicides after opening windows. Why?"
>
> **真相**：这个男人是灯塔看守员。他忘记开灯塔的灯就睡了。第二天早上打开窗户，看到海面上有船只撞毁的残骸，意识到是因为自己没开灯导致船只失事，于是自杀。
>
> **真相的 key points**：(1) 男人是灯塔看守员 (2) 他忘记开灯塔的灯 (3) 船只因此撞毁 (4) 他看到残骸后感到自责
>
> **交互过程**：
>
> Solver: "Was the man alone?" → Host: "Yes"
>
> Solver: "Did the man see something outside the window that made him upset?" → Host: "Yes"
>
> Solver: "Was the man responsible for what he saw?" → Host: "Yes"
>
> Solver: "Was he a lighthouse keeper who forgot to turn on the light?" → Host: "Yes! He saw ships crashed because the lighthouse light was off."

Game Progress 的计算方式：Agent 的每个被 Host 回答 "Yes" 的问题被转换为陈述句（如 "Was he a lighthouse keeper?" → "He was a lighthouse keeper"），然后与每个 key point 做语义比对（由 gpt-3.5-turbo 判断陈述句是否包含了 key point 的全部信息）。猜中一个 key point 后该点从列表中移除，避免重复计分。

**评估方式**：四个指标，其中 **Game Progress（GP）** 为主要指标：

| 指标 | 含义 |
|------|------|
| Game Progress (GP) | 真相的 key points 被猜中的比例（如 5 个 key points 猜中 3 个则 GP=60%） |
| Single Game Accuracy (SGA) | 单局游戏中，每轮提问中"接近真相"的轮次占比 |
| Round Efficiency (RE) | 猜出真相所需的轮数，越少越好 |
| Query Relevance (QR) | 每轮提问与真相的相关性 |

论文通过人工验证评估了自动评估系统的准确性，发现自动评估在 SGA 和 QR 上偶尔比人工评估更宽松（尤其对开源模型），但在 GP 和 RE 上的客观性与人工评估接近。

---

### 6. House Holding (HH / ALFWorld)

**环境与任务**：基于 ALFWorld 文本游戏环境的家居操作任务。Agent 置身于一个模拟的家庭环境中（浴室、厨房、卧室等），需要理解高层任务指令，将其分解为一系列具体动作，逐步完成目标。环境在每步提供当前位置的描述和可用动作列表，Agent 需要从中选择。

这个环境的自由度很高——家庭场景中有大量实体（cabinet、countertop、sinkbasin、fridge、microwave 等），每个实体可能包含多个物品（soapbar 1、cloth 2、spraybottle 3 等），可用动作也很多（go to、take、put、clean、heat、cool、examine、open 等）。Agent 必须在众多选项中保持清晰的推理链条。

任务分六类：

| 任务类型 | 说明 | 示例 |
|---------|------|------|
| pick and place | 拿起物品放到指定位置 | "Put some spraybottle on toilet" |
| pick clean then place | 拿起物品 → 清洗 → 放到指定位置 | "Clean some soapbar and put it in countertop" |
| pick heat then place | 拿起物品 → 加热 → 放到指定位置 | "Heat some egg and put it in countertop" |
| pick cool then place | 拿起物品 → 冷却 → 放到指定位置 | "Cool some tomato and put it in countertop" |
| look at obj | 在灯光下检查某物品 | "Look at pen under desklamp" |
| pick two obj | 拿起两个同类物品放到指定位置 | "Put two cellphone in sofa" |

数据来自 ALFWorld 的 eval out of distribution 分集中的 134 个可解问题，采样 50 个用于测试。每题约 35 步。由于任务复杂且格式要求严格，采用 1-shot CoT 评估——每类任务提供一个来自训练集的完整交互示例（使用 ReAct 论文中的 few-shot 格式）。

**交互流程**：每轮 Agent 可以选择两种输出模式：`THOUGHT`（先思考再行动，格式为 "THOUGHT: ... \n ACTION: ..."）或直接 `ACTION`（格式为 "ACTION: ..."）。环境返回动作的反馈结果，如果动作无效则返回 "Nothing happened"。

**示例**（论文中的 1-shot 示例，展示完整交互链）：

> **环境**：你在浴室中间，看到 cabinet 1-4, countertop 1, garbagecan 1, handtowelholder 1-2, sinkbasin 1-2, toilet 1, toiletpaperhanger 1, towelholder 1
>
> **任务**："Put some spraybottle on toilet."
>
> AGENT: THOUGHT: The task is to put some spraybottle on toilet, so first I need to search the room for some spraybottle. I will start my search on cabinets.
> ACTION: go to cabinet 1
>
> USER: On the cabinet 1, you see a cloth 1, a soapbar 1, a soapbottle 1.
>
> AGENT: ACTION: go to cabinet 2
>
> USER: The cabinet 2 is closed.
>
> AGENT: ACTION: open cabinet 2
>
> USER: You open the cabinet 2. The cabinet 2 is open. In it, you see a candle 1, and a spraybottle 2.
>
> AGENT: ACTION: take spraybottle 2 from cabinet 2
>
> USER: You pick up the spraybottle 2 from the cabinet 2.
>
> AGENT: ACTION: go to toilet 1
>
> USER: On the toilet 1, you see a soapbottle 2.
>
> AGENT: ACTION: put spraybottle 2 in/on toilet 1
>
> USER: You put the spraybottle 2 in/on the toilet 1. ✓

这个示例展示了典型的推理链：先确定目标物品在哪里（遍历 cabinet），找到后拿起，再导航到目标位置放下。更复杂的任务（如 pick clean then place）需要在中间增加"去 sinkbasin 清洗"的步骤。

**评估方式**：Success Rate——在最大步数内完成整个任务流程即为成功。有两个特殊处理：(1) 如果 Agent 连续 3 次输出完全相同的内容，判定为"重复失败"，提前终止；(2) 如果 Agent 的输出格式无效（不在预定义动作空间中），用 BLEU 分数找到与输出最相似的合法动作作为替代——这比直接判定失败更宽容，但也意味着模型可能"蒙对"一些动作。

---

### 7. Web Shopping (WS / WebShop)

**环境与任务**：基于 WebShop 模拟电商平台。这个环境模拟了一个完整的网购流程：从搜索商品、浏览列表、查看详情页到选择规格和下单。环境的后端数据库包含约 100 万件从 Amazon 抓取的真实商品，每件商品都标注了属性标签。人工编写了 12,087 条购物指令，每条指令关联了目标商品的期望属性（如类别、颜色、尺寸、价格上限等）。

与直接操作 HTML 不同，WebShop 将网页简化为文本观察——Agent 看到的是结构化的文本描述（商品标题、价格、可选项等），以及当前可用的交互按钮列表。

动作空间只有两种：
- **search[keywords]**：在搜索框中输入关键词进行搜索，环境返回搜索结果页面
- **click[value]**：点击页面上的某个按钮（如商品链接、尺寸选项、"Buy Now" 等），环境返回点击后的新页面

交互以 1-shot CoT 格式进行。取前 500 条指令中的 200 个作为测试集，每题约 5 轮。

**示例**（论文中的 1-shot 示例，展示完整购物流程）：

> **初始页面**：WebShop [SEP] Instruction: i need a long lasting 6.76 fl oz bottle of l'eau d'issey, and price lower than 100.00 dollars [SEP] Search
>
> AGENT: Thought: I think I should use the search bar to look for the product I need.
> Action: search[l'eau d'issey 6.76 fl oz bottle price < 100.00]
>
> **搜索结果页**：Page 1 (Total results: 50) [SEP] B000VOHH8I [SEP] L'eau D'issey By Issey Miyake for Men Eau De Toilette Spray, 6.7 Fl Oz Bottle [SEP] $64.98 [SEP] B000MJZOPK [SEP] L'eau d'Issey by Issey Miyake for Women 3.3 oz ... [SEP] $49.98 [SEP] ...
>
> AGENT: Thought: I think I should click on the product I need, which is B000VOHH8I.
> Action: click[B000VOHH8I]
>
> **商品详情页**：... size [SEP] 2.5 fl oz [SEP] 6.76 fl oz (pack of 1) [SEP] L'eau D'issey By Issey Miyake... [SEP] Price: $64.98 [SEP] Rating: N.A. [SEP] Description [SEP] Features [SEP] Reviews [SEP] Buy Now
>
> AGENT: Thought: I think I should click on the '6.76 fl oz (pack of 1)' option to select the size I need.
> Action: click[6.76 fl oz (pack of 1)]
>
> **选择尺寸后**：（页面更新，确认选中了 6.76 fl oz）
>
> AGENT: Thought: I think I should click on the 'Buy Now' button to purchase the product.
> Action: click[Buy Now] ✓

这个流程清晰地展示了 Web Shopping 环境的特点：Agent 需要 (1) 设计有效的搜索关键词，(2) 从多个结果中选择最匹配的商品，(3) 在详情页正确选择规格选项，(4) 最终完成购买。

**评估方式**：Reward（0-1 之间的连续匹配分数），因为可能有多个商品符合要求，所以不用二元对错而用匹配程度。计算公式为：

$$
\text{Reward} = \frac{|U_{att} \cap Y_{att}| + |U_{opt} \cap Y_{opt}| + \mathbb{I}[y_{price} \leq u_{price}]}{|U_{att}| + |U_{opt}| + 1} \cdot r_{type}
$$

其中 $U$ 和 $Y$ 分别代表目标商品和实际购买的商品；$att$ 是属性集合（如品牌、材质）；$opt$ 是选项集合（如颜色、尺寸）；价格约束是一个 0/1 判断。分子是匹配上的属性数 + 匹配上的选项数 + 价格是否在预算内，分母是总属性数 + 总选项数 + 1。

$r_{type}$ 是一个基于文本匹配度的衰减乘子——如果购买的商品标题与目标商品标题的名词/代词/专有名词匹配度（TextMatch）太低，reward 会被大幅衰减：TextMatch=0 时 $r_{type}=0$（完全不匹配），TextMatch<0.1 时 $r_{type}=0.1$，TextMatch≤0.2 且查询和类别都不匹配时 $r_{type}=0.5$，否则 $r_{type}=1$。这个设计防止模型通过随机购买获得高分。

---

### 8. Web Browsing (WB / Mind2Web)

**环境与任务**：基于 Mind2Web 数据集，在真实网站上完成复杂的多步任务。与 WebShop 的模拟电商环境不同，这里面对的是真实世界的网页——涵盖 Travel、Information、Service、Shopping、Entertainment、Housing、Job、Social Media、Education、Health、Government、Home Service 等多个领域，来自 73 个不同网站。

每个任务样本包含：
- **Task Description**：一个高层次的目标描述（非逐步指令），如 "Get the highest rated SAP S/4 HANA course rated 4, and up with a duration between 3 to 6 hours for an intermediate, and add this to your cart and checkout"
- **Reference Action Sequence**：标注者在网站上实际执行任务时记录的交互序列，每步包含目标 HTML 元素的后端 ID ($e_t$) 和操作类型 ($o_t$)——Click、Type 或 Select Options（后两者附带文本值）
- **Webpage Information**：每步对应的网页快照，包含原始 HTML 代码和之前的交互历史

真实网页 HTML 的复杂性是这个环境最大的挑战。一个网页可能包含成千上万个 HTML 元素，直接让 LLM 处理全部 HTML 既不现实也不高效。Mind2Web 的解决方案是**两阶段处理**：

1. **元素过滤**（Element Ranking）：用一个 fine-tuned DeBERTa 模型对网页中的所有 HTML 元素进行排序，根据任务描述和当前上下文选出 top-k（k=10）个最可能相关的候选元素
2. **元素选择**（Element Selection）：将候选元素呈现为多选题（A/B/C/D 四个选项 + "None of the above"），让 LLM 从中选择正确的目标元素，并指定操作类型和文本值

这意味着 LLM 面对的实际上是一个经过简化的决策问题——不需要处理完整的 HTML，只需要从 5 个候选中做出选择。但即便如此，Agent 仍然需要理解简化后的 HTML 片段、任务上下文和之前的操作历史来做出判断。

测试集来自 Mind2Web 的 Cross Domain test set（912 个任务），取 100 个用于评估。每题约 10 步。使用 3-shot CoT prompting。

**示例**（论文中的 few-shot 示例之一）：

> **网页 HTML（简化后）**：
> ```html
> <button id=0 book a reservation. toggle open>
>   <span>Book a reservation</span>
> </button>
> <select id=1 type>
>   <option reservations true>Dine in</option>
>   ...
> </select>
> ```
>
> **Task**: Check for pickup restaurant available in Boston, NY on March 18, 5 pm with just one guest
>
> **Previous actions**: None
>
> **选项**：
> A. None of the above
> B. `<button id=0 book a reservation...>` Book a reservation
> C. `<select id=1 type>` Dine in
> D. `<div id=2>` Celebrating and supporting leading women...
>
> AGENT: Thought: I need to select pickup restaurant first.
> Answer: C.
> Action: SELECT
> Value: Pickup

另一个更复杂的示例展示了"None of the above"的情况：

> **Task**: Compare the fare types to book a 1-adult ticket from Springfield, IL to Austin, TX for April 29th 2023
>
> **Previous actions**: [combobox] Enter your departing city... → TYPE: SPRINGFIELD → [button] Springfield, IL, US (SPI) → CLICK → [combobox] Enter your destination... → TYPE: AUSTIN → [button] Austin, TX, US (AUS) → CLICK
>
> **当前页面候选元素**：都是 Hotel 相关的标签和日期选择器
>
> AGENT: Thought: There are no elements that allow setting the date or viewing the fare, so there is no correct action.
> Answer: A. (None of the above)

这个示例说明 Agent 不仅需要选对元素，还需要判断当前页面是否包含了需要操作的元素——如果 DeBERTa 过滤后的候选中不包含正确元素，Agent 应该选择 "None of the above"。

**评估方式**：三个层次的指标：

| 指标 | 含义 |
|------|------|
| Element Accuracy | Agent 选择的 HTML 元素 $e_t$ 是否与标注的目标元素一致 |
| Action F1 | 操作类型 $o_t$ 的 token 级匹配分数——对 Click 是精确匹配，对 Type 和 Select Options 需要比较文本值 |
| Step Success Rate（主要指标） | 每步独立判断：所选元素正确 **且** 操作类型及文本值正确 |
| Task Success Rate（参考指标） | 一个任务的所有步骤全部正确才算成功 |

论文以 **Step Success Rate** 为主要指标报告结果，因为 Task Success Rate 要求所有步骤全部正确，是一个极其严格的指标——目前即使最好的模型也只能达到个位数百分比的 Task Success Rate。这反映了多步网页操作任务的组合复杂性：即使每步有 80% 的准确率，10 步任务的整体成功率也只有 $0.8^{10} \approx 10\%$。
