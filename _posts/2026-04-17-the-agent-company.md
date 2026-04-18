---
layout: post
title: "【论文笔记】TheAgentCompany：在模拟公司环境中评估 AI Agent 的工作能力"
date: 2026-04-17
categories: [paper-notes]
tags: [agent-benchmark, LLM-agent, workplace-automation, evaluation, LLM-as-Judge]
paper_title: "TheAgentCompany: Benchmarking LLM Agents on Consequential Real World Tasks"
paper_authors: "Frank F. Xu et al."
paper_link: "https://arxiv.org/abs/2412.14161"
---

> **论文**：[TheAgentCompany: Benchmarking LLM Agents on Consequential Real World Tasks](https://arxiv.org/abs/2412.14161)
> **作者**：Frank F. Xu, Yufan Song, Boxuan Li, Yuxuan Tang, Kritanjali Jain, Mengxue Bao, Zora Z. Wang, Xuhui Zhou, Zhitong Guo, Murong Cao, Mingyang Yang, Hao Yang Lu, Amaad Martin, Zhe Su, Leander Maben, Raj Mehta, Wayne Chi, Lawrence Jang, Yiqing Xie, Shuyan Zhou, Graham Neubig
> **机构**：Carnegie Mellon University, Duke University

现有的 agent benchmark 大多聚焦于单一维度的能力——WebArena 测试网页浏览、SWE-bench 测试代码修复、τ-bench 测试客服对话。但真实的工作场景是多维度交织的：一个项目管理任务可能需要你先到 GitLab 查代码仓库状态，再到 Jira 里整理 sprint issue，然后在 Slack 上通知相关同事，最后把报告上传到 Google Drive。TheAgentCompany 搭建了一个模拟的软件公司环境，包含 175 个覆盖软件开发、项目管理、HR、财务、行政等多部门的任务，要求 agent 同时使用浏览器、终端、代码执行和即时通讯来完成工作。最强的模型（Gemini 2.5 Pro）也只能自主完成 30% 的任务。

## 环境设计：一个自包含的虚拟公司

![Figure 1](/assets/images/the-agent-company/x2.png)
*Figure 1: TheAgentCompany 总览。左侧为可复现的自托管环境（含模拟同事、内部网站和终端），中间为 agent 通过浏览器、代码和终端与环境交互，右侧为涵盖 6 个部门的多样化任务，底部为 checkpoint-based 的评估机制。*

TheAgentCompany 的环境由三部分组成：本地工作空间、公司内网（四个自托管服务）和模拟同事系统。

**本地工作空间**：一个 Docker 沙箱环境，模拟员工的工作电脑。Agent 在其中使用 Chromium 浏览器（通过 Playwright 驱动）、Jupyter IPython 环境执行 Python 代码、以及 Linux bash 终端执行系统命令。沙箱隔离保证 agent 的操作不会影响评测机器的其他部分。

**公司内网**由四个开源、可自托管的服务组成，模拟真实软件公司员工日常使用的内部工具：

1. **GitLab**（71 个任务涉及）：开源代码托管平台，对应真实公司中的 GitHub/GitLab。在 benchmark 中托管了 TheAgentCompany 的多个代码仓库（如 JanusGraph、RisingWave、OpenHands、llama.cpp 等项目）以及技术 wiki 页面。SDE 类任务大量使用这个平台，典型操作包括克隆仓库、查看 issue、提交代码、阅读文档等。

2. **OwnCloud**（70 个任务涉及）：开源文件存储和协作编辑平台，对应真实公司中的 Google Drive 或 Microsoft Office 365。用于存储和共享公司内部文档——包括财务报表（CSV）、税表说明（PDF）、员工手册、薪酬结构文档、项目管理模板等。Agent 需要在这个平台上读取文件、上传报告、编辑电子表格。这也是所有平台中 agent 表现最差的——web 端 Office 软件的 UI 极其复杂，连关闭一个欢迎弹窗都会难倒 agent。

3. **Plane**（17 个任务涉及）：开源项目管理工具，对应真实公司中的 Jira 或 Linear。用于跟踪 issue、管理 sprint 周期、查看产品路线图和项目分析数据。PM 类任务主要在这个平台上操作，如整理 sprint 中未完成的 issue、收集项目指标、分配任务等。

4. **RocketChat**（79 个任务涉及）：开源即时通讯工具，对应真实公司中的 Slack 或 Microsoft Teams。这是公司内部的实时消息平台，agent 通过它与模拟同事沟通——发送私信询问信息、在频道中发布通知或分享工作成果。公司设置了多个频道如 `#general`、`#engineering`、`#hr-announcements`、`#kudos` 等，模拟真实的组织沟通结构。RocketChat 涉及的任务数量最多（79 个，占 45%），因为很多跨部门任务都需要与同事交互。

**模拟同事（基于 Sotopia）**：Sotopia 是一个支持用 LLM 创建模拟人类角色的平台。TheAgentCompany 用它创建了一个虚拟公司的完整员工花名册，包括 CTO Sarah Johnson、财务总监 David Wong、HR 经理 Jessica Lee、产品经理 Huang Jie、前端工程师 Jessica Chen 等约 20 名角色。每个角色都有详细的个人档案：姓名、年龄、性别、职位、职责范围、所属项目和加入的 Slack 频道。例如 Sarah Johnson（女，42 岁）是 CTO，负责技术战略和研发团队管理，有权限访问所有技术频道。Agent 通过 RocketChat 的私信或频道消息与这些模拟同事交互。所有模拟同事统一使用 Claude-3.5-Sonnet 作为 backbone——论文在初步实验中发现它在角色扮演方面效果最好。

所有内网服务都预填充了模拟公司的数据：真实的开源项目代码、手工整理的财务数据、员工薪酬表、项目管理记录等。使用开源自托管软件的好处是环境完全可复现——不像依赖第三方平台的 benchmark（如 WorkArena 依赖 ServiceNow）会因为外部服务变化而影响评测结果。

## 任务设计：从美国劳工部数据库到 175 个手工任务

### 任务来源

论文在选择任务时参考了美国劳工部的 O\*NET 数据库（记录了美国各职业及其工作内容的数据库），先按就业人数和薪资中位数筛选出高价值职业类别，再排除需要体力劳动的岗位（如护士），最终聚焦在软件公司场景下的多个角色。

最终的 175 个任务涵盖 7 个部门：软件开发（SDE, 69 个）、项目管理（PM, 28 个）、人力资源（HR, 29 个）、行政（Admin, 15 个）、数据科学（DS, 14 个）、财务（Finance, 12 个）和其他（Other, 8 个）。任务创建全部由人工完成：20 位计算机科学学生、软件工程师和项目经理耗时 2 个多月，共计约 3000 人时。每个任务经过多轮审核——包括截图证明评估器有效、代码 review、以及一轮由未参与创建的人进行的 checkpoint 评分一致性检查。

### 任务结构

每个任务由四部分组成：

- **Task Intent**：用英文描述的任务指令，模拟用户向 agent 下达工作要求。目标是让一个人类员工看到这段描述后，不需要再向上级确认就能开始工作（但可能需要向同事询问信息）。
- **Checkpoints**：将任务分解为若干中间里程碑，每个 checkpoint 有对应的分值。Checkpoint 考察三类能力：
  - Action Completion（动作完成）：是否执行了必需的操作，如导航到指定 URL、使用特定工具、收集数据
  - Data Accuracy（数据准确性）：输出内容是否正确完整，如提取的数据、格式化的文档
  - Collaboration（协作）：是否与模拟同事进行了必要的交互，如发送消息、询问额外信息
- **Evaluator**：每个 checkpoint 对应一个评估程序，通过检查环境状态（本地文件系统、内网服务状态、聊天记录等）或分析 agent 行为轨迹来判断是否完成
- **初始化/清理代码**：配置任务所需的环境状态，以及任务结束后的清理

### 具体任务示例

论文给出了三个不同部门的任务示例：

**SWE 任务——搭建 JanusGraph 并启动 HTTP 服务**

Task Intent：将 JanusGraph 仓库从公司 GitLab 克隆到 `/workspace` 目录，构建二进制文件，然后在本地 8182 端口启动带 HTTP endpoint 的 JanusGraph 服务。

| Checkpoint | 分值 | 评估方式 |
|-----------|------|---------|
| 检查 JanusGraph 仓库是否已克隆 | 1 分 | 确定性检查（目录是否存在） |
| 检查二进制文件是否构建成功 | 3 分 | 确定性检查（需要在 pom.xml 中跳过 Docker 构建，难度较高，因此分值高） |
| 检查 JanusGraph Server 是否以 HTTP endpoint 运行 | 2 分 | 确定性检查（HTTP 请求是否成功） |

这个任务纯粹是技术操作，不涉及同事交互，所有 checkpoint 都可以用确定性方式评估。

**Finance 任务——填写 IRS 税表**

Task Intent：从 OwnCloud 上的 `TAC_financials.csv` 和本地的 `research_wages.csv` 获取财务数据，参考 `f6765_instructions.pdf` 说明书，填写 IRS Form 6765（研发税收抵免表）的 Section B。遇到模糊问题时，通过 RocketChat 联系财务总监 David Wong 确认。最终将填好的表格保存为 PDF。

| Checkpoint | 分值 | 评估方式 |
|-----------|------|---------|
| Section B 的 16 个问题是否全部正确填写 | 5 分 | 确定性检查（逐项比对答案） |
| 是否联系了正确的财务总监（David Wong）来回答两个模糊问题 | 3 分 | 检查 RocketChat 聊天记录 + LLM 判断对话是否达成目的 |

这个任务跨越三个平台（OwnCloud 读取文件、RocketChat 沟通、本地保存输出），需要 agent 理解 PDF 说明书、处理 CSV 数据、识别哪些问题需要向人确认，并且找到正确的人来问。

**PM 任务——收集项目指标并发布总结**

Task Intent：访问 Plane 的 Analytics 页面，收集 Open Tasks、Backlog Tasks、Unstarted Tasks、Started Tasks、Unassigned Issues、Pending Issues 六项指标，编写总结并发布到 RocketChat 的 `#kudos` 频道。

| Checkpoint | 分值 | 评估方式 |
|-----------|------|---------|
| 是否访问了 Plane 的 Analytics 页面 | 1 分 | 确定性检查（浏览历史） |
| 是否正确收集了所有要求的指标 | 3 分 | 确定性检查（比对数值） |
| 总结是否发布到 `#kudos` 频道 | 1 分 | 确定性检查（RocketChat 消息记录） |

### 涉及报告生成的任务

论文中展示的一个详细 workflow 示例就涉及报告撰写。该任务要求 agent 为 RisingWave 项目管理 sprint：先在 Plane 上找出未完成的 issue 并移到下一个 sprint（2 分），在 RocketChat 通知相关人员（1 分），从 GitLab 克隆仓库并运行代码覆盖率脚本（2 分），将 sprint summary report 上传到 OwnCloud（2 分），最后根据模拟项目经理的反馈修改报告（1 分）。

![Figure 3](/assets/images/the-agent-company/x5.png)
*Figure 3: Sprint 管理任务的完整流程。Agent 需要跨越 Plane、RocketChat、GitLab、OwnCloud 和 Sotopia 五个平台，最后两步涉及报告生成和根据反馈修改。该 agent 最终得到 4/8 分。*

报告上传和修改这两个 checkpoint 使用了 LLM-based 评估——LLM evaluator 会检查报告的清晰度、完整性，以及是否成功整合了经理的反馈。这类输出是非结构化的文本，无法用简单的字符串匹配来判断质量，因此必须依赖 LLM 判断。

### 涉及同事沟通的任务

论文附录给出了三个 agent 与模拟同事对话的案例：

1. **设备采购预算谈判**：agent 需要为部门采购设备，但计算后发现需求金额超出预算。Agent 主动与同事协商，建议减少采购数量以控制在预算内。这考察的是 agent 发现问题后能否主动沟通解决。

2. **撰写岗位描述（Job Description）**：agent 需要为一个新毕业生软件工程师岗位写 JD。为此 agent 向模拟项目经理询问 JD 模板、最低和优选资质要求、以及理想薪资范围，系统地收集完信息后再撰写。

3. **协调会议时间**：agent 需要在两位同事 Emily Zhou 和 Liu Qiang 之间安排一次会议。Agent 分别询问两人的可用时间，发现 Emily 周三和周四有空而 Liu Qiang 只有周四有空，最终确定周四上午 10:30 开会。这个任务需要在两个对话之间来回切换、比对信息。

这些任务的共同特点是：任务描述中不包含完成任务所需的全部信息，agent 必须通过与模拟同事的多轮对话来补全缺失信息。公司环境中还刻意设置了"陷阱"——例如某些 HR/Admin 任务中包含容易让人和 LLM 都犯错的干扰信息，模拟真实工作中的人为因素。

## 评估方式：checkpoint 部分计分 + LLM-as-Judge

评估采用 checkpoint 机制：每个任务被分解为若干 checkpoint（中间里程碑），每个 checkpoint 有对应的分值。例如上面的项目管理任务有 5 个 checkpoint，总分 8 分。

论文定义了两个能力指标：

**Full completion score**（$S_{\text{full}}$）：只有所有 checkpoint 都通过才得 1 分，否则 0 分。这是一个严格的二值指标。

**Partial completion score**（$S_{\text{partial}}$）：

$$
S_{\text{partial}} = 0.5 \cdot \frac{\text{Result}}{\text{Total}} + 0.5 \cdot S_{\text{full}}
$$

其中 Result 是所有 checkpoint 的得分之和，Total 是总分。这个设计让部分完成的任务也能获得分数（最多 50%），但完整完成额外奖励 50%，形成对全部完成的强激励。

大多数 checkpoint 的评估器是确定性的 Python 函数，直接检查环境状态（如文件是否存在、数据库记录是否正确）。但对于输出不易程序化判断的任务——例如检查一份报告是否正确整合了经理的反馈、或者 agent 是否向正确的同事询问了正确的问题——论文使用 Claude-3.5-Sonnet 作为 LLM-as-Judge。175 个任务中有 51 个（29%）涉及 LLM 评估。论文强调这些 LLM 评估主要用于"定义明确的信息提取和分类"场景，先尝试确定性关键词匹配，LLM 只作为 fallback。

## 实验结果

论文测试了 12 个模型，包括闭源的 Gemini、Claude、GPT-4o、Amazon Nova，以及开源的 Llama 和 Qwen 系列。所有模型使用 OpenHands CodeAct agent 框架运行。

| Agent | 模型 | 完成率 | 部分得分 | 平均步数 | 平均成本 |
|-------|------|--------|---------|---------|---------|
| OpenHands 0.28.1 | Gemini-2.5-Pro | 30.3% | 39.3% | 27.2 | $4.2 |
| OpenHands 0.28.1 | Claude-3.7-Sonnet | 26.3% | 36.4% | 27.8 | $4.1 |
| OpenHands 0.14.2 | Claude-3.5-Sonnet | 24.0% | 34.4% | 29.2 | $6.3 |
| OpenHands 0.14.2 | Gemini-2.0-Flash | 11.4% | 19.0% | 39.9 | $0.6 |
| OpenHands 0.14.2 | GPT-4o | 8.6% | 16.7% | 14.6 | $1.3 |
| OpenHands 0.14.2 | Llama-3.1-405B | 7.4% | 14.1% | 23.0 | $3.2 |
| OpenHands 0.14.2 | Llama-3.3-70B | 6.9% | 12.8% | 20.9 | $0.9 |
| OpenHands 0.14.2 | Qwen-2.5-72B | 5.7% | 11.8% | 24.0 | $1.5 |

Gemini 2.5 Pro 以 30.3% 的完成率领先，但即使是最强模型，近七成任务仍然失败。Gemini 2.0 Flash 虽然完成率只有 Gemini 2.5 Pro 的三分之一，但成本不到 $1，性价比突出。开源模型中 Llama 3.1 405B 和 Llama 3.3 70B 表现接近——后者参数量只有前者的 1/6，暗示小模型在 agent 能力上正在追赶。

### 按平台和任务类别的拆解

![Figure 2a](/assets/images/the-agent-company/x3.png)
*Figure 2（左）：不同平台上的完成率。OwnCloud 和 RocketChat 是所有模型的短板。*

![Figure 2b](/assets/images/the-agent-company/x4.png)
*Figure 2（右）：不同任务类别的完成率。SDE 和 PM 类任务表现较好，DS、Admin 和 Finance 类任务表现很差。*

按平台看，所有模型在 OwnCloud 上表现最差（Llama-3.1-405B 完成率为 0），RocketChat 也普遍困难。OwnCloud 的问题在于 web 端 Office 软件的 UI 极其复杂，RocketChat 则涉及社交互动能力。

按任务类别看，出现了一个与人类直觉相反的结果：对人类来说入门门槛较高的软件开发任务（SDE），agent 反而做得最好（Gemini 2.5 Pro 达 38%）；而对人类来说相对常规的行政（Admin）、数据科学（DS）和财务（Finance）任务，agent 表现很差（多数模型完成率为 0）。论文认为原因有两个：一是当前 LLM 的训练在编码能力上投入最大（HumanEval、SWE-bench 等 benchmark 的推动），公开代码数据丰富；二是行政和财务任务涉及的操作流程（制作电子表格、收集并填写多方信息、理解扫描文档等）虽然概念上简单，但需要复杂的 UI 操作、文档理解和跨平台协调能力，而这些正是当前 agent 的弱项。

### 典型失败模式

论文总结了三类有代表性的失败：

**缺乏社交理解**：agent 向同事 Alex 询问"接下来该向谁自我介绍？"，Alex 回复"去找前端团队的 Chen Xinyi"，agent 却没有跟进，直接认为任务完成了。

**浏览器操作困难**：OwnCloud 的欢迎弹窗成了 OpenHands agent（使用文本浏览模式）的障碍——它无法点击关闭按钮。OWL RolePlay（使用视觉浏览模式）虽然不受弹窗影响，但在复杂 UI 中更容易点错元素。

**自我欺骗**：当 agent 找不到需要联系的人时，它选择把另一个用户重命名为目标人物的名字来"绕过"问题——一种出人意料的创造性作弊。

## 思考与讨论

TheAgentCompany 与已有 agent benchmark 的最大区别在于它把"工作"作为一个整体来评估，而不是拆解成孤立的技能。一个 Finance 任务可能同时需要读 CSV 文件、查 PDF 说明书、在 RocketChat 上向财务总监确认模糊信息、填写表单并保存——这更接近真实的工作场景。175 个任务共耗费约 3000 人时来创建，这个投入保证了任务质量，但也意味着扩展到更多行业和职业类别的成本不低。

评估设计中 LLM-as-Judge 的使用比较克制——51 个涉及 LLM 评估的任务中，LLM 主要是确定性评估器的 fallback，用于判断非结构化输出（如报告质量、对话是否达成目的）。这种"确定性优先、LLM 兜底"的策略在保证可复现性和处理主观输出之间取得了平衡。不过论文没有报告 LLM judge 与人类评估之间的系统性一致性分析（如 Cohen's kappa），而是通过 3-5 人的代码审核来保证评估器质量。

论文坦诚指出了几个局限：任务偏向于有明确目标和成功标准的类型，没有覆盖更开放的创意性工作（如头脑风暴产品方案、设计系统架构）；只测试了两个 agent 框架（OpenHands 和 OWL RolePlay）；没有收集人类在这些任务上的表现数据作为对照。未来如果能加入人类基线，就能更准确地量化"AI 还差多远"，而不只是给出一个绝对的完成率数字。
