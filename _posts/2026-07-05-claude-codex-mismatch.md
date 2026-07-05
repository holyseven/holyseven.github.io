---
layout: post
title: "【笔记】为什么 Claude 在 CodeX harness 中表现崩塌"
date: 2026-07-05
categories: [paper-notes]
tags: [LLM-agent, harness-engineering, Claude, CodeX, agent-compatibility]
---

## 现象

在 AgenticDataBench 的评测中，CodeX (Claude Sonnet 4.6) 以 31.6% 的得分垫底，是所有 12 种 agent 配置中最差的——甚至远低于同 harness 下的 Kimi-K2.5（48.8%）和 Qwen3.5（39.9%）。考虑到 Claude 4.6 在其他 harness 中表现正常（Claude Code 46.6%、DA-Agent 46.1%），这个异常需要解释。

## 论文给出的直接原因

AgenticDataBench 论文在实验分析中明确指出两点：

1. **Claude 在 CodeX 中频繁提前终止**：不产出 task solution 就停止执行，而 Kimi/Qwen 会持续尝试
2. **无法受益于 CodeX 的 auto-compaction**：当 context 溢出时，CodeX 的自动压缩机制对 Claude 不生效——Claude 直接停工而非让 harness 帮它压缩上下文继续

## 原因分析：API 格式兼容性问题（附源码验证）

经过深入讨论和 CodeX 开源代码验证，根本原因是 **API 协议层面的格式不兼容**，而非模型能力问题。以下分析基于 [CodeX CLI 源码](https://github.com/openai/codex) 的实际代码逻辑。

### 排除模型能力因素

如果 Claude 在数据科学任务上有模型能力缺陷，应该在所有 harness 中一致表现差。但事实是：

| Harness | Claude 4.6 得分 | 排名 |
|---------|----------------|------|
| Claude Code | 46.6% | 该 harness 最优 |
| DA-Agent | 46.1% | 该 harness 最优 |
| Smolagents | 46.7% | 该 harness 最优 |
| CodeX | 31.6% | 该 harness 最差 |

Claude 在三个 harness 中都是最好的 LLM，唯独在 CodeX 中崩塌。这明确排除了模型能力问题。

### 源码验证：`end_turn` 字段是核心决策点

CodeX 的 agent 循环继续/停止的决策逻辑在 `codex-rs/core/src/session/turn.rs`：

```rust
// turn.rs:2232
if let Some(false) = end_turn {
    needs_follow_up = true;
}
break Ok(SamplingRequestResult {
    needs_follow_up,
    last_agent_message,
});
```

这段代码的语义非常明确：**只有当 `end_turn == Some(false)` 时，CodeX 才认为模型还需要继续工作**。如果 `end_turn` 是 `None` 或 `Some(true)`，agent 循环就会停止。

源码注释直接承认了这个问题（`codex-api/src/common.rs:94`）：

```rust
/// Did the model affirmatively end its turn? Some providers do not set this,
/// so we rely on fallback logic when this is `None`.
end_turn: Option<bool>,
```

"Some providers do not set this"——这正是 Claude 通过兼容层接入时会遇到的情况。

### CodeX 只支持 Responses API

从 `model-provider-info/src/lib.rs` 可以确认，CodeX **已经移除了 Chat Completions API 的支持**：

```rust
pub enum WireApi {
    /// The Responses API exposed by OpenAI at `/v1/responses`.
    #[default]
    Responses,
}

// 尝试使用 chat 格式会报错
"chat" => Err(serde::de::Error::custom(CHAT_WIRE_API_REMOVED_ERROR)),
```

这意味着所有第三方模型（包括 Claude）必须通过兼容层将其原生 API 转译为 OpenAI Responses API 格式。Claude 的原生 API 是 `/v1/messages`，两者在关键字段上存在结构性差异。

### `end_turn` 映射问题的具体机制

OpenAI Responses API 的 `response.completed` 事件中包含 `end_turn: bool` 字段。CodeX SSE 解析器（`codex-api/src/sse/responses.rs:119-120`）将其反序列化为 `Option<bool>`：

```rust
struct ResponseCompleted {
    id: String,
    usage: Option<ResponseCompletedUsage>,
    end_turn: Option<bool>,  // 关键字段
}
```

当 Claude 通过 LiteLLM 等兼容层接入时，Claude 原生 API 的 `stop_reason` 字段需要被映射为 Responses API 的 `end_turn`：

| Claude 原生 `stop_reason` | 应映射为 `end_turn` | 实际可能的映射 |
|---------------------------|-------------------|--------------|
| `"end_turn"` | `false`（如果 Claude 只是在等待工具结果） | `true` 或 `None`（被误判为任务结束） |
| `"tool_use"` | `false` | 不适用（tool call 有独立的 follow-up 机制） |
| `"stop_sequence"` | `true` | `true`（正确） |

关键问题：Claude 的 `stop_reason: "end_turn"` 在语义上是模糊的——它可能表示"我说完这句话了，等你回复"（应映射为 `end_turn: false`），也可能表示"整个任务完成了"（应映射为 `end_turn: true`）。兼容层无法可靠区分这两种情况，倾向于映射为 `true` 或直接不设置（`None`）。

无论哪种情况，CodeX 都会停止 agent 循环。

### Auto-compaction 失效的机制

CodeX 的 auto-compaction 触发条件（`turn.rs:346`）：

```rust
if needs_follow_up && (sess.take_new_context_window_request().await || token_limit_reached) {
    run_auto_compact(...).await;
}
```

Auto-compaction 需要 **两个条件同时满足**：`needs_follow_up == true` 且 token 超限。如果 `end_turn` 映射错误导致 `needs_follow_up` 为 false，即使 context 已经爆满，compaction 也不会触发——CodeX 认为任务已经结束了，没有必要压缩。

### Tool call 解析是唯一的"保险丝"

源码中还有一条独立路径可以触发 `needs_follow_up`（`stream_events_utils.rs:442`）：

```rust
// 如果模型发出了 tool call，无条件继续
Ok(Some(call)) => {
    output.needs_follow_up = true;
    output.tool_future = Some(tool_future);
}
```

这意味着：**只要 tool call 被正确解析，即使 `end_turn` 映射错误，agent 循环也会继续**。这条保险丝解释了为什么类别 2（空手终止）只有 55 个而非全部——只有当 Claude 那一轮**没有发出 tool call、只回了纯文本**时，`end_turn` 误映射才会真正触发终止。

### 为什么 Kimi-K2.5 不受影响

Kimi 等模型的 API 直接兼容 OpenAI 格式，`end_turn` 字段能被正确传递。通过兼容层接入 CodeX 时接近透传，不存在语义映射的歧义。

## 归因链总结

源码验证确认了 `end_turn` 映射问题导致提前终止的具体机制。但这只是假说，还需要用实际数据验证它能解释多少比例的失败。

## 定量验证：Per-Task 数据

AgenticDataBench 公开了 344 个任务的 per-task 结果（[仓库](https://github.com/AgenticDataBench/AgenticDataBench/tree/main/testbed/results)）。每个任务记录了 `finished`（harness 是否正常结束）、`added_files`/`changed_files`（产出的文件）和 `total_score`。对比 `codex-claude`、`claude-code-claude`、`codex-kimi` 三份数据，可以检验 `end_turn` 假说能解释 15 个百分点差距（31.6% vs 46.6%）的多少。

### 只用可靠信号分组

公开数据里**没有逐步执行 trace，也没有 API 请求日志**。能用的可靠信号只有两个：`finished` 标志，以及是否产出了真实输出文件（排除 `.run_codex.py`、`.task_prompt.txt` 等脚手架文件）。下面的分组只基于这两个信号。

| 类别 | 判定条件 | 任务数 | CodeX-Claude | CC-Claude | 占 15 点差距 |
|------|---------|--------|------|------|------|
| 1. Context 溢出 | `finished=false` | 41 | 0.012 | 0.282 | 21.5% |
| 2. 空手终止 | `finished=true` 但无输出文件 | 55 | 0.000 | 0.448 | 47.9% |
| 3. 正常产出 | `finished=true` 且有输出文件 | 248 | 0.437 | 0.500 | 30.6% |

### Kimi 对照：类别 1、2 是 Claude 特有的提前终止

引入 CodeX-Kimi（同一个 harness，只换模型）作为对照：

- **类别 1**：CodeX-Claude 有 41 个任务 `finished=false`；同样跑在 CodeX 上的 Kimi 只有 1 个，Claude Code 里的 Claude 也只有 1 个。在这 41 个任务上，Kimi 有 40 个正常结束。
- **类别 2**：这 55 个任务里，CodeX-Kimi 在**全部 55 个**上都产出了输出文件（均分 0.449），而 CodeX-Claude 一个都没产出（均分 0.000）。

同一个 harness，换成 Kimi 就正常结束、正常产出。这说明类别 1、2 是 **Claude 特有的提前终止**，与源码分析的 `end_turn` 机制完全吻合：

- 类别 1：`needs_follow_up=false` 导致 auto-compaction 不触发，context 撑到上限后崩溃
- 类别 2：`end_turn` 被误映射为 `true`/`None`，Claude 还没产出结果就被判定"任务结束"

**类别 1 + 类别 2 = 69.4% 的差距**，都由 Claude 特有的提前终止解释。

#### 类别 1 案例（context 溢出）

`healthcare_08`（CodeX-Claude=0.00，CC=1.00，Kimi=1.00）：Claude 在 CodeX 中执行了 19 步，最后一步不是模型主动结束，而是 harness 抛出 `TurnFailed`：

```json
{"type": "turn.failed", "error": {"message": "Codex ran out of room in the model's context window"}}
```

同一任务 Claude Code 和 CodeX-Kimi 都拿了满分。区别只在于：CodeX 的 auto-compaction 因 `needs_follow_up=false` 没被触发，context 一路涨到上限后崩溃。

#### 类别 2 案例（空手终止）

`loan_risk_33`（CodeX-Claude=0.00，CC=1.00，Kimi=1.00）：`finished=true`，但只执行了 2 步就停了，没写任何输出文件：

```
[0] Think: Let me start by exploring the workspace to understand the data files.
[1] Bash:  ls /workspace && head -3 /workspace/input_1.csv
```

第 2 步刚看了眼数据，Claude 那一轮回了纯文本、没发 tool call——`end_turn` 保险丝断开，CodeX 直接判定"任务完成"。这类任务里也有执行到 13~20 步、但每一步都停在 Think（纯文本）而非 tool call 的（如 `strategy_3` 19 步、`tourism_12` 18 步），最后一句往往是"Now let me create the output file"然后就被判定结束了——正是 `end_turn` 保险丝的典型断裂点。

### 类别 3：正常产出但结果差（30.6%）

248 个"正常产出"任务内部高度分化：

| 子类 | 判定 | 任务数 | 对 15 点差距的贡献 |
|------|------|--------|-------------------|
| 3A：两者一致 | \|差距\| < 0.1 | 167 | ≈ 0% |
| 3B：CC 明显更好 | 差距 ≥ 0.1 | 55 | +54.9% |
| 3C：CodeX 反而更好 | 差距 ≤ -0.1 | 26 | -23.5% |

净贡献 30.6%，主要来自 55 个 3B 任务。

#### 3A 案例（两者一致，对差距无贡献）

`agriculture_02`、`energy_09` 等 167 个任务：CodeX-Claude 与 CC-Claude 都拿满分（如 `agriculture_02` 两边都是 1.00）。这些任务 Claude 在 CodeX 里同样能做对，说明"正常产出"路径本身没问题。

#### 3B 案例（CC 明显更好）——这 55 个任务里 Claude 到底错得离谱吗？

先按失败模式拆开这 55 个任务：

| 失败模式 | 任务数 | 说明 |
|---------|--------|------|
| 脚本报错/输出文件没写成 | 12 | 跑了分析但最终 `output.csv`/`output.json` 不存在或路径错 |
| 产出了文件但 0 分 | 15 | 答案完全错 |
| 产出了文件且拿了部分分 | 28 | 大部分对、错在细节 |

**结论：多数不是"错得离谱"，而是 near-miss。** 28/55 拿了部分分，真正 0 分的只有 15 个。逐个看错误：

- `financial_325`（CX=0.00，CC=1.00）：要求按行业算 Q3 同比、取 Top3。Claude 给的是 `软件和信息技术服务业 / 电气机械和器材制造业 / 非金属矿物制品业`，gold 是 `电气机械和器材制造业 / 非金属矿物制品业 / 酒、饮料和精制茶制造业`——**3 个里对了 2 个**，第 3 个错源于 Q3 2024 数据口径（Claude 在第 17 步自己也发现"2024 和 2025 公司数差异很大"但没处理对）。评测按列精确匹配，2/3 对也判 0 分。
- `energy_10`（CX=0.00，CC=1.00）：算 R²/MAE/MSE/RMSE。Claude 的 R²=0.9477，gold=0.8251——**用了不同的模型/特征**，数值系统性偏差，不是小数点问题。
- `financial_95`（CX=0.00，CC=1.00）：分组标签写成 `规模前10%`，gold 要 `总管理规模排名前10%`；且重仓行业选择也不同。**部分是命名格式、部分是选择逻辑。**
- `real_estate_27`（CX=0.25，CC=1.00）：画图任务，数据分布与 gold 逐元素对比相对误差多在 1~5%，但评测容差是 1%，**接近但没卡进容差**。

同一个 Claude 模型在 Claude Code 里对这些任务几乎全对（如 `financial_325` CC 输出的 Top3 与 gold 完全一致）。所以不是 Claude "不会做数据分析"，而是在 CodeX 里更容易在口径选择、标签格式、精度上偏一点点。

> **一个必须说明的数据局限：公开数据里没有任务原文。** 每个任务的题面存在输出目录下的 `.task_prompt.txt` 文件里，公开的 per-task JSON 只引用了文件名、不含内容。因此对这 55 个任务，我能核到的只有两样：评测器的期望值/报错（`info` 字段）和 Claude 自己的执行 trace（`actions` 字段）。**拿不到题面，就无法逐个区分**每个失败到底是"题面欠定、Claude 的默认口径与 gold 不一致"还是"题面已说清、Claude 就是做错了"。上面 `financial_325` 是少数能确认存在真实口径歧义的——Claude 在 trace 里自己写了"2024 和 2025 公司数差异很大"，说明"算同比用哪一批公司"题面没锁死。但这**不能推广成"整个 benchmark 普遍欠定"**，那是未经证实的推断。

#### 3C 案例（CodeX 反而更好，抵消 23.5%）

反方向也存在：26 个任务 CodeX-Claude 比 CC-Claude 好。如 `financial_297`（CX=1.00，CC=0.00）、`transportation_10`（CX=1.00，CC=0.00）——这里是 Claude Code 那次跑偏了。这说明 3B/3C 里有相当一部分是**单次运行的方差**，而非稳定的系统性差异（论文没做重复实验，无法进一步分离）。

#### 三方对比

在 55 个 3B 任务上：

| 系统 | 均分 |
|------|------|
| CC-Claude | 0.742 |
| CodeX-Kimi | 0.606 |
| CodeX-Claude | 0.229 |

Kimi 对照排除了两个简单解释：不是 Claude 模型能力不足（CC-Claude=0.74），也不是 CodeX 对这类任务处理差（Kimi=0.61）。Claude 确实执行完整、产出了文件，但结果在细节上偏离。

**这 30.6% 的失败机制无法从公开数据确定。** 之前设想的"兼容层截断了 tool 返回值"这条方向，在读过 CodeX 源码后可以基本排除：tool 的执行和输出截断都是 CodeX 在本地统一完成的（stdout 上限 1MB、发给模型上限 10k tokens，见 `exec.rs` 与 `unified_exec/mod.rs`），对 Claude 和 Kimi 一视同仁；兼容层（Responses API ↔ Messages API）只做字段格式转换，不裁剪 tool result 的文本内容。既然 Kimi 通过同一条路径拿到了正确结果，"harness 级别的内容截断"就不成立。

结合案例看，剩下的两个方向是：
- **CodeX 的 system prompt 对输出格式/口径的引导适配 Kimi 但不适配 Claude**（能解释命名格式、口径选择类的偏差）
- **单次运行方差**（3C 反向案例 + 论文未做重复实验，说明这部分不可忽略）

验证前者需要提取 CodeX 的实际 system prompt、用相同 prompt 独立测试 Claude 与 Kimi 的格式/口径遵从差异；验证后者需要重复跑多次算方差。两者都超出了当前公开数据的能力。

### 源码补充：CodeX 的 system prompt 是为 GPT-5 定制的

第一个方向能从 CodeX 源码找到间接支持。CodeX 的 base instructions 文件只有 GPT-5 系列的变体：

```
codex-rs/core/gpt_5_codex_prompt.md
codex-rs/core/gpt_5_1_prompt.md
codex-rs/core/gpt_5_2_prompt.md
codex-rs/core/gpt-5.1-codex-max_prompt.md
codex-rs/core/gpt-5.2-codex_prompt.md
```

**没有任何 Claude/Anthropic 变体，且这些文件里没有一处提到 Claude 或 Anthropic。** 这份几百行的 prompt（讲 Codex CLI 的 personality、AGENTS.md 规范、`apply_patch` 格式、sandbox 审批流程等）会被 `client.rs:835-849` 作为 Responses API 的 `instructions` 字段（或 `developer` role 消息）原样发给**所有模型**：

```rust
// client.rs:849 —— base_instructions 直接作为 instructions 发出，不区分模型
(prompt.base_instructions.text.clone(), Some(tools))
```

也就是说 Claude 在 CodeX 里收到的是一份**专门为 GPT-5 调过、从没在其上对齐过的** system prompt。这**为**"输出口径/格式引导不适配 Claude"提供了一个可能的机制——但只到"可能"为止。要坐实它，得拿到任务原文、用同一份 prompt 独立对比 Claude 与 Kimi 在这些欠定任务上的口径遵从差异；而如上一节所述，公开数据里没有题面。所以准确的定位是：**"GPT 定制 prompt 导致口径偏差"是一个有源码线索（只有 GPT-5 变体、无 Claude 变体）但未经实测证实的假说，不是结论。**

一个容易混淆但**不成立**的方向是 chat template：Claude 有自己的对话模板（special tokens），但这层由 Anthropic 服务端在收到结构化 messages 后自行渲染，CodeX/兼容层控制不到，也不会错配。真正被"拼上"的是 CodeX 自己的 GPT 定制 base instructions，而非 chat template。

### 总结

| 归因 | 占 15 点差距 | 确定性 |
|------|------------|--------|
| `end_turn` 提前终止（context 溢出 + 空手终止） | 69% | 源码 + 数据 + Kimi 对照，三重验证 |
| Claude × CodeX 正常产出但结果偏差 | 31% | 确认存在，机制未定（GPT 定制 prompt 口径引导 / 单次方差） |

**确定的结论**：约 69% 的差距是 `end_turn` API 映射导致的提前终止——纯工程问题，与模型能力无关。最直接的证据是 Kimi 通过同一 harness 正常运行：41 个 context 溢出里 Kimi 只溢出 1 个，55 个空手终止里 Kimi 全部产出。

**未解决的问题**：剩余 31% 来自 Claude 正常执行但结果偏差。已排除模型能力（Claude 在 CC 上最强）、harness 通用问题（Kimi 在 CodeX 上正常）、兼容层内容截断（源码证伪）。这些偏差**多数是 near-miss 而非离谱错误**（55 个里 28 个拿了部分分，且存在 26 个反向案例）。源码显示 CodeX 的 system prompt 是纯 GPT-5 定制、对 Claude 零适配，这**为**"口径/格式引导不适配"提供了一个可能机制，叠加单次运行方差——但**这两者都只是假说**：公开数据里没有任务原文，无法逐个区分"题面欠定导致口径分歧"与"Claude 单纯做错"，也无法把"系统性不适配"和"运气差"拆开确诊。

## Harness 工程的更大图景

这个案例把一个容易被忽略的事实摆到台前：**同一个模型，换个 harness，评测分数可以从 46.6% 掉到 31.6%。** 差距的大头（69%）不是模型变笨了，而是 harness 与模型的接口没对齐。

### 模型-Harness 适配是一等问题

业界已经反复观察到同一现象：同一模型在不同 harness 下分数天差地别（有报告称同模型在两套 harness 下 78 vs 42）。原因可以归到三个层面，本文的案例恰好各踩一个：

1. **协议层**：Agent 循环的继续/停止依赖某个字段（这里是 `end_turn`）。当模型经兼容层接入、该字段语义映射不可靠时，循环会在错误的时机终止。CodeX 移除 Chat Completions、只保留 Responses API，进一步放大了非 OpenAI 模型的这类风险。
2. **资源管理层**：auto-compaction 这类"救援"机制往往挂在同一个信号上（`needs_follow_up`）。信号错了，救援不触发，长任务直接因 context 溢出而崩。
3. **引导层**：system prompt / 工具描述 / 输出格式约定通常是围绕某一个模型家族调出来的。换一个模型，遵从方式的细微差异就会在严格评测下变成失分。

前两层是**硬失败**（任务直接没做完），容易定位；第三层是**软失败**（做完了但结果偏一点），最难归因，也最容易被误判成"模型能力差"。

### 对使用者的启示

- **评测一个模型，其实是在评测"模型 × harness"这个组合。** 单一 harness 的分数不能直接下"模型 A 比模型 B 弱"的结论——先确认接口是否对齐。
- **跨厂商接入优先走原生 API，慎用兼容层。** 兼容层在格式转换上通常没问题，但在控制流字段（何时结束、何时压缩）的语义映射上是脆弱点。
- **system prompt 可能需要按模型家族适配。** CodeX 只有 GPT-5 变体、无 Claude 变体，这是"一份为 GPT 调的 prompt 直接喂给 Claude 会在输出细节上丢分"的合理怀疑方向——但本案例中它仍是假说，未经实测证实。
- **重复实验很重要。** 本案例 31% 的软失败里混着单次方差，没有多次运行就无法把"系统性不适配"和"运气差"分开。

harness 工程正在从"给模型套个循环"演变成一门需要针对模型特性精细调校的工程。这个 31.6% 的异常值，本质上是这门工程还不成熟的一个注脚。
## 参考来源

- [Model-Harness-Fit — Nicolas Bustamante](https://nicolasbustamante.com/blog/model-harness-fit)
- [Same model 78 vs 42: the harness made the difference](https://natesnewsletter.substack.com/p/same-model-78-vs-42-the-harness-made)
- [6 categories AI Agent Behavior-Improvement Lifecycle](https://chierhu.medium.com/the-ai-agent-landscape-and-the-behavior-improvement-lifecycle-8e0560df0e08)
- [Codex vs Claude Code (2026): A Real Head-to-Head](https://nimbalyst.com/blog/codex-vs-claude-code-workflow-harness/)
- [Cloudflare: OpenAI Codex Gateway docs](https://developers.cloudflare.com/ai-gateway/integrations/coding-agents/openai-codex/)
- [Codex CLI custom providers](https://openai-codex.mintlify.app/configuration/custom-providers)
- [OpenAI: Harness engineering](https://openai.com/index/harness-engineering/)
