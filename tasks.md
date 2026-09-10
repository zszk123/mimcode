# mimcode 任务分解（tasks）

> 15 个任务，每个可在一次专注会话内完成。顺序即建议实施顺序。
> 参考定位均指 pi 源码（`e:\wmm\agent study\pi`）。

## T1 工程脚手架

- 内容：`uv init` 建项目；`pyproject.toml` 声明 Python >=3.11 与依赖（openai、anthropic、prompt-toolkit、rich、pydantic、pytest、pytest-asyncio）；src/mimcode/ 布局；`mimcode` 命令入口（先只有 --help/--version）；pytest 可跑空套件。
- 影响文件：`pyproject.toml`、`src/mimcode/__init__.py`、`src/mimcode/__main__.py`、`src/mimcode/cli.py`、`tests/test_smoke.py`
- 依赖：无
- 参考：`packages/coding-agent/package.json`（bin 入口形态）；pi 依赖面板无对照价值，按 Python 生态选型。

## T2 消息与事件类型系统

- 内容：pydantic 建模三类消息与内容块（text/thinking/toolCall/image 占位）、用量统计（含 cache 读写与 reasoning 字段）、AgentEvent 联合（agent_start / agent_end / turn_start / turn_end / message_start / message_update / message_end / tool_execution_start / update / end）、助手流事件（start / text_delta / thinking_delta / toolcall_delta / done / error）；JSON 往返序列化测试。
- 影响文件：`src/mimcode/types/__init__.py`（或拆 messages.py / events.py）、`tests/test_types.py`
- 依赖：T1
- 参考：`packages/agent/src/types.ts` L428-443（AgentEvent 联合）；`packages/agent/src/types.ts` L28-50（StreamFn/工具执行模式契约）；`packages/ai/src/types.ts`（Message/AssistantMessage/ToolResultMessage/Usage 定义）。

## T3 双协议 Provider 层 + faux transport

- 内容：Provider 抽象基类（模型目录、鉴权解析、流式调用）；OpenAIProtocolProvider：openai SDK chat.completions 流式，解析 tool_calls 增量与 `reasoning_content` 增量字段，归一为助手流事件；ClaudeProtocolProvider：anthropic SDK messages 流式，解析 content_block_delta 中的 thinking/text/tool 增量，支持 thinking 级别到 budget 的映射；faux transport：从 JSON fixture 回放流式 chunk 序列，供测试与后续 e2e；流函数契约对齐 pi：不抛异常，失败编码为流内 error + stopReason。
- 影响文件：`src/mimcode/provider/base.py`、`src/mimcode/provider/openai_protocol.py`、`src/mimcode/provider/anthropic_protocol.py`、`src/mimcode/provider/faux.py`、`tests/fixtures/*.json`、`tests/test_provider_*.py`
- 依赖：T2
- 参考：`packages/ai/src/models.ts` L88-149（Provider 接口：auth/getModels/stream/streamSimple）；`packages/ai/src/api/anthropic-messages.ts` L218-236、L755-761、L807-846（thinking 级别映射与 reasoning 用量归一）；`packages/agent/src/types.ts` L19-32（StreamFn 不抛异常契约）。reasoning_content 解析为 mimcode 新增（pi 无此路径）。

## T4 模型目录、配置与凭据解析

- 内容：预置模型目录（openai/anthropic/deepseek/zhipu/kimi 等条目：id、上下文窗口、是否支持 thinking、计价可留空）；配置文件读写（TOML，全局 + 项目两级）；端点与模型条目自定义合并；API key 解析链（环境变量 > 配置文件）；`--list-models` CLI 输出。
- 影响文件：`src/mimcode/provider/catalog.py`、`src/mimcode/config.py`、`src/mimcode/provider/auth.py`、`tests/test_config.py`
- 依赖：T3
- 参考：`packages/ai/src/providers/all.ts` L88-141（内置 provider 注册表与 builtinModels）；`packages/coding-agent/src/core/settings-manager.ts` 与 `src/core/defaults.ts`（配置层次与默认值思路）。

## T5 内置工具七件套

- 内容：工具协议（JSON schema 描述、execute、进度回调、结果 content/details、渲染元数据）；bash 工具（Windows 检测降级 PowerShell）、read、write、edit（唯一替换式 + 失败提示）、ls、find、grep；输出截断上限；串行/并行执行模式声明。
- 影响文件：`src/mimcode/agent/tools/{base,bash,read,write,edit,ls,find,grep}.py`、`tests/test_tools_*.py`
- 依赖：T2
- 参考：`packages/coding-agent/src/core/tools/bash.ts` + `powershell.ts`（Windows 降级与超时处理）；`core/tools/edit.ts` + `edit-diff.ts`（替换式编辑与 diff）；`core/tools/read.ts`/`write.ts`/`ls.ts`/`find.ts`/`grep.ts`；`core/tools/output-accumulator.ts` + `truncate.ts`（输出截断）；`core/tools/tool-definition-wrapper.ts`（包装与校验）。

## T6 Agent loop 核心

- 内容：双循环迁移（外循环 steering/follow-up 消息注入；内循环流式响应 + 工具批量执行）；截断消息的工具调用整批判错；before/after 工具钩子；prepare_next_turn（运行中换模型/思考级别）；should_stop_after_turn；工具结果按助手原始顺序回填；事件按序发出；中止信号处理。
- 影响文件：`src/mimcode/agent/loop.py`、`src/mimcode/agent/registry.py`、`src/mimcode/agent/stream.py`、`tests/test_agent_loop.py`
- 依赖：T3、T5
- 参考：`packages/agent/src/agent-loop.ts` L155-275（runLoop 双循环全逻辑）；L381-406（failToolCallsFromTruncatedMessage）；L489-554（并行批 + 按源序回填）；L600-668（prepareToolCall 校验与拦截）；`packages/agent/src/types.ts` L97-147（钩子与 AgentLoopTurnUpdate）。

## T7 会话持久化

- 内容：JSONL append-only 树（每行一条记录，含消息与分支指针）；会话 ID 与文件名（时间戳前缀）；按 cwd 编码目录；列出/读取/追加；恢复最近会话与指定会话分叉；损坏文件隔离（不影响其他会话发现）。
- 影响文件：`src/mimcode/app/session_store.py`、`src/mimcode/app/session.py`、`tests/test_session.py`
- 依赖：T2、T6
- 参考：`packages/coding-agent/src/core/session-manager.ts` L845-953（append-only 树、L953 文件命名）；L619-641、L817-841（发现容错）；`packages/agent/src/harness/session/jsonl/{codec,storage,repo}.ts`（JSONL 编解码与仓库）。

## T8 Compaction 上下文压缩

- 内容：上下文 token 估算（provider 用量优先，无用量时按字符估算）；阈值判断与触发；摘要生成请求（复用 provider 流）；压缩后保留近期上下文的裁剪逻辑；压缩结果作为特殊消息入会话。
- 影响文件：`src/mimcode/app/compaction.py`、`tests/test_compaction.py`
- 依赖：T6、T7
- 参考：`packages/agent/src/harness/compaction/compaction.ts` L148-162（默认设置：reserveTokens=16384、keepRecentTokens=20000）；L165-167（calculateContextTokens）；L216-271（估算）；L102-124（completeSimpleWithRetries 摘要请求）；`packages/coding-agent/src/core/compaction/`（交互层触发与摘要渲染）。

## T9 Skills 技能加载

- 内容：全局与项目两级技能目录发现；SKILL.md frontmatter 解析与校验（名称 ≤64 字符、描述 ≤1024 字符、忽略文件规则 .gitignore/.ignore/.fdignore）；技能列表格式化注入系统提示。
- 影响文件：`src/mimcode/app/skills.py`、`tests/test_skills.py`
- 依赖：T6（挂入系统提示）
- 参考：`packages/coding-agent/src/core/skills.ts` L11-16（限制常量）；L168-277（loadSkillsFromDir/loadSkillFromFile）；L355-391（formatSkillsForPrompt/XML 转义）。

## T10 Slash 命令

- 内容：命令注册表（名称、描述、参数提示、执行回调）；内置命令：/model、/thinking、/sessions、/resume、/fork、/compact、/skills、/extensions、/exit（及 /help）；命令补全数据源。
- 影响文件：`src/mimcode/app/commands.py`、`tests/test_commands.py`
- 依赖：T4、T7、T8、T9
- 参考：`packages/coding-agent/src/core/slash-commands.ts`（注册结构与分发）；`src/modes/interactive/components/*selector.ts`（交互式选择的命令形态，仅参考语义）。

## T11 Extensions 插件机制

- 内容：扫描全局（~/.mimcode/extensions/）与项目（.mimcode/extensions/）目录的 .py 文件；调用约定入口（模块级 `register(ctx)`）；ctx 提供工具注册、事件订阅、会话只读访问；加载异常隔离（单插件失败记录诊断、不中断启动）；/extensions 列出已加载插件。
- 影响文件：`src/mimcode/app/extensions/{loader,context,types}.py`、`tests/test_extensions.py`
- 依赖：T6
- 参考：`packages/coding-agent/src/core/extensions/loader.ts`（目录扫描与动态加载）；`runner.ts`（执行隔离）；`types.ts`（插件 API 面）；注意 Python 版按「register(ctx) 注册函数」重新设计，不做 TS API 面的逐字翻译。

## T12 Interactive TUI 骨架

- 内容：prompt_toolkit 应用（异步事件循环与 agent 协程的整合）；输入框（多行、历史、粘贴大文本截断提示）；AgentEvent → 渲染队列 → 刷新管线；流式文本增量追加；footer（当前模型/会话/用量）；退出与中断（Esc/Ctrl-C 分级：中断请求 vs 退出）；事件顺序与并发写安全。
- 影响文件：`src/mimcode/tui/app.py`、`src/mimcode/tui/input.py`、`src/mimcode/tui/renderer.py`、`tests/test_tui_pipeline.py`
- 依赖：T6、T10
- 参考：`packages/tui/src/tui.ts` L20-47（组件契约 render/handleInput/invalidate——Python 版以「渲染队列 + 增量失效」等价实现，不逐字复刻）；`packages/coding-agent/src/modes/interactive/interactive-mode.ts`（模式控制器与 TUI 的职责边界）。

## T13 高保真渲染

- 内容：Rich Markdown 渲染（助手消息）；主题系统（深/浅主题 + 语义色：用户/助手/工具/错误/thinking）；diff 语法高亮（编辑工具结果的新旧对照，按文件类型着色）；思考块折叠（流式期间显示动态指示，结束后可展开/收起）；工具执行展示（名称、参数摘要、耗时、结果预览）；流式光标动画。
- 影响文件：`src/mimcode/tui/markdown.py`、`src/mimcode/tui/theme.py`、`src/mimcode/tui/diff.py`、`src/mimcode/tui/components/{assistant,tool,thinking,diff}.py`、`tests/test_render_*.py`
- 依赖：T12
- 参考：`packages/coding-agent/src/modes/interactive/components/assistant-message.ts`（助手消息渲染结构）；`components/markdown-transform.ts`（流式 markdown 的容错处理思路）；`components/diff.ts` + `core/tools/edit-diff.ts`（diff 渲染）；`components/thinking-selector.ts`（思考折叠交互）；`theme/theme.ts`（主题结构）。

## T14 主流程接入（print 模式 + CLI 装配）

- 内容：参数解析对齐 pi 语义：`-p/--print`、`-c/--continue`、`--fork`、`--session`、`--model`、`--thinking`、`--list-models`、`--version`、`--help`；非 TTY 或 -p 时走 print 模式（流式文本 + thinking 分离输出，退出码语义）；TTY 走 interactive；AgentSession 装配（provider 解析、模型解析、系统提示组装、工具注册、技能/插件加载、会话挂接）；首次启动引导（无配置时提示配置端点）。
- 影响文件：`src/mimcode/cli.py`（扩展）、`src/mimcode/modes/print_mode.py`、`src/mimcode/app/agent_session.py`、`src/mimcode/core/system_prompt.py`、`tests/test_cli.py`、`tests/test_print_mode.py`
- 依赖：T4、T6、T7、T9、T10、T11、T12、T13
- 参考：`packages/coding-agent/src/main.ts` L109-120（模式分发：非 TTY/-p → print）；L350-440（会话 flag 语义）；L443-520（模型/思考级别 flag 到会话选项）；`src/cli/args.ts`（参数定义）；`src/modes/print-mode.ts`（print 输出细节）；`src/core/agent-session.ts` L309-408（AgentSession 装配面）；`src/core/system-prompt.ts`（系统提示组装）。

## T15 端到端验证

- 内容：faux transport 驱动的端到端用例：print 模式全链路（提示 → 流式 → 工具调用 → 工具结果 → 最终回复 → 会话落盘 → --continue 恢复）；interactive 模式 headless 冒烟（模拟按键序列：输入提示、观察渲染管线产出、中断、slash 命令切换模型）；compaction 触发路径（构造超长会话 fixture 验证压缩）；Windows PowerShell 降级验证；`npm run check` 的 Python 等价物（ruff + mypy + pytest 单命令门禁）落地。
- 影响文件：`tests/e2e/test_print_e2e.py`、`tests/e2e/test_interactive_e2e.py`、`tests/e2e/test_compaction_e2e.py`、`scripts/check.py`（或 Makefile/justfile）
- 依赖：T14
- 参考：`packages/coding-agent/test/suite/harness.ts` + faux provider（测试 harness 形态）；`AGENTS.md` 的 check/test 纪律（门禁语义）。

---

## 依赖关系速览

```
T1 → T2 → T3 → T4
        T3 ┐
        T5 ┴→ T6 → T7 → T8
              T6 → T9
              T6 → T11
   T4+T7+T8+T9 → T10
        T6+T10 → T12 → T13
T4+T6+T7+T9+T10+T11+T12+T13 → T14 → T15
```

## 会话纪律

- 每个任务完成即跑该任务的测试文件，红→绿再进入下一任务。
- 所有测试零真实网络；faux fixture 与真实 SDK 响应形态的差异在 fixture 注释中标注。
- 不逐字翻译 pi 的 TypeScript：同构事件/流程语义，Python 侧用地道写法（async 生成器、pydantic、dataclass）。
