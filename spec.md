# mimcode 规格（spec）

> pi 的 Python 迁移版：关键逻辑不变，做语言与生态适配。
> 参照源码：`e:\wmm\agent study\pi`（earendil-works/pi-mono，TypeScript monorepo）。

## 背景

pi 是一个成熟的终端 AI 编程助手（TypeScript），核心由四层构成：AI 消息与流式事件模型、双循环 agent loop（含工具批量执行与 steering 消息注入）、会话/压缩/技能/插件等外围子系统、TUI 交互层。mimcode 将这套关键逻辑迁移到 Python，让熟悉 Python 生态的开发者能以 OpenAI 兼容端点或 Claude 兼容端点为模型来源使用同等的终端编程助手能力。

与 pi 的核心差异只有一处：provider 层不再按厂商枚举 40+ 个 provider，而是收敛为「协议即 provider」——OpenAI 协议与 Claude 协议各一个实现，用各自官方 Python SDK，通过 base_url 覆盖吃下所有兼容端点。

## 目标用户

- 本人（学习 agent 架构 + 日常使用）。
- 模型来自 OpenAI 兼容端点（DeepSeek / GLM / Kimi / Ollama / vLLM 等）或 Claude 兼容端点的中文开发者。
- Windows 为主要使用平台，同时保持 Linux/macOS 可用。

## 能力清单

一句话一条，全部为 v1 交付范围：

1. 以 OpenAI 协议接入任意兼容端点：官方 OpenAI 及第三方，支持自定义 base_url 与 API key，用官方 openai SDK。
2. 以 Claude 协议接入 Anthropic 及兼容端点：支持自定义 base_url 与 API key，用官方 anthropic SDK，支持 extended thinking 的预算与级别控制。
3. 预置常用端点的模型目录（官方 OpenAI、Anthropic、DeepSeek、GLM、Kimi 等），并允许在配置文件中自定义端点与模型条目。
4. 解析并渲染两类推理内容：Claude 的 thinking 块（流式）与 OpenAI 兼容端点的增量推理字段（DeepSeek-R1/GLM 等风格）。
5. Agent loop 与 pi 同构：流式助手响应；单条消息内的多个工具调用按串行或并行批量执行；agent 运行中可插入用户消息（steering）；每轮之间可切换模型与思考级别；输出被截断时整批工具调用判错而非误执行。
6. 工具执行的前置/后置拦截钩子（阻断执行、改写结果），供插件与会话层使用。
7. 内置七件套工具：shell 执行（Windows 上自动降级 PowerShell）、读文件、写文件、编辑文件（替换式）、列目录、查找文件、搜索内容；输出截断保护与进度回调。
8. 会话持久化：append-only 树状结构（含分支/分叉语义）存 JSONL 文件；支持按目录列出历史会话、恢复最近会话、从指定会话分叉。
9. 上下文压缩：依据 provider 用量估算上下文占用，超过阈值时自动触发摘要压缩并保留近期上下文。
10. skills：按 pi 语义发现并加载 SKILL.md（frontmatter 校验：名称/描述长度限制、忽略规则），格式化注入系统提示。
11. extensions：扫描全局与项目两级扩展目录中的 .py 文件，以注册函数形式加载插件；插件可注册自定义工具、事件监听器；单个插件加载失败不影响其余插件。
12. slash 命令：内置模型切换、思考级别切换、会话列表/恢复/分叉、压缩触发、技能列表、插件列表、退出等。
13. 交互式 TUI：prompt_toolkit 驱动输入（多行、历史、快捷键），Rich 做高保真渲染：markdown、主题色、diff 语法高亮、流式光标、思考块折叠展示、工具执行过程展示。
14. print 模式：`-p` 一次性执行提示词并输出结果，语义对齐 pi。
15. CLI 参数：模型/端点指定、思考级别、会话继续/分叉/指定 ID、初始提示词携带文件、非 TTY 自动降级 print 模式。

## 非功能要求

- Python 最低版本 3.11；uv 管理依赖与虚拟环境；单包布局（src/mimcode/），不做 monorepo。
- 认证仅 API key：环境变量与配置文件两种来源，无 OAuth、无加密凭据库。
- 测试零真实网络：全部走 faux transport（预录流式数据），pytest 全绿是合并前提。
- Windows 一等公民：shell 工具、路径处理、终端渲染在 Windows（PowerShell/Windows Terminal）下优先验证。
- 冷启动到可输入提示 < 2 秒（不含模型请求）。
- 事件与消息模型语义同构 pi（AgentEvent 命名与语义一致），便于逐模块对照迁移与回归。
- 异常精细化捕获，不捕获裸 Exception；外部 SDK 错误在 provider 层归一为流内 error 事件（对齐 pi 的 StreamFn 契约：流函数不抛异常，失败编码进流）。

## 设计骨架

分层（自底向上，单向依赖）：

1. `mimcode.types`：消息（用户/助手/工具结果）、内容块（文本/思考/工具调用）、用量统计、AgentEvent 联合类型。纯数据层，pydantic 建模 + JSON 序列化。
2. `mimcode.provider`：协议即 provider。两个实现各自把官方 SDK 的流式 chunk 翻译为统一的助手流事件；模型目录与端点配置解析；API key 解析（环境变量 > 配置文件）。
3. `mimcode.agent`：agent loop（双循环：steering 外循环 + 工具执行内循环）、工具注册表与执行、事件总线。
4. `mimcode.app`：会话管理（JSONL append-only 树）、compaction、skills、extensions 加载、slash 命令、系统提示组装、配置。
5. `mimcode.cli` / `mimcode.tui`：参数解析与模式分发；print 模式；interactive 模式（prompt_toolkit 应用 + Rich 渲染器 + 主题）。

## Out of Scope（明确不做）

- RPC/JSON 模式与 server/client 远程模式（pi 的 modes/rpc、packages/server、packages/client）。
- OAuth 设备流与凭据加密存储。
- pi 的 40 家厂商枚举式 provider 与其模型生成脚本（models.generated）。
- 图像多模态输入（截图、粘贴图、图片工具）与图片结果渲染。
- 与 pi 会话文件的互导兼容（语义同构即可，不承诺格式兼容）。
- 自更新机制、版本检查、telemetry 上报、evals 包。
- MCP 接入（pi 本身也未做）。
- llama/本地模型扩展（pi 的 extensions/llama）。
- Bun/Node 双运行时分发。

## 什么算完成

`checklist.md` 全部条目可勾选。核心判据：faux transport 端到端跑通 interactive 与 print 两种模式，pytest 全绿，零真实网络依赖。
