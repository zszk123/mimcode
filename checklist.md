# mimcode 验收清单（checklist）

> 全部条目可勾选、可观测。spec 砍掉的具体值（路径、阈值、命令行、错误行为）都在这里落地。

## 工程基线

- [ ] `uv run mimcode --help` 退出码 0，输出含 `-p`、`--model`、`--list-models`、`-c`、`--fork` 字样
- [ ] `uv run mimcode --version` 输出形如 `mimcode 0.x.y` 的版本号
- [ ] `uv run pytest` 全绿且总用例数 ≥ 60
- [ ] `rg "except Exception" src/ -U` 无「裸 Exception 捕获」（`except Exception as e` 后直接 raise/pass 的模式为 0 条）
- [ ] 检查命令单入口存在（`scripts/check.py` 或等价物），一条命令跑 ruff + mypy + pytest
- [ ] `pyproject.toml` 中 `requires-python >= 3.11`，依赖含 `openai`、`anthropic`、`prompt-toolkit`、`rich`、`pydantic`
- [ ] 测试套件运行期间无真实网络访问：在禁网环境（或 monkeypatch 掉 SDK HTTP 层）下 `uv run pytest` 仍全绿

## 类型系统（T2）

- [ ] `pytest tests/test_types.py -k roundtrip` 通过：每类消息与事件 JSON 序列化→反序列化后与原对象相等
- [ ] `rg "agent_start|turn_end|tool_execution_update" src/mimcode/types` 各返回 ≥ 1 条（事件命名与 pi 对齐）
- [ ] 用量模型含 cache_read / cache_write / reasoning 字段，缺字段时缺省为 0 且序列化稳定

## Provider 层（T3、T4）

- [ ] OpenAIProtocolProvider：faux fixture 含 tool_calls 增量分片（参数跨 chunk 拆开）时，最终 toolCall 块的 arguments 能被 `json.loads` 解析
- [ ] OpenAIProtocolProvider：faux fixture 中 delta 带 `reasoning_content` 字段时，产生 thinking_start/thinking_delta/thinking_end 事件，且最终消息含 thinking 内容块
- [ ] ClaudeProtocolProvider：faux fixture 模拟 content_block_delta（thinking_delta/text_input_delta/tool_use 输入增量）时，三种事件流均正确产出
- [ ] 流函数契约：faux fixture 模拟 SDK 抛出连接错误时，`stream()` 不抛异常，而是产出 error 事件 + stopReason 为 error 的最终消息
- [ ] 配置文件位于 `~/.mimcode/config.toml`，项目级 `.mimcode/config.toml` 可覆盖同名端点条目（单测验证合并顺序）
- [ ] API key 解析顺序：环境变量（`OPENAI_API_KEY` / `ANTHROPIC_API_KEY` 及端点自定义变量名）优先于配置文件（单测注入两种来源断言取值）
- [ ] `uv run mimcode --list-models` 列出预置目录条目，每行含 provider 协议、模型 id、上下文窗口；配置文件新增自定义条目后该条目出现在列表中
- [ ] 预置目录至少含：openai 官方、anthropic 官方、deepseek、zhipu、kimi 共 5 组端点条目（`rg "deepseek" src/mimcode/provider/catalog.py` ≥ 1 条）

## 内置工具（T5）

- [ ] bash 工具：Windows 上执行 `echo hello` 走 PowerShell（`cmd /c` / bash 直调的反证单测存在），输出含 `hello`
- [ ] read 工具：读取带中文内容的 UTF-8 文件无乱码（单测断言字节相等）
- [ ] edit 工具：目标字符串在文件中出现 2 次时执行失败，错误信息含出现次数与唯一性提示；恰好 1 次时替换成功且 diff 记录新旧内容
- [ ] bash 输出截断：构造 > 截断上限的 stdout 时，结果含截断标记且长度不超过上限 + 标记长度（上限取具体数值写入常量并在单测中引用）
- [ ] grep/find/ls 三工具各有 ≥ 1 个通过的行为单测（含忽略规则：.gitignore 命中的文件不出现在结果中）

## Agent loop（T6）

- [ ] 双循环：faux 流第一轮带工具调用、用户 steering 队列有一条消息时，事件序列为 agent_start → message(用户) → 助手流 → tool_execution_* → steering 消息 message_start/end → 第二轮助手流 → agent_end（单测按序断言事件类型列表）
- [ ] 截断保护：stopReason 为 length 且含 2 个工具调用时，两个工具均不执行，各产出 isError 的工具结果，错误文本含「未执行/截断」语义（单测断言文本关键词）
- [ ] 并行批：同消息 2 个可并行工具的 `tool_execution_end` 先到先发，而工具结果消息按助手消息中的原始顺序发出（单测用延时工具验证乱序完成）
- [ ] before 钩子返回 block 时，工具不执行且产出含 reason 的错误结果（单测断言 reason 原样出现在结果文本中）
- [ ] prepare_next_turn 换模型：第一轮结束后注入新模型，第二轮流式调用收到的模型 id 等于新值（faux 侧断言）

## 会话（T7）

- [ ] 会话目录为 `~/.mimcode/sessions/<cwd 编码名>/`，文件名含创建时间戳与会话 id，扩展名 `.jsonl`
- [ ] append-only：一 talks 中新增消息只追加行，已写入行不被改写（单测记录行数与已有行哈希后追加再比对）
- [ ] 分叉：从含 2 条消息的会话 fork 后，新会话首 2 行与源会话逐行相等，且源会话文件未被修改
- [ ] 发现容错：会话目录混入 1 个损坏 jsonl 与 1 个超大文件后，列表会话仍返回其余合法会话（单测构造后断言）
- [ ] `--continue`：第一次 print 模式会话落盘后，第二次带 `-c` 启动，faux 侧收到请求的 messages 前缀包含首次会话全部消息

## Compaction（T8）

- [ ] 触发阈值与默认值：reserveTokens=16384、keepRecentTokens=20000 作为具名常量存在且单测引用（`rg "16384" src/mimcode/app/compaction.py` ≥ 1 条）
- [ ] 构造上下文占用超过窗口 90% 的 fixture 会话后，compact 触发，产出摘要消息且保留近期消息条数对应 keepRecent 语义（单测断言裁剪后保留集）
- [ ] 无用量数据时按字符估算路径有单测覆盖（中文按多字节计数的断言）

## Skills（T9）

- [ ] SKILL.md 校验：名称 > 64 字符或描述 > 1024 字符时该技能被拒并进入诊断列表（单测两个边界各 1 例）
- [ ] `.gitignore` 命中的技能目录不参与发现（单测）
- [ ] 格式化注入：加载 2 个技能后系统提示含两个技能名（单测断言技能名出现在最终系统提示文本中）

## Extensions（T11）

- [ ] 插件加载：`.mimcode/extensions/` 放置含 `register(ctx)` 的插件文件后启动，其注册的自定义工具出现在工具注册表（单测临时目录驱动）
- [ ] 故障隔离：一个插件 `register` 抛异常时，其余插件仍加载成功，异常插件出现在诊断输出中（单测双插件一坏一好）
- [ ] `/extensions` 命令输出已加载插件名列表（命令单测断言）

## Slash 命令（T10）

- [ ] `/model`、`/thinking`、`/resume`、`/fork`、`/compact`、`/skills`、`/extensions`、`/exit`、`/help` 九个命令注册齐全（`rg "slash" src/mimcode/app/commands.py` 或命令表测试逐项断言）
- [ ] `/model <id>` 后下一次流式请求使用的模型 id 等于所设值（faux 断言）
- [ ] 输入未知命令 `/nope` 时得到错误提示而非崩溃（单测断言退出码/输出）

## TUI（T12、T13）

- [ ] 渲染管线：headless 驱动一段含 thinking + text + toolCall 的 faux 流，渲染产出包含：折叠态思考区、markdown 渲染后的正文、工具执行块（名称 + 耗时）（快照/文本断言）
- [ ] 流式增量：text_delta 分 3 片到达时渲染更新被调用 ≥ 3 次且最终拼接完整（管线计数单测）
- [ ] diff 渲染：edit 工具结果渲染产出含删除行（红/减号语义）与新增行（绿/加号语义）的标记（文本断言 `-`/`+` 行存在）
- [ ] 中断：agent 运行中触发中断键后，流以 stopReason=aborted 结束且 UI 不残留加载动画状态（headless 断言）
- [ ] 主题：至少两套主题常量定义，切换后用户/助手/工具/错误四类语义色取自当前主题（单测切换前后取色对比）

## 端到端验收（必须全过）

- [ ] **E2E-1 print 全链路**：faux transport 下 `uv run mimcode -p "测试提示" --model faux-test`，stdout 依次出现流式正文与最终完整回复，进程退出码 0
- [ ] **E2E-2 会话续接**：接 E2E-1，再跑 `uv run mimcode -p "继续" --model faux-test -c`，faux 收到的消息序列包含上一会话的消息（断言脚本检查 faux 记录文件）
- [ ] **E2E-3 interactive 冒烟**：headless（PTY 模拟）启动 interactive，输入一条提示，观察到助手回复渲染完成，再输入 `/exit` 正常退出，退出码 0
- [ ] **E2E-4 compaction 路径**：构造超长会话 fixture 后以 interactive headless 打开并发送一条消息，渲染产出含压缩摘要标记
- [ ] **E2E-5 Windows shell**：在 Windows 本机运行 E2E-1 时附带一次真实工具调用（echo），输出含工具执行块与命令输出

## 交付前终检

- [ ] `rg "TODO|FIXME|XXX" src/` 条数为 0 或全部有跟踪注释（含去向 issue/任务号）
- [ ] 所有对外 CLI 参数在 `--help` 输出中有对应行（help 文本与 argparse/typer 定义一一对应）
- [ ] 冷启动计时：`Measure-Command { uv run mimcode --version }` 与到交互首帧（headless 计时钩子）均 < 2 秒
- [ ] `checklist.md` 自身全部勾选，且 spec.md 的 Out of Scope 未被偷偷实现（抽查 git log 无相关实现提交）
