"""Agent loop（迁移自 pi packages/agent/src/agent-loop.ts 的完整主路径）。

双循环结构（对齐 pi runLoop）：
- 外循环：agent 本应停止时轮询 follow-up 队列，有消息则继续
- 内循环：流式助手响应 → 工具批量执行 → steering 注入 → 下一轮

关键语义（对齐 pi）：
- 截断保护：stopReason=length 时整批工具调用判错（参数可能被截断，
  不安全执行），错误文本提示模型重发完整参数
- 并行批：tool_execution_end 按完成序发出；工具结果消息按助手
  消息中的原始顺序发出（源序回填）
- 钩子：before/after 工具拦截、每轮后换模型/上下文、should_stop
- 流契约：失败不抛异常（StreamError 事件），loop 把 error/aborted
  终态作为 agent 的正常结束处理

并发事件安全：事件经 asyncio.Queue 串行化（对齐 pi EventStream.push
的入队语义）；agent_loop 是 async generator，agent_end 保证最后产出。
后台任务异常经哨兵 + await task 传播给消费者。
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from mimcode.agent.tools.base import AgentTool, ToolAbortedError
from mimcode.agent.validate import validate_tool_arguments
from mimcode.types import (
    AgentEnd,
    AgentEvent,
    AgentMessage,
    AgentStart,
    AssistantMessage,
    AssistantStreamEvent,
    LlmContext,
    MessageEnd,
    MessageStart,
    MessageUpdate,
    ModelInfo,
    StreamOptions,
    StreamTextDelta,
    StreamTextEnd,
    StreamTextStart,
    StreamThinkingDelta,
    StreamThinkingEnd,
    StreamThinkingStart,
    StreamToolCallDelta,
    StreamToolCallEnd,
    StreamToolCallStart,
    TextBlock,
    ThinkingLevel,
    ToolCallBlock,
    ToolExecutionEnd,
    ToolExecutionStart,
    ToolExecutionUpdate,
    ToolResult,
    ToolResultMessage,
    TurnEnd,
    TurnStart,
    UserContentBlock,
)

StreamFn = Callable[[ModelInfo, LlmContext, StreamOptions], AsyncIterator[AssistantStreamEvent]]
"""流函数（Provider.stream 满足此形态）。"""

MessageSource = Callable[[], Awaitable[Sequence[AgentMessage]]]
"""异步消息队列钩子形态（steering / follow-up）。"""

_SENTINEL = object()
"""队列哨兵：后台运行结束标记（非 AgentEvent）。"""

EventSink = Callable[[Any], None]
"""事件入队函数（接受 AgentEvent 或哨兵）。"""

_PARTIAL_EVENT_CLASSES = (
    StreamTextStart,
    StreamTextDelta,
    StreamTextEnd,
    StreamThinkingStart,
    StreamThinkingDelta,
    StreamThinkingEnd,
    StreamToolCallStart,
    StreamToolCallDelta,
    StreamToolCallEnd,
)
"""partial 事件类元组（isinstance 窄化用）。"""


@dataclass
class AgentContext:
    """agent 运行上下文（对齐 pi AgentContext）。"""

    system_prompt: str | None = None
    messages: list[AgentMessage] = field(default_factory=list)
    tools: list[AgentTool] = field(default_factory=list)

    def to_llm_context(self, llm_messages: list[AgentMessage]) -> LlmContext:
        """构建 LLM 调用上下文（消息经 convert_to_llm 转换后）。"""
        return LlmContext(
            system_prompt=self.system_prompt,
            messages=llm_messages,
            tools=[tool.to_spec() for tool in self.tools],
        )


# ---------------------------------------------------------------------------
# 钩子上下文与结果（对齐 pi types.ts L56-147）
# ---------------------------------------------------------------------------


@dataclass
class BeforeToolCallContext:
    """before 钩子入参。"""

    assistant_message: AssistantMessage
    tool_call: ToolCallBlock
    args: dict[str, Any]
    context: AgentContext


@dataclass
class BeforeToolCallResult:
    """before 钩子返回：block=True 阻断执行。"""

    block: bool = False
    reason: str | None = None
    terminate: bool = False


@dataclass
class AfterToolCallContext:
    """after 钩子入参（result 为执行结果原值）。"""

    assistant_message: AssistantMessage
    tool_call: ToolCallBlock
    args: dict[str, Any]
    result: ToolResult
    is_error: bool
    context: AgentContext


@dataclass
class AfterToolCallResult:
    """after 钩子返回：逐字段覆盖（省略字段保留原值，无深合并）。"""

    content: list[UserContentBlock] | None = None
    details: Any = None
    is_error: bool | None = None
    terminate: bool | None = None


@dataclass
class TurnEndContext:
    """轮级钩子入参（should_stop / prepare_next_turn 共用形态）。"""

    message: AssistantMessage
    tool_results: list[ToolResultMessage]
    context: AgentContext
    new_messages: list[AgentMessage]


@dataclass
class TurnUpdate:
    """prepare_next_turn 返回：替换下一轮运行状态。"""

    context: AgentContext | None = None
    model: ModelInfo | None = None
    thinking_level: ThinkingLevel | None = None


@dataclass
class AgentLoopConfig:
    """agent loop 配置（对齐 pi AgentLoopConfig 的 v1 子集）。

    钩子契约（对齐 pi）：不得抛异常；失败以安全回落值表达。
    """

    model: ModelInfo
    stream_fn: StreamFn
    api_key: str | None = None
    thinking_level: ThinkingLevel | None = None
    max_tokens: int | None = None

    # 消息转换边界（v1 默认恒等；T8 compaction 经 transform_context 接入）
    transform_context: Callable[[list[AgentMessage]], list[AgentMessage]] | None = None
    convert_to_llm: Callable[[list[AgentMessage]], list[AgentMessage]] | None = None

    # 队列钩子
    get_steering_messages: MessageSource | None = None
    get_follow_up_messages: MessageSource | None = None

    # 工具执行钩子
    tool_execution: str = "parallel"  # "sequential" | "parallel"
    before_tool_call: (
        Callable[[BeforeToolCallContext], Awaitable[BeforeToolCallResult | None]] | None
    ) = None
    after_tool_call: (
        Callable[[AfterToolCallContext], Awaitable[AfterToolCallResult | None]] | None
    ) = None

    # 轮级钩子
    prepare_next_turn: Callable[[TurnEndContext], Awaitable[TurnUpdate | None]] | None = None
    should_stop_after_turn: Callable[[TurnEndContext], Awaitable[bool]] | None = None


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------


async def _pump(queue: asyncio.Queue, task: asyncio.Task) -> AsyncIterator[AgentEvent]:
    """从队列产出事件直到哨兵；消费者提前退出时取消后台任务。"""
    try:
        while True:
            item = await queue.get()
            if item is _SENTINEL:
                break
            yield item
    finally:
        if not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task


async def agent_loop(
    prompts: Sequence[AgentMessage],
    context: AgentContext,
    config: AgentLoopConfig,
    signal: asyncio.Event | None = None,
) -> AsyncIterator[AgentEvent]:
    """带新提示启动 agent（对齐 pi agentLoop）。

    Yields:
        AgentEvent 序列，以 agent_end（含本次新增消息）终结。
    """
    queue: asyncio.Queue = asyncio.Queue()

    def emit(item: Any) -> None:
        queue.put_nowait(item)

    task = asyncio.create_task(_run_agent_loop(prompts, context, config, signal, emit))
    async for event in _pump(queue, task):
        yield event
    await task  # 传播后台异常


async def agent_loop_continue(
    context: AgentContext,
    config: AgentLoopConfig,
    signal: asyncio.Event | None = None,
) -> AsyncIterator[AgentEvent]:
    """从当前上下文继续（不注入新消息；对齐 pi agentLoopContinue）。

    上下文最后一条消息必须是 user 或 toolResult（LLM 协议要求）。
    """
    if not context.messages:
        raise ValueError("Cannot continue: no messages in context")
    if context.messages[-1].role == "assistant":
        raise ValueError("Cannot continue from message role: assistant")

    queue: asyncio.Queue = asyncio.Queue()

    def emit(item: Any) -> None:
        queue.put_nowait(item)

    task = asyncio.create_task(_run_agent_loop_continue(context, config, signal, emit))
    async for event in _pump(queue, task):
        yield event
    await task


async def _run_agent_loop(
    prompts: Sequence[AgentMessage],
    context: AgentContext,
    config: AgentLoopConfig,
    signal: asyncio.Event | None,
    emit: EventSink,
) -> None:
    """后台任务体：注入提示并运行双循环。"""
    try:
        new_messages: list[AgentMessage] = list(prompts)
        current: AgentContext = AgentContext(
            system_prompt=context.system_prompt,
            messages=[*context.messages, *prompts],
            tools=context.tools,
        )
        emit(AgentStart())
        emit(TurnStart())
        for prompt in prompts:
            emit(MessageStart(message=prompt))
            emit(MessageEnd(message=prompt))
        await _run_loop(current, new_messages, config, signal, emit)
        emit(AgentEnd(messages=new_messages))
    finally:
        emit(_SENTINEL)


async def _run_agent_loop_continue(
    context: AgentContext,
    config: AgentLoopConfig,
    signal: asyncio.Event | None,
    emit: EventSink,
) -> None:
    """后台任务体：继续模式。"""
    try:
        new_messages: list[AgentMessage] = []
        emit(AgentStart())
        emit(TurnStart())
        await _run_loop(context, new_messages, config, signal, emit)
        emit(AgentEnd(messages=new_messages))
    finally:
        emit(_SENTINEL)


# ---------------------------------------------------------------------------
# 双循环核心（对齐 pi runLoop L155-275）
# ---------------------------------------------------------------------------


async def _run_loop(
    initial_context: AgentContext,
    new_messages: list[AgentMessage],
    initial_config: AgentLoopConfig,
    signal: asyncio.Event | None,
    emit: Callable[[AgentEvent], None],
) -> None:
    """双循环主逻辑（外循环 follow-up；内循环工具与 steering）。"""
    context = initial_context
    config = initial_config
    first_turn = True
    # 起点即检查 steering（用户可能在等待期间输入，对齐 pi）
    pending = await _drain_source(config.get_steering_messages)

    while True:
        has_more_tool_calls = True

        while has_more_tool_calls or pending:
            if first_turn:
                first_turn = False
            else:
                emit(TurnStart())

            # steering 消息注入（下一轮助手响应之前）
            if pending:
                for message in pending:
                    emit(MessageStart(message=message))
                    emit(MessageEnd(message=message))
                    context.messages.append(message)
                    new_messages.append(message)
                pending = []

            # 流式助手响应
            message = await _stream_assistant_response(context, config, signal, emit)
            new_messages.append(message)

            if message.stop_reason in ("error", "aborted"):
                emit(TurnEnd(message=message, tool_results=[]))
                return

            # 工具调用批
            tool_calls = [block for block in message.content if isinstance(block, ToolCallBlock)]
            tool_results: list[ToolResultMessage] = []
            has_more_tool_calls = False
            if tool_calls:
                batch = (
                    await _fail_truncated_batch(tool_calls, emit)
                    if message.stop_reason == "length"
                    else await _execute_tool_calls(
                        context, message, tool_calls, config, signal, emit
                    )
                )
                tool_results = batch.messages
                has_more_tool_calls = not batch.terminate

                for result in tool_results:
                    context.messages.append(result)
                    new_messages.append(result)

            emit(TurnEnd(message=message, tool_results=tool_results))

            # 轮级钩子：prepare_next_turn（换模型/上下文/思考级别）
            turn_context = TurnEndContext(
                message=message,
                tool_results=tool_results,
                context=context,
                new_messages=new_messages,
            )
            if config.prepare_next_turn is not None:
                update = await config.prepare_next_turn(turn_context)
                if update is not None:
                    if update.context is not None:
                        context = update.context
                    config = _replace_config(
                        config,
                        model=update.model or config.model,
                        thinking_level=(
                            config.thinking_level
                            if update.thinking_level is None
                            else update.thinking_level
                        ),
                    )

            # 轮级钩子：should_stop_after_turn（优雅停止）
            if config.should_stop_after_turn is not None:
                if await config.should_stop_after_turn(turn_context):
                    return

            pending = await _drain_source(config.get_steering_messages)

        # agent 本应停止：轮询 follow-up
        follow_up = await _drain_source(config.get_follow_up_messages)
        if follow_up:
            pending = follow_up
            continue
        break


def _replace_config(
    config: AgentLoopConfig,
    *,
    model: ModelInfo,
    thinking_level: ThinkingLevel | None,
) -> AgentLoopConfig:
    """复制配置并替换模型/思考级别（对齐 pi 的 config spread）。"""
    return AgentLoopConfig(
        model=model,
        stream_fn=config.stream_fn,
        api_key=config.api_key,
        thinking_level=thinking_level,
        max_tokens=config.max_tokens,
        transform_context=config.transform_context,
        convert_to_llm=config.convert_to_llm,
        get_steering_messages=config.get_steering_messages,
        get_follow_up_messages=config.get_follow_up_messages,
        tool_execution=config.tool_execution,
        before_tool_call=config.before_tool_call,
        after_tool_call=config.after_tool_call,
        prepare_next_turn=config.prepare_next_turn,
        should_stop_after_turn=config.should_stop_after_turn,
    )


async def _drain_source(source: MessageSource | None) -> list[AgentMessage]:
    """拉取一次队列钩子（未配置返回空；契约要求不抛异常）。"""
    if source is None:
        return []
    return list(await source())


# ---------------------------------------------------------------------------
# 流式助手响应（对齐 pi streamAssistantResponse L281-372）
# ---------------------------------------------------------------------------


async def _stream_assistant_response(
    context: AgentContext,
    config: AgentLoopConfig,
    signal: asyncio.Event | None,
    emit: Callable[[AgentEvent], None],
) -> AssistantMessage:
    """流式请求一轮助手响应并转换为 Agent 事件。

    partial 引用语义（对齐 pi）：流开始时把 partial 消息 append 进
    context.messages，增量事件就地更新该消息；终态替换原位。
    """
    messages = context.messages
    if config.transform_context is not None:
        messages = config.transform_context(messages)
    llm_messages = config.convert_to_llm(messages) if config.convert_to_llm else messages

    llm_context = context.to_llm_context(llm_messages)
    options = StreamOptions(
        api_key=config.api_key,
        signal=signal,
        thinking_level=config.thinking_level,
        max_tokens=config.max_tokens,
    )

    final_message: AssistantMessage | None = None
    added_partial = False

    async for event in config.stream_fn(config.model, llm_context, options):
        if event.type == "start":
            context.messages.append(event.partial)
            added_partial = True
            emit(MessageStart(message=event.partial))

        elif isinstance(event, _PARTIAL_EVENT_CLASSES):
            # partial 就地更新（provider 翻译器维护同一引用）
            if added_partial and context.messages:
                context.messages[-1] = event.partial
            emit(MessageUpdate(message=event.partial, assistant_event=event))

        elif event.type == "done":
            final_message = event.message
            break

        elif event.type == "error":
            final_message = event.error
            break

    # 流中断兜底（未到终态事件）
    if final_message is None:
        fallback = (
            context.messages[-1]
            if added_partial
            and context.messages
            and isinstance(context.messages[-1], AssistantMessage)
            else None
        )
        final_message = fallback or AssistantMessage(
            content=[],
            api=config.model.api,
            provider=config.model.provider,
            model=config.model.id,
            stop_reason="error",
            error_message="Stream ended without a terminal event",
        )
        if final_message.stop_reason == "pending":
            final_message.stop_reason = "error"

    if added_partial and context.messages:
        context.messages[-1] = final_message
    else:
        context.messages.append(final_message)
        emit(MessageStart(message=final_message))
    emit(MessageEnd(message=final_message))
    return final_message


# ---------------------------------------------------------------------------
# 工具批执行（对齐 pi executeToolCalls / 截断保护）
# ---------------------------------------------------------------------------


@dataclass
class ExecutedBatch:
    """一批工具调用的执行结果。"""

    messages: list[ToolResultMessage]
    terminate: bool


@dataclass
class ToolCallOutcome:
    """单个工具调用的终局结果。"""

    tool_call: ToolCallBlock
    result: ToolResult
    is_error: bool


async def _fail_truncated_batch(
    tool_calls: list[ToolCallBlock],
    emit: Callable[[AgentEvent], None],
) -> ExecutedBatch:
    """截断保护（对齐 pi failToolCallsFromTruncatedMessage）。

    stopReason=length 时参数可能不完整：整批判错不执行，
    错误文本提示模型重发完整参数。
    """
    outcomes: list[ToolCallOutcome] = []
    for tool_call in tool_calls:
        emit(
            ToolExecutionStart(
                tool_call_id=tool_call.id,
                tool_name=tool_call.name,
                args=tool_call.arguments,
            )
        )
        error_text = (
            f'Tool call "{tool_call.name}" was not executed: the response hit '
            "the output token limit, so its arguments may be truncated. "
            "Re-issue the tool call with complete arguments."
        )
        outcome = ToolCallOutcome(
            tool_call=tool_call,
            result=ToolResult(content=[TextBlock(text=error_text)]),
            is_error=True,
        )
        _emit_execution_end(outcome, emit)
        outcomes.append(outcome)
    messages = [_emit_result_message(outcome, emit) for outcome in outcomes]
    return ExecutedBatch(messages=messages, terminate=False)


async def _execute_tool_calls(
    context: AgentContext,
    assistant_message: AssistantMessage,
    tool_calls: list[ToolCallBlock],
    config: AgentLoopConfig,
    signal: asyncio.Event | None,
    emit: Callable[[AgentEvent], None],
) -> ExecutedBatch:
    """分发工具批：全 sequential 配置或批内含 sequential 工具时串行。"""
    has_sequential = any(
        tool.execution_mode == "sequential"
        for tool_call in tool_calls
        for tool in context.tools
        if tool.name == tool_call.name
    )
    if config.tool_execution == "sequential" or has_sequential:
        return await _execute_batch_sequential(
            context, assistant_message, tool_calls, config, signal, emit
        )
    return await _execute_batch_parallel(
        context, assistant_message, tool_calls, config, signal, emit
    )


async def _execute_batch_sequential(
    context: AgentContext,
    assistant_message: AssistantMessage,
    tool_calls: list[ToolCallBlock],
    config: AgentLoopConfig,
    signal: asyncio.Event | None,
    emit: Callable[[AgentEvent], None],
) -> ExecutedBatch:
    """串行批（对齐 pi executeToolCallsSequential）：逐个执行，完成即发。"""
    outcomes: list[ToolCallOutcome] = []
    messages: list[ToolResultMessage] = []

    for tool_call in tool_calls:
        outcome = await _run_single_tool_call(
            context, assistant_message, tool_call, config, signal, emit
        )
        outcomes.append(outcome)
        messages.append(_emit_result_message(outcome, emit))
        if signal is not None and signal.is_set():
            break

    return ExecutedBatch(messages=messages, terminate=_should_terminate(outcomes))


async def _execute_batch_parallel(
    context: AgentContext,
    assistant_message: AssistantMessage,
    tool_calls: list[ToolCallBlock],
    config: AgentLoopConfig,
    signal: asyncio.Event | None,
    emit: Callable[[AgentEvent], None],
) -> ExecutedBatch:
    """并行批（对齐 pi executeToolCallsParallel）。

    tool_execution_start/end 按完成序发出（各任务内 emit）；
    工具结果消息按助手消息源序发出（gather 保序回填）。
    """
    tasks: list[asyncio.Task] = [
        asyncio.create_task(
            _run_single_tool_call(context, assistant_message, call, config, signal, emit)
        )
        for call in tool_calls
    ]
    outcomes: list[ToolCallOutcome] = list(await asyncio.gather(*tasks))

    messages: list[ToolResultMessage] = [
        _emit_result_message(outcome, emit) for outcome in outcomes
    ]
    return ExecutedBatch(messages=messages, terminate=_should_terminate(outcomes))


async def _run_single_tool_call(
    context: AgentContext,
    assistant_message: AssistantMessage,
    tool_call: ToolCallBlock,
    config: AgentLoopConfig,
    signal: asyncio.Event | None,
    emit: Callable[[AgentEvent], None],
) -> ToolCallOutcome:
    """准备（验证+before 钩子）→ 执行 → 终结（after 钩子）单个调用。

    所有失败路径（未找到/验证失败/被阻断/执行异常/中止）都归一为
    错误结果并发出 tool_execution_start/end（对齐 pi 的 immediate 语义）。
    """
    emit(
        ToolExecutionStart(
            tool_call_id=tool_call.id,
            tool_name=tool_call.name,
            args=tool_call.arguments,
        )
    )
    tool = next(
        (candidate for candidate in context.tools if candidate.name == tool_call.name), None
    )
    if tool is None:
        return _emit_execution_end(
            ToolCallOutcome(tool_call, _error_result(f"Tool {tool_call.name} not found"), True),
            emit,
        )

    try:
        args = validate_tool_arguments(tool, tool_call)
    except ValueError as exc:
        return _emit_execution_end(ToolCallOutcome(tool_call, _error_result(str(exc)), True), emit)

    if config.before_tool_call is not None:
        before = await config.before_tool_call(
            BeforeToolCallContext(
                assistant_message=assistant_message,
                tool_call=tool_call,
                args=args,
                context=context,
            )
        )
        if before is not None and before.block:
            result = _error_result(before.reason or "Tool execution was blocked")
            result.terminate = before.terminate
            return _emit_execution_end(ToolCallOutcome(tool_call, result, True), emit)

    if signal is not None and signal.is_set():
        return _emit_execution_end(
            ToolCallOutcome(tool_call, _error_result("Operation aborted"), True), emit
        )

    async def on_update(partial: ToolResult) -> None:
        emit(
            ToolExecutionUpdate(
                tool_call_id=tool_call.id,
                tool_name=tool_call.name,
                args=tool_call.arguments,
                partial_result=partial,
            )
        )

    try:
        result = await tool.execute(tool_call.id, args, signal, on_update)
        is_error = False
    except ToolAbortedError:
        return _emit_execution_end(
            ToolCallOutcome(tool_call, _error_result("Operation aborted"), True), emit
        )
    except Exception as exc:  # noqa: BLE001 - 工具错误归一边界（对齐 pi catch(error)）
        return _emit_execution_end(
            ToolCallOutcome(tool_call, _error_result(str(exc) or type(exc).__name__), True),
            emit,
        )

    if config.after_tool_call is not None:
        after = await config.after_tool_call(
            AfterToolCallContext(
                assistant_message=assistant_message,
                tool_call=tool_call,
                args=args,
                result=result,
                is_error=is_error,
                context=context,
            )
        )
        if after is not None:
            if after.content is not None:
                result.content = after.content
            if after.details is not None:
                result.details = after.details
            if after.is_error is not None:
                is_error = after.is_error
            if after.terminate is not None:
                result.terminate = after.terminate

    return _emit_execution_end(ToolCallOutcome(tool_call, result, is_error), emit)


def _emit_execution_end(
    outcome: ToolCallOutcome, emit: Callable[[AgentEvent], None]
) -> ToolCallOutcome:
    """发出 tool_execution_end 并透传 outcome。"""
    emit(
        ToolExecutionEnd(
            tool_call_id=outcome.tool_call.id,
            tool_name=outcome.tool_call.name,
            result=outcome.result,
            is_error=outcome.is_error,
        )
    )
    return outcome


def _emit_result_message(
    outcome: ToolCallOutcome, emit: Callable[[AgentEvent], None]
) -> ToolResultMessage:
    """发出 toolResult 消息的 message_start/end 并返回消息。"""
    message = ToolResultMessage(
        tool_call_id=outcome.tool_call.id,
        tool_name=outcome.tool_call.name,
        content=outcome.result.content,
        details=outcome.result.details,
        usage=outcome.result.usage,
        is_error=outcome.is_error,
    )
    emit(MessageStart(message=message))
    emit(MessageEnd(message=message))
    return message


def _should_terminate(outcomes: list[ToolCallOutcome]) -> bool:
    """早停判定（对齐 pi shouldTerminateToolBatch）：全部 terminate 才停。"""
    return bool(outcomes) and all(outcome.result.terminate for outcome in outcomes)


def _error_result(message: str) -> ToolResult:
    """错误工具结果。"""
    return ToolResult(content=[TextBlock(text=message)])
