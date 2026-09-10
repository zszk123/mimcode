"""T13 高保真渲染测试。

checklist 对应项：
- diff 渲染：edit 工具结果渲染产出含删除行（红/减号语义）与
  新增行（绿/加号语义）的标记
- 主题：至少两套主题常量定义，切换后 user/assistant/tool/error
  四类语义色取自当前主题（切换前后取色对比）
- markdown 渲染：正文含 markdown 语法时渲染形态正确
"""

from __future__ import annotations

import io
import json
from typing import Any

from rich.console import Console

from mimcode.agent.loop import AgentContext, AgentLoopConfig
from mimcode.agent.tools.base import AgentTool
from mimcode.provider.faux import FauxFixture, FauxOpenAIProvider
from mimcode.tui.app import InteractiveApp
from mimcode.tui.diff import diff_renderable
from mimcode.tui.markdown import render_markdown
from mimcode.tui.renderer import RenderAction, RendererPipeline
from mimcode.tui.rich_presenter import RichPresenter
from mimcode.tui.theme import DARK, LIGHT, get_theme

SAMPLE_DIFF = """--- a/src/app.py
+++ b/src/app.py
@@ -1,3 +1,3 @@
 def main():
-    return 1
+    return 2
     # end
"""


# ---------------------------------------------------------------------------
# 主题（checklist：两套主题 + 切换取色对比）
# ---------------------------------------------------------------------------


def test_two_themes_defined() -> None:
    """checklist：至少两套主题常量。"""
    assert DARK.name == "dark"
    assert LIGHT.name == "light"
    assert DARK is not LIGHT
    assert get_theme("dark") is DARK
    assert get_theme("light") is LIGHT
    # 未知名回落默认
    assert get_theme("nonexistent") is DARK


def test_theme_role_colors_differ_between_themes() -> None:
    """checklist：切换后 user/assistant/tool/error 取色随主题。"""
    for role in ("user", "assistant", "tool", "error"):
        dark_style = DARK.styles[role]
        light_style = LIGHT.styles[role]
        assert dark_style != light_style, f"role {role} 两主题取色应不同"

    # presenter 取色随当前主题切换
    dark_presenter = RichPresenter(theme=DARK, stream=io.StringIO())
    light_presenter = RichPresenter(theme=LIGHT, stream=io.StringIO())
    for role in ("user", "assistant", "tool", "error", "thinking"):
        assert dark_presenter.style_for(role) == DARK.styles[role]
        assert light_presenter.style_for(role) == LIGHT.styles[role]

    # 运行时切换：同 presenter 切主题后取色变化
    presenter = RichPresenter(theme=DARK, stream=io.StringIO())
    before = presenter.style_for("user")
    presenter.set_theme(LIGHT)
    after = presenter.style_for("user")
    assert before == DARK.styles["user"]
    assert after == LIGHT.styles["user"]
    assert before != after


# ---------------------------------------------------------------------------
# diff 渲染（checklist：删除红/新增绿）
# ---------------------------------------------------------------------------


def test_diff_render_semantic_colors() -> None:
    """checklist：diff 渲染的减/加行语义（红/绿）。"""
    # 语义色：Text spans 携带红/绿样式
    text_obj = diff_renderable(SAMPLE_DIFF)
    span_styles = {span.style for span in text_obj.spans}
    assert "bold red" in span_styles
    assert "bold green" in span_styles
    assert "bold cyan" in span_styles

    # 终端形态：record + export_text(styles=True) 不依赖终端能力
    console = Console(record=True, width=100, highlight=False)
    console.print(diff_renderable(SAMPLE_DIFF))
    ansi = console.export_text(styles=True)
    assert "\x1b[" in ansi
    assert "-    return 1" in ansi
    assert "+    return 2" in ansi
    assert "31" in ansi  # 红基色码
    assert "32" in ansi  # 绿基色码


def test_diff_renderable_text_content() -> None:
    """diff 渲染对象保留行序与内容。"""
    text_obj = diff_renderable(SAMPLE_DIFF)
    plain = text_obj.plain
    assert "-    return 1" in plain
    assert "+    return 2" in plain
    assert "@@ -1,3 +1,3 @@" in plain


def test_empty_diff_renders() -> None:
    """空 diff 不崩。"""
    plain = diff_renderable("").plain
    assert plain == ""


# ---------------------------------------------------------------------------
# markdown 渲染（checklist：正文 markdown 形态）
# ---------------------------------------------------------------------------


def test_markdown_renders_headers_and_code() -> None:
    """markdown：标题与代码块渲染形态。"""
    md = "# 标题\n\n正文 **加粗** 段落。\n\n```python\nprint('hi')\n```\n"
    rendered = render_markdown(md, width=80)
    # 标题独立成行
    assert "标题" in rendered
    # 代码块内容保留
    assert "print('hi')" in rendered
    # 加粗文本保留（ANSI 无关的文本层）
    assert "加粗" in rendered
    # 代码块呈现为缩进/围栏形态（多行）
    assert "\n" in rendered


def test_markdown_incomplete_syntax_tolerant() -> None:
    """流式截断的 markdown（未闭合代码块）不抛异常。"""
    partial = "```python\nprint('hi')"
    rendered = render_markdown(partial, width=80)
    assert "print" in rendered


def test_markdown_plain_paragraph() -> None:
    """普通段落渲染为文本。"""
    rendered = render_markdown("普通回复文本", width=80)
    assert "普通回复文本" in rendered


# ---------------------------------------------------------------------------
# RichPresenter 全链路（注入 app）
# ---------------------------------------------------------------------------


class EchoTool(AgentTool):
    name = "echo"
    description = "echo"

    def parameters_schema(self) -> dict[str, Any]:
        return {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}

    async def execute(self, tool_call_id, args, signal=None, on_update=None):
        return self.text_result(f"echo:{args.get('text', '')}")


def openai_tool_call_chunks(call_id: str, name: str, args: str) -> list[dict[str, Any]]:
    return [
        {
            "id": f"chatcmpl-{call_id}",
            "model": "faux-gpt",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": call_id,
                                "type": "function",
                                "function": {"name": name, "arguments": ""},
                            }
                        ],
                    },
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": f"chatcmpl-{call_id}",
            "model": "faux-gpt",
            "choices": [
                {
                    "index": 0,
                    "delta": {"tool_calls": [{"index": 0, "function": {"arguments": args}}]},
                    "finish_reason": "tool_calls",
                }
            ],
        },
        {"id": f"chatcmpl-{call_id}", "model": "faux-gpt", "choices": []},
    ]


def openai_text_chunks(text: str) -> list[dict[str, Any]]:
    return [
        {
            "id": "chatcmpl-text",
            "model": "faux-gpt",
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": text},
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": "chatcmpl-text",
            "model": "faux-gpt",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        },
        {"id": "chatcmpl-text", "model": "faux-gpt", "choices": []},
    ]


class ScriptedProvider(FauxOpenAIProvider):
    def __init__(self, scripts: list[FauxFixture]) -> None:
        super().__init__(scripts[0])
        self.scripts = scripts
        self._index = 0

    async def _raw_chunks(self, payload, options):
        self.requests.append(payload)
        index = min(self._index, len(self.scripts) - 1)
        self._index += 1
        for chunk in self.scripts[index].chunks:
            yield chunk


async def test_app_with_rich_presenter_full_turn() -> None:
    """全链路：Rich presenter 注入后 markdown 正文 + 主题色行落地。"""
    provider = ScriptedProvider(
        [
            FauxFixture(
                protocol="openai",
                model="faux-gpt",
                chunks=openai_tool_call_chunks("c1", "echo", json.dumps({"text": "hi"})),
            ),
            FauxFixture(
                protocol="openai",
                model="faux-gpt",
                chunks=openai_text_chunks("完成。\n\n## 小结\n\n要点内容。"),
            ),
        ]
    )
    output = io.StringIO()
    presenter = RichPresenter(stream=output, width=80)
    context = AgentContext(tools=[EchoTool()])
    config = AgentLoopConfig(model=provider.get_models()[0], stream_fn=provider.stream)

    app = InteractiveApp(
        cwd=".",
        agent_context=context,
        config_factory=lambda: config,
        stream=output,
        presenter=presenter,
    )
    events = await app.run_agent_turn("做点什么")

    captured = output.getvalue()
    # 用户行（主题色行内容）
    assert "你: 做点什么" in captured
    # 工具块
    assert "⚙ echo" in captured
    assert "✓ echo" in captured
    # markdown 正文：标题渲染形态
    assert "小结" in captured
    assert "要点内容" in captured
    # 事件正常收尾
    assert events[-1].type == "agent_end"
    assert app.state.busy is False


def test_presenter_styles_in_output() -> None:
    """presenter 主题色：终端渲染带 ANSI 样式；纯文本态可读。"""
    # 终端态：record 模式导出带样式的文本（不依赖真实终端）
    record_console = Console(record=True, width=80, highlight=False)
    presenter = RichPresenter.__new__(RichPresenter)
    presenter.theme = DARK
    presenter.console = record_console
    presenter.apply_action(RenderAction(kind="write_line", text="你: 带色用户行", style="user"))
    presenter.apply_action(RenderAction(kind="write_line", text="  ✗ bash: 失败", style="error"))
    ansi = record_console.export_text(styles=True)
    assert "\x1b[" in ansi
    assert "带色用户行" in ansi
    assert "失败" in ansi
    assert DARK.styles["user"] == "bold cyan"  # 主题常量稳定

    # 无终端态（StringIO 捕获）：纯文本仍可读
    plain_out = io.StringIO()
    plain_presenter = RichPresenter(stream=plain_out, width=80)
    plain_presenter.apply_action(RenderAction(kind="write_line", text="你: 纯文本", style="user"))
    assert "你: 纯文本" in plain_out.getvalue()


def test_presenter_footer_and_stream() -> None:
    """footer 主题色行；非 TTY 流式打字区不外泄（定稿 markdown 承载）。"""
    output = io.StringIO()
    presenter = RichPresenter(stream=output, width=80)
    presenter.apply_action(RenderAction(kind="footer", text="m1 | 3 轮 | 100 tokens"))
    presenter.apply_action(RenderAction(kind="stream_chunk", text="流式"))
    presenter.apply_action(RenderAction(kind="stream_chunk", text="片段"))
    presenter.apply_action(RenderAction(kind="clear_stream"))
    presenter.apply_action(RenderAction(kind="write_line", text="mim: 定稿正文", style="assistant"))
    presenter.apply_action(RenderAction(kind="clear_busy"))

    captured = output.getvalue()
    assert "3 轮" in captured
    assert "流式片段" not in captured  # 打字区不重复输出
    assert "定稿正文" in captured
    assert captured.endswith("\n")


def test_presenter_stream_typewriter_and_erase_on_tty() -> None:
    """TTY 流式：打字机追加 + clear_stream ANSI 擦除 + 跟踪复位。"""

    class FakeTty(io.StringIO):
        def isatty(self) -> bool:
            return True

    output = FakeTty()
    presenter = RichPresenter(stream=output, width=80)
    presenter.apply_action(RenderAction(kind="stream_chunk", text="打字机"))
    assert "打字机" in output.getvalue()

    presenter.apply_action(RenderAction(kind="clear_stream"))
    captured = output.getvalue()
    # 擦除转义：上移 + 回行首 + 清屏尾
    assert "\x1b[" in captured
    assert "J" in captured
    # 跟踪复位：空区域再次 clear 不再发出转义
    marker = len(output.getvalue())
    presenter.apply_action(RenderAction(kind="clear_stream"))
    assert len(output.getvalue()) == marker


def test_presenter_erasable_line_skipped_when_not_tty() -> None:
    """非 TTY：瞬时指示行（思考中）不进入输出。"""
    output = io.StringIO()
    presenter = RichPresenter(stream=output, width=80)
    presenter.apply_action(
        RenderAction(kind="write_line", text="  ✻ 思考中…", style="thinking", erasable=True)
    )
    assert "思考中" not in output.getvalue()


def test_presenter_renders_diff_action() -> None:
    """write_line 携带 diff：预览行 + 语义色 diff 块。"""
    output = io.StringIO()
    presenter = RichPresenter(stream=output, width=100)
    presenter.apply_action(
        RenderAction(
            kind="write_line",
            text="  ✓ edit: Successfully replaced 1 block(s) in a.py.",
            style="tool",
            diff=SAMPLE_DIFF,
        )
    )
    captured = output.getvalue()
    assert "✓ edit" in captured
    assert "-    return 1" in captured
    assert "+    return 2" in captured


def test_presenter_error_action() -> None:
    """error 动作渲染错误色行。"""
    output = io.StringIO()
    presenter = RichPresenter(stream=output, width=80)
    presenter.apply_action(RenderAction(kind="error", text="连接失败"))
    assert "连接失败" in output.getvalue()


def test_presenter_renders_diff() -> None:
    """presenter.render_diff：checklist 的删除/新增行标记。"""
    output = io.StringIO()
    presenter = RichPresenter(stream=output, width=100)
    rendered = presenter.render_diff(SAMPLE_DIFF)
    assert "-    return 1" in rendered
    assert "+    return 2" in rendered


def test_pipeline_actions_carry_styles() -> None:
    """管线动作带语义角色（主题渲染数据源）。"""
    from mimcode.tui.state import TuiState
    from mimcode.types import (
        AssistantMessage,
        MessageEnd,
        MessageStart,
        TextBlock,
        ToolExecutionEnd,
        ToolExecutionStart,
        ToolResult,
        UserMessage,
    )

    pipeline = RendererPipeline(TuiState())
    partial = AssistantMessage(
        content=[], api="openai", provider="p", model="m", stop_reason="pending"
    )

    user_action = pipeline.handle(MessageStart(message=UserMessage(content="hi")))[0]
    assert user_action.style == "user"

    tool_actions = pipeline.handle(ToolExecutionStart(tool_call_id="c", tool_name="t", args={}))
    assert tool_actions[0].style == "tool"

    end_actions = pipeline.handle(
        ToolExecutionEnd(
            tool_call_id="c",
            tool_name="t",
            result=ToolResult(content=[TextBlock(text="r")]),
            is_error=True,
        )
    )
    assert end_actions[0].style == "error"

    assistant_actions = pipeline.handle(
        MessageEnd(
            message=AssistantMessage(
                content=[TextBlock(text="正文")],
                api="openai",
                provider="p",
                model="m",
                stop_reason="stop",
            )
        )
    )
    assert any(action.style == "assistant" for action in assistant_actions)

    del partial, pipeline
