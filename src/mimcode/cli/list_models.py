"""``--list-models`` 的输出渲染。

checklist 对应项：每行含 provider 协议、模型 id、上下文窗口；
配置文件新增自定义条目后该条目出现在列表中。
"""

from __future__ import annotations

import sys

from mimcode.provider.registry import Registry


def format_model_line(endpoint_name: str, protocol: str, model_id: str, context_window: int) -> str:
    """单行模型条目：``<endpoint> (<protocol>)  <model_id>  <window>``。"""
    return f"{endpoint_name} ({protocol})\t{model_id}\t{context_window}"


def list_models(registry: Registry, stream=sys.stdout) -> None:
    """输出全部模型目录到指定流。"""
    for endpoint_name, provider in sorted(registry.providers().items()):
        for model in provider.get_models():
            print(
                format_model_line(endpoint_name, model.api, model.id, model.context_window),
                file=stream,
            )
