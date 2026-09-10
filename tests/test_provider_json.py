"""流式 JSON 部分解析器测试。"""

from mimcode.provider._json import partial_json_loads


def test_complete_json_passthrough() -> None:
    """完整 JSON 直接解析。"""
    assert partial_json_loads('{"command": "ls"}') == {"command": "ls"}


def test_empty_input() -> None:
    """空输入返回空对象。"""
    assert partial_json_loads(None) == {}
    assert partial_json_loads("") == {}
    assert partial_json_loads("   ") == {}


def test_truncated_string_value() -> None:
    """值字符串被截断：补引号后可解析。"""
    assert partial_json_loads('{"command": "ec') == {"command": "ec"}


def test_truncated_escape_sequence() -> None:
    """值字符串截断在转义序列中间：剥离半个转义。"""
    assert partial_json_loads('{"path": "C:\\') == {"path": "C:"}


def test_trailing_comma() -> None:
    """悬挂逗号：剥离后闭合容器。"""
    assert partial_json_loads('{"a": 1,') == {"a": 1}


def test_colon_without_value() -> None:
    """冒号后缺值：补占位 null。"""
    assert partial_json_loads('{"command": ') == {"command": None}


def test_truncated_nested_containers() -> None:
    """嵌套容器被截断：逆序闭合。"""
    assert partial_json_loads('{"a": [1, 2') == {"a": [1, 2]}
    assert partial_json_loads('{"outer": {"inner": "x') == {"outer": {"inner": "x"}}


def test_open_object_only() -> None:
    """仅有开括号：闭合为空对象。"""
    assert partial_json_loads("{") == {}


def test_non_dict_top_level() -> None:
    """顶层非对象（如纯字符串）：返回空对象。"""
    assert partial_json_loads('"just a string') == {}


def test_unrecoverable_garbage() -> None:
    """不可修复的垃圾输入：返回空对象而非抛异常。"""
    assert partial_json_loads("{invalid json : : :") == {}
