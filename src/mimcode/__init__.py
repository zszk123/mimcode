"""mimcode - 终端 AI 编程助手（pi 的 Python 迁移版）。

分层结构（自底向上，单向依赖）：
- mimcode.types      消息/内容块/事件的数据模型
- mimcode.provider   协议即 provider（OpenAI 协议 / Claude 协议）
- mimcode.agent      agent loop 双循环与工具注册表
- mimcode.app        会话/压缩/技能/插件/命令等应用层
- mimcode.cli / tui  CLI 参数分发与交互式界面
"""

__version__ = "0.1.0"
