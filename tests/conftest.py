"""共享测试装置。"""

from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixtures_dir() -> Path:
    """fixture JSON 文件目录。"""
    return FIXTURES_DIR
