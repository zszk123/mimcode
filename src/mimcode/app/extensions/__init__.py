"""mimcode.app.extensions：Python 插件（目录扫描 + register(ctx) 约定）。"""

from mimcode.app.extensions.loader import (
    ExtensionDiagnostic,
    LoadedExtension,
    LoadExtensionsResult,
    apply_extensions,
    extension_names,
    load_extensions,
)
from mimcode.app.extensions.types import EventHandler, ExtensionContext

__all__ = [
    "EventHandler",
    "ExtensionContext",
    "ExtensionDiagnostic",
    "LoadedExtension",
    "LoadExtensionsResult",
    "apply_extensions",
    "extension_names",
    "load_extensions",
]
