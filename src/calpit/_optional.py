"""Imports of optional dependencies with errors that name the extra to install."""

import importlib
import types


def import_optional(module_name: str, extra: str, feature: str) -> types.ModuleType:
    """Imports an optional dependency.

    Args:
        module_name: The module to import, such as "torch".
        extra: The calpit extra that installs it, such as "torch".
        feature: What needs the module, for the error message.

    Returns:
        The imported module.

    Raises:
        ImportError: If the module is not installed.
    """
    try:
        return importlib.import_module(module_name)
    except ImportError as error:
        raise ImportError(
            f"{feature} requires the optional dependency {module_name}. "
            f"Install it with: pip install 'calpit[{extra}]'"
        ) from error
