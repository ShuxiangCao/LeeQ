"""
Helpers for optional notebook and AI dependencies.

The runtime package should be importable without notebook UI packages or AI
integrations. Features that genuinely require those dependencies should fail
when called, not when unrelated LeeQ modules are imported.
"""

from __future__ import annotations

from typing import Any, Callable, TypeVar

from leeq.utils.utils import Singleton

F = TypeVar("F", bound=Callable[..., Any])


def missing_optional_dependency(dependency: str, feature: str) -> ImportError:
    """Build a clear error for optional features that are unavailable."""
    return ImportError(
        f"{feature} requires optional dependency '{dependency}'. "
        "Install LeeQ with the relevant optional extras or install requirements-dev.txt."
    )


def _missing_optional_callable(dependency: str, feature: str) -> Callable[..., Any]:
    """Return a function that raises a clear import error when called."""

    def _missing(*args: Any, **kwargs: Any) -> Any:
        raise missing_optional_dependency(dependency, feature)

    return _missing


def _identity_inspection_decorator(*decorator_args: Any, **decorator_kwargs: Any) -> Callable[[F], F] | F:
    """No-op replacement for k_agents inspection decorators."""
    if len(decorator_args) == 1 and callable(decorator_args[0]) and not decorator_kwargs:
        return decorator_args[0]

    def _decorate(func: F) -> F:
        return func

    return _decorate


try:
    from k_agents.inspection.decorator import text_inspection, visual_inspection
except ImportError:
    text_inspection = _identity_inspection_decorator
    visual_inspection = _identity_inspection_decorator


try:
    from IPython.display import display
except ImportError:

    def display(*args: Any, **kwargs: Any) -> None:
        """No-op fallback for non-notebook runtime environments."""
        return None


try:
    from k_agents.execution.agent import execute_experiment_from_instruction
except ImportError:
    execute_experiment_from_instruction = _missing_optional_callable(
        "k_agents", "execute_experiment_from_instruction"
    )


try:
    from k_agents.execution.stage_execution import get_exp_from_var_table
except ImportError:
    get_exp_from_var_table = _missing_optional_callable("k_agents", "get_exp_from_var_table")


try:
    from k_agents.io_interface import code_to_html, dict_to_html, display_chat
except ImportError:
    code_to_html = _missing_optional_callable("k_agents", "code_to_html")
    dict_to_html = _missing_optional_callable("k_agents", "dict_to_html")
    display_chat = _missing_optional_callable("k_agents", "display_chat")


try:
    from mllm import Chat
except ImportError:
    Chat = _missing_optional_callable("mllm", "Chat")
