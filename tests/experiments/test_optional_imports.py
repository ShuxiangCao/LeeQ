"""Regression tests for optional notebook and AI dependencies."""

import os
import subprocess
import sys
import textwrap


def _run_with_blocked_optional_imports(script: str) -> subprocess.CompletedProcess:
    """Run a Python snippet while making optional imports unavailable."""
    block_imports = """
import builtins
import sys

_real_import = builtins.__import__
_blocked_roots = {"fibers", "k_agents", "mllm", "IPython"}


def _blocked_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name.split(".")[0] in _blocked_roots:
        raise ImportError(f"blocked optional dependency {name}")
    return _real_import(name, globals, locals, fromlist, level)


builtins.__import__ = _blocked_import
for _module_name in list(sys.modules):
    if _module_name.split(".")[0] in _blocked_roots:
        sys.modules.pop(_module_name, None)
"""
    env = os.environ.copy()
    env["LEEQ_SUPPRESS_LOGGING"] = "true"
    return subprocess.run(
        [sys.executable, "-c", block_imports + "\n" + textwrap.dedent(script)],
        cwd=os.getcwd(),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_import_leeq_without_optional_ai_or_ipython():
    result = _run_with_blocked_optional_imports(
        """
        import leeq
        from leeq.experiments.experiments import K_AGENTS_AVAILABLE

        assert K_AGENTS_AVAILABLE is False
        assert leeq is not None
        """
    )

    assert result.returncode == 0, result.stderr


def test_minimal_experiment_runs_without_k_agents():
    result = _run_with_blocked_optional_imports(
        """
        from leeq.experiments.experiments import LeeQAIExperiment


        class DemoExperiment(LeeQAIExperiment):
            def run(self, value=3):
                self.value = value
                return value


        experiment = DemoExperiment(value=7)
        assert experiment.value == 7
        """
    )

    assert result.returncode == 0, result.stderr


def test_representative_decorator_modules_import_without_k_agents():
    result = _run_with_blocked_optional_imports(
        """
        from leeq.experiments.builtin.basic.calibrations import rabi
        from leeq.experiments.builtin.basic.calibrations import resonator_spectroscopy
        from leeq.experiments.builtin.multi_qubit_gates import conditional_stark_ai
        from leeq.experiments.builtin.multi_qubit_gates.sizzel import calibration

        assert rabi is not None
        assert resonator_spectroscopy is not None
        assert conditional_stark_ai is not None
        assert calibration is not None
        """
    )

    assert result.returncode == 0, result.stderr


def test_ai_modules_import_without_optional_ai_packages():
    result = _run_with_blocked_optional_imports(
        """
        from leeq.utils.ai import translation_agent
        from leeq.utils.ai.experiment_generation import data_analysis
        from leeq.utils.ai.experiment_generation import data_visualization
        from leeq.utils.ai.experiment_generation import experiment_generation
        from leeq.utils.ai.experiment_generation import pulse_sequences

        assert translation_agent is not None
        assert data_analysis is not None
        assert data_visualization is not None
        assert experiment_generation is not None
        assert pulse_sequences is not None
        """
    )

    assert result.returncode == 0, result.stderr


def test_optional_helpers_are_noop_or_fail_late_without_dependencies():
    result = _run_with_blocked_optional_imports(
        """
        from leeq.utils.optional_dependencies import Chat
        from leeq.utils.optional_dependencies import dict_to_html
        from leeq.utils.optional_dependencies import text_inspection
        from leeq.utils.optional_dependencies import visual_inspection


        @text_inspection
        def text_checked():
            return "text"


        @visual_inspection("prompt")
        def visual_checked():
            return "visual"


        assert text_checked() == "text"
        assert visual_checked() == "visual"

        try:
            Chat("prompt")
        except ImportError as exc:
            assert "mllm" in str(exc)
        else:
            raise AssertionError("Chat should fail late when mllm is unavailable")

        try:
            dict_to_html({"a": 1})
        except ImportError as exc:
            assert "k_agents" in str(exc)
        else:
            raise AssertionError("dict_to_html should fail late when k_agents is unavailable")
        """
    )

    assert result.returncode == 0, result.stderr
