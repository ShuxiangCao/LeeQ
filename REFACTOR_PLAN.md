# LeeQ Refactor Plan

This plan is for improving repository structure, package boundaries, import safety, and maintainability without changing scientific behavior.

## Implementation Status

Status: implemented on `dev_experiments`.

Completed:
- Moved the runtime package to `src/leeq`.
- Added explicit lazy top-level APIs in `leeq.__init__`, `leeq.api`, and `leeq.experiments.__init__`.
- Moved optional AI integration code to `leeq.integrations.ai` with `leeq.utils.ai` compatibility wrappers.
- Moved pyGSTi integration code to `leeq.integrations.pygsti` with `leeq.experiments.integrations.pygsti` compatibility wrappers.
- Moved Chronicle viewer implementation to `leeq.apps.chronicle_viewer` with `leeq.chronicle.viewer` compatibility wrappers.
- Moved built-in experiment domains to canonical packages:
  - `leeq.experiments.calibrations`
  - `leeq.experiments.characterizations`
  - `leeq.experiments.gates`
  - `leeq.experiments.tomography`
  - `leeq.experiments.hamiltonian_tomography`
  - `leeq.experiments.optimal_control`
- Kept legacy `leeq.experiments.builtin...` imports available through compatibility packages.
- Split the large priority experiment modules listed in this plan into smaller class-focused modules while keeping legacy module filenames as wrappers.
- Updated CI path filters, lint paths, docs, notebooks, examples, and scripts for the new layout.

Validation completed:

```bash
.venv/bin/python -m pytest
# 969 passed, 35 skipped, 39 deselected, 19 warnings

PYTHON=.venv/bin/python bash ci_scripts/lint.sh
# Blocking linting checks passed.

.venv/bin/python -m compileall -q src tests
# passed

.venv/bin/mkdocs build --strict
# passed

.venv/bin/python scripts/test_notebook_infrastructure.py --check-deps
# passed

git diff --check
# passed
```

## Goals

- Make LeeQ easier to install, test, and package correctly.
- Keep `import leeq` lightweight and predictable.
- Separate runtime core, optional integrations, apps, notebooks, and development tooling.
- Reduce circular imports and broad `import *` API surfaces.
- Split large experiment modules into smaller, testable units.
- Preserve backward compatibility during the transition wherever practical.

## Non-Goals

- Do not rewrite experiment algorithms as part of the structural refactor.
- Do not remove public APIs without a compatibility layer and deprecation path.
- Do not combine packaging, behavior changes, and large module splits in one PR.
- Do not move notebooks or docs in a way that breaks published documentation without updating links.

## Current Pain Points

- The package lives at repo root as `leeq/`, and tests patch `sys.path` in `tests/conftest.py`.
- `leeq/__init__.py` imports broad module trees with `import *`, which makes imports heavier and increases circular import risk.
- Optional AI, notebook, viewer, and integration features are mixed into runtime paths.
- Large experiment modules are hard to review and test in isolation.
- `utils` contains cross-cutting functionality that should be split by responsibility.
- CI currently targets `main` and `develop`, but not the active `dev_experiments` branch.
- Multiple requirements files duplicate dependency sources that already exist in `pyproject.toml`.

## Target Repository Layout

```text
.
├── src/
│   └── leeq/
│       ├── __init__.py
│       ├── api.py
│       ├── config.py
│       ├── core/
│       ├── experiments/
│       │   ├── base.py
│       │   ├── sweeper.py
│       │   ├── calibrations/
│       │   ├── characterizations/
│       │   ├── gates/
│       │   └── tomography/
│       ├── theory/
│       ├── setups/
│       ├── compiler/
│       ├── chronicle/
│       ├── integrations/
│       │   ├── ai/
│       │   └── pygsti/
│       ├── apps/
│       │   └── chronicle_viewer/
│       └── utils/
├── tests/
├── docs/
├── notebooks/
├── examples/
├── scripts/
└── ci_scripts/
```

## Dependency Rules

- `core` may depend on standard runtime utilities only.
- `theory` should stay pure: no experiment runtime, setup, UI, notebook, or AI imports.
- `experiments` may depend on `core`, `theory`, `setups`, and optional integrations through guarded helpers.
- `compiler` may depend on `core` primitives and setup/compiler interfaces, but not built-in experiments.
- `integrations` may depend on optional third-party packages and must fail late with clear `ImportError`s.
- `apps` may depend on UI libraries and should not be imported by runtime package paths.
- `utils` should contain narrow shared utilities only; domain-specific helpers should move near their domain.

## Phase 1: Packaging Foundation

Move to a `src/` layout with no behavior changes.

Tasks:
- Move `leeq/` to `src/leeq/`.
- Update `pyproject.toml` package config:

```toml
packages = [{ include = "leeq", from = "src" }]
```

- Remove `sys.path.insert(...)` from `tests/conftest.py` after confirming editable installs work.
- Update CI, docs build, notebook tooling, and scripts for the new package path.
- Run full tests on Python 3.10, 3.11, and 3.12.

Acceptance criteria:
- `pip install -e .` installs from `src/leeq`.
- Tests pass without modifying `sys.path`.
- `python -c "import leeq"` works from outside the repo root.
- Existing public imports still work.

## Phase 2: Public API Cleanup

Make top-level imports explicit and lightweight.

Tasks:
- Replace broad imports in `src/leeq/__init__.py` with explicit stable exports.
- Add `src/leeq/api.py` for convenience exports if needed.
- Keep compatibility paths for common imports such as `Experiment`, `Sweeper`, `setup`, and `basic_run`.
- Stop importing integrations from the top-level package.
- Add tests that assert `import leeq` does not import optional AI, notebook, viewer, or heavy integration modules.

Acceptance criteria:
- `import leeq` is fast and does not require optional extras.
- Public imports used by tests and notebooks still work.
- Optional integrations are imported only when explicitly requested.

## Phase 3: Optional Feature Boundaries

Separate optional AI, notebook, UI, and integration code from runtime core.

Tasks:
- Move AI experiment-generation code under `src/leeq/integrations/ai/`.
- Move pygsti code under `src/leeq/integrations/pygsti/`.
- Move Chronicle viewer app entry points under `src/leeq/apps/chronicle_viewer/`.
- Keep optional dependency helpers in one place and extend them as needed.
- Review `pyproject.toml` extras:
  - `ai`
  - `notebooks`
  - `viewer`
  - `docs`
  - `dev`

Acceptance criteria:
- Runtime install can import and run core tests without AI/notebook/viewer packages.
- Installing extras enables the related feature tests.
- Missing optional features fail late with clear errors.

## Phase 4: Experiment Package Restructure

Move built-in experiments into clearer domain packages while preserving compatibility imports.

Proposed mapping:

```text
experiments/builtin/basic/calibrations/      -> experiments/calibrations/
experiments/builtin/basic/characterizations/ -> experiments/characterizations/
experiments/builtin/multi_qubit_gates/       -> experiments/gates/
experiments/builtin/tomography/              -> experiments/tomography/
experiments/builtin/hamiltonian_tomography/  -> experiments/hamiltonian_tomography/
```

Tasks:
- Move one domain at a time.
- Add compatibility modules that re-export old paths.
- Update internal imports to use new canonical paths.
- Update docs and notebooks after each domain move.

Acceptance criteria:
- Old imports continue to work.
- New imports are documented and used internally.
- Full tests pass after each domain move.

## Phase 5: Split Large Modules

Break down large files into cohesive modules.

Priority targets:
- `conditional_stark_ai.py`
- `resonator_spectroscopy.py`
- `ac_stark_shift.py`
- `sizzel/hamiltonian_tomography.py`
- `sizzel/calibration.py`

Example split for conditional Stark:

```text
experiments/gates/conditional_stark/
├── __init__.py
├── experiments.py
├── analysis.py
├── fitting.py
├── plotting.py
├── ai_inspection.py
└── tuning_env.py
```

Acceptance criteria:
- Public classes/functions keep their old import paths through compatibility exports.
- New tests target extracted helpers directly where useful.
- No behavior changes beyond import paths and internal organization.

## Phase 6: Tooling and CI

Make quality checks easier to trust and faster to run.

Tasks:
- Add CI for `dev_experiments` or standardize on `develop`.
- Add a fast PR job:
  - install package
  - lint blocking checks
  - focused unit tests
  - optional import smoke tests
- Keep slow, notebook, and integration jobs separate.
- Gradually tighten Ruff rules for touched files or new directories.
- Decide whether `requirements*.txt` are generated artifacts or maintained by hand.

Acceptance criteria:
- PR checks catch packaging/import mistakes.
- Slow and optional tests are still available without blocking every small PR.
- Dependency definitions have one clear source of truth.

## Compatibility Strategy

- Keep old import paths as thin compatibility modules for at least one release cycle.
- Emit `DeprecationWarning` only after docs and notebooks have moved to new imports.
- Maintain a migration guide with old-to-new path mappings.
- Prefer mechanical moves with tests before behavior refactors.

## Validation Commands

Run these after each phase:

```bash
pip install -e .
python -c "import leeq; print(leeq.__file__)"
.venv/bin/python -m pytest tests/experiments/test_optional_imports.py
.venv/bin/python -m pytest
PYTHON=.venv/bin/python bash ci_scripts/lint.sh
.venv/bin/python -m compileall -q src tests
git diff --check
```

For notebook/docs-impacting phases:

```bash
mkdocs build --strict
python scripts/test_notebook_infrastructure.py --check-deps
```

## Recommended PR Sequence

1. `src/` layout and test import cleanup.
2. Explicit top-level API exports.
3. CI branch and packaging verification updates.
4. Optional integration package boundaries.
5. One experiment domain move with compatibility exports.
6. Repeat domain moves.
7. Split `conditional_stark_ai.py`.
8. Split remaining large experiment modules.
9. Tighten lint/type rules for newly structured areas.

## Risks

- Moving modules can break notebooks, docs, and user imports if compatibility exports are incomplete.
- Broad `import *` usage can hide missing dependencies during refactors.
- Optional extras can regress if tests run only in a fully provisioned dev environment.
- Large mechanical moves make code review difficult unless behavior changes are avoided.

## First Action

Start with a no-behavior-change `src/` layout PR. It gives the highest packaging confidence and makes later architecture work safer.
