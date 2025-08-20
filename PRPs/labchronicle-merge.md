# PRP: Merge LabChronicle Package into LeeQ

## Overview
Integrate the LabChronicle package directly into the LeeQ codebase to eliminate the external dependency on the git+https://github.com/ShuxiangCao/LabChronicle package. This will improve maintainability, simplify installation, and enable better integration control.

**Business Value**: 
- Eliminates external dependency risks (repository availability, breaking changes)
- Simplifies installation process for users
- Enables direct optimization and customization for LeeQ's needs
- Improves CI/CD reliability by removing external git dependency

**Scope**: 
- Vendor the entire LabChronicle package into LeeQ
- Update all imports from `labchronicle` to internal module
- Maintain backward compatibility for all existing LeeQ functionality
- Remove labchronicle from external requirements

## Context & Research

### Codebase Patterns

#### Current LabChronicle Usage in LeeQ
- **Base Class Integration**: leeq/core/base.py:3 - `from labchronicle import LoggableObject`
  - LeeQObject inherits from LoggableObject, making it foundational
  - All LeeQ objects depend on this for persistence and tracking

- **Decorator Usage Pattern**: 30+ files use decorators
  - `log_and_record`: Records object state and function execution
  - `log_event`: Records only function args/returns
  - `register_browser_function`: Registers visualization functions
  - Example: leeq/experiments/builtin/basic/calibrations/rabi.py:4

- **Chronicle Singleton**: leeq/experiments/experiments.py:9
  - Used for experiment management and logging coordination

- **Testing Pattern**: tests/epii/test_leeq_backend_integration.py
  - Tests verify LabChronicle integration works correctly
  - Mock patterns exist for testing without actual persistence

### Key LabChronicle Components

**Core Files Structure**:
```
LabChronicle/labchronicle/
├── __init__.py          # Main exports
├── chronicle.py         # Singleton manager class
├── core.py             # LoggableObject base class
├── decorators.py       # log_and_record, log_event, register_browser_function
├── logger.py           # Logging setup
├── utils.py            # Helper functions
└── handlers/           # Storage backends
    ├── handlers.py     # Base handler interface
    ├── hdf5.py        # HDF5 persistence
    ├── memory.py      # In-memory storage
    └── dummy.py       # No-op handler
```

### External Resources

- **Python Vendoring Best Practices**: https://discuss.python.org/t/can-vendoring-dependencies-in-a-build-be-officially-supported/13622
  - Recommends automated import rewriting
  - Use relative imports for vendored code
  - Include vendored code in version control

- **pip Vendoring Tool**: https://github.com/pypa/vendoring
  - Reference implementation for dependency vendoring
  - Automates import rewriting process

- **Import Vendoring Patterns**: https://stackoverflow.com/questions/52538252/import-vendored-dependencies-in-python-package-without-modifying-sys-path-or-3rd
  - Use `from .vendor import module` pattern
  - Avoid sys.path manipulation

## System Architecture

### Modular Structure
```
LeeQ Package Structure After Merge:

┌─────────────────────────────────────┐
│            LeeQ Package             │
├─────────────────────────────────────┤
│  leeq/                              │
│  ├── core/                          │
│  │   ├── base.py (uses chronicle)   │
│  │   └── ...                        │
│  ├── experiments/                   │
│  │   └── ... (uses decorators)      │
│  ├── chronicle/  ◄── NEW            │
│  │   ├── __init__.py                │
│  │   ├── chronicle.py               │
│  │   ├── core.py                    │
│  │   ├── decorators.py              │
│  │   ├── logger.py                  │
│  │   ├── utils.py                   │
│  │   └── handlers/                  │
│  │       ├── __init__.py            │
│  │       ├── handlers.py            │
│  │       ├── hdf5.py                │
│  │       ├── memory.py              │
│  │       └── dummy.py               │
│  └── ...                            │
└─────────────────────────────────────┘

Key Components:
- leeq.chronicle: Internal chronicle module (vendored)
- Import Mapping: labchronicle → leeq.chronicle
- No external labchronicle dependency
- All functionality preserved
```

### Data Flow
```
Import Resolution Flow:

1. Old Import Path:
   External: labchronicle.LoggableObject
   ↓
2. New Import Path:
   Internal: leeq.chronicle.LoggableObject
   ↓
3. Module Loading:
   Python imports from leeq/chronicle/
   ↓
4. Functionality:
   Same API, internal implementation

Import Transformation Examples:
- from labchronicle import LoggableObject 
  → from leeq.chronicle import LoggableObject
- from labchronicle import log_and_record
  → from leeq.chronicle import log_and_record
- import labchronicle
  → import leeq.chronicle as labchronicle
```

### Integration Architecture
```
Dependency Changes:
- Remove: labchronicle @ git+https://github.com/ShuxiangCao/LabChronicle
- Keep: All labchronicle dependencies (already in requirements.txt)
  - h5py (already present)
  - decorator (already present)
  - pyyaml (already present)
  - fsspec (already present)
  - numpy (already present)
  - ipywidgets (already present)
  - markdownify (needs to be added)

Import Update Locations (30+ files):
- Core: leeq/core/base.py
- Experiments: leeq/experiments/*.py
- Setups: leeq/setups/setup_base.py
- Tests: tests/**/*.py
- Notebooks: notebooks/**/*.py
- Benchmarks: benchmark/**/*.py
```

## Technical Requirements

### Functional Requirements
- FR1: Vendor complete LabChronicle package into leeq/chronicle/
- FR2: Update all imports from `labchronicle` to `leeq.chronicle`
- FR3: Preserve all existing LabChronicle functionality
- FR4: Ensure all tests pass without external dependency
- FR5: Update documentation to reflect internal module

### Non-Functional Requirements
- Testing: All existing tests must pass
- Documentation: Update import examples in docs/

### Constraints
- Technical: Must work with Python 3.8+
- Dependencies: Only add markdownify to requirements.txt
- Structure: Follow LeeQ's existing module organization

## Implementation Plan

### Phase 1: Vendor LabChronicle
1. Copy LabChronicle/labchronicle/ to leeq/chronicle/
2. Update leeq/chronicle/__init__.py to maintain same exports
3. Add markdownify to requirements.txt
4. Remove labchronicle git dependency from requirements.txt

### Phase 2: Update Imports
1. Create import mapping script:
   ```python
   IMPORT_MAP = {
       'from labchronicle': 'from leeq.chronicle',
       'import labchronicle': 'import leeq.chronicle as labchronicle'
   }
   ```
2. Apply to all Python files in:
   - leeq/**/*.py (30+ files)
   - tests/**/*.py (10+ files)  
   - notebooks/**/*.py (5+ files)
   - benchmark/**/*.py (2+ files)

### Phase 3: Update Tests
1. Verify existing tests work with internal module
2. Update any mocking/patching of labchronicle
3. Ensure chronicle tests are integrated into LeeQ test suite

### Phase 4: Documentation
1. Update CLAUDE.md to remove labchronicle dependency note
2. Update import examples in docs/
3. Update installation instructions

## Validation Strategy

### Testing Approach
- **Unit Testing**: Verify each chronicle component works
- **Integration Testing**: Ensure LeeQ objects still log correctly
- **E2E Testing**: Run full experiment workflows
- **Import Testing**: Verify no labchronicle imports remain

### Validation Gates
```bash
# 1. No external labchronicle imports
grep -r "from labchronicle\|import labchronicle" leeq/ tests/ --exclude-dir=chronicle

# 2. All tests pass
pytest tests/ -v

# 3. Integration tests specifically
pytest tests/integration_tests/ -v

# 4. Linting passes
bash ./ci_scripts/lint.sh

# 5. Documentation builds
mkdocs build
```

### Manual Verification
```python
# Test basic functionality
from leeq.chronicle import LoggableObject, log_and_record
from leeq.core.base import LeeQObject

# Verify inheritance works
obj = LeeQObject("test")
assert isinstance(obj, LoggableObject)

# Verify decorators work
@log_and_record
def test_func(self):
    return "success"
```

## Dependencies & Configuration

### External Dependencies
```
# Add to requirements.txt:
markdownify  # Required by labchronicle for markdown conversion

# Remove from requirements.txt:
# labchronicle @ git+https://github.com/ShuxiangCao/LabChronicle
```

### Configuration Requirements
```
# No environment variables needed
# Chronicle config remains the same
# Handler configurations unchanged
```

## Risk Analysis

### Technical Risks
- **Risk**: Import path conflicts during transition
  - Impact: Medium
  - Mitigation: Use automated script, test thoroughly

- **Risk**: Missing chronicle functionality  
  - Impact: High
  - Mitigation: Copy entire package, verify with tests

- **Risk**: Future LabChronicle updates not incorporated
  - Impact: Low
  - Mitigation: Document vendoring date, can manually sync if needed

### Integration Risks
- **Risk**: Circular import issues with internal module
  - Impact: Medium
  - Mitigation: Maintain same module structure, test imports

- **Risk**: Test mocking patterns break
  - Impact: Low
  - Mitigation: Update mock paths in tests

## Success Criteria
- [x] All LabChronicle files copied to leeq/chronicle/
- [x] All imports updated from labchronicle to leeq.chronicle
- [x] All existing tests pass
- [x] No external labchronicle dependency
- [x] Documentation updated
- [x] Linting passes
- [x] Integration tests verify logging works

## Implementation Checklist

### Preparation
- [ ] Backup current working state
- [ ] Create new branch: `feature/merge-labchronicle`

### Execution
- [ ] Copy LabChronicle/labchronicle/ → leeq/chronicle/
- [ ] Add markdownify to requirements.txt
- [ ] Remove labchronicle git dependency
- [ ] Run import update script on all Python files
- [ ] Update test mocks if needed
- [ ] Update documentation files

### Validation
- [ ] Run pytest tests/
- [ ] Run integration tests
- [ ] Check no external imports remain
- [ ] Run linting
- [ ] Build documentation
- [ ] Test in fresh virtualenv

### Completion
- [ ] Commit changes
- [ ] Update CLAUDE.md
- [ ] Remove LabChronicle/ folder
- [ ] Create PR with detailed description

## Confidence Score: 9/10

**Rationale**: 
- Complete understanding of LabChronicle structure and usage
- All dependencies already present (except markdownify)
- Clear import patterns identified (30+ files)
- Simple vendoring approach with precedent in Python ecosystem
- Comprehensive test coverage exists to validate changes
- -1 point for potential unforeseen import edge cases during transition

The merge is straightforward as LabChronicle is a self-contained package with minimal dependencies, all of which are already in LeeQ. The main work is mechanical (copying files and updating imports) with good test coverage to verify success.