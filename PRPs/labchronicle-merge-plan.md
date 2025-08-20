# Implementation Plan: Merge LabChronicle Package into LeeQ

## Implementation Blueprint

### Approach (Pseudocode)
```python
# Main vendoring and migration workflow
def merge_labchronicle_workflow():
    # Step 1: Vendor the package
    source = "LabChronicle/labchronicle/"
    target = "leeq/chronicle/"
    copy_directory_structure(source, target)
    preserve_module_exports(target)
    
    # Step 2: Update dependencies
    requirements = load_requirements()
    requirements.add("markdownify")
    requirements.remove("labchronicle @ git+...")
    save_requirements(requirements)
    
    # Step 3: Transform imports across codebase
    import_map = {
        'from labchronicle': 'from leeq.chronicle',
        'import labchronicle': 'import leeq.chronicle as labchronicle'
    }
    for file in find_python_files(['leeq/', 'tests/', 'notebooks/', 'benchmark/']):
        transform_imports(file, import_map)
    
    # Step 4: Update test mocks
    for test_file in find_test_files():
        update_mock_paths(test_file, 'labchronicle', 'leeq.chronicle')
    
    # Step 5: Validate integration
    run_test_suite()
    verify_no_external_imports()
    verify_functionality()
```

### File Structure
```
leeq/
  chronicle/              # Vendored LabChronicle package
    __init__.py          # Maintains original exports
    chronicle.py         # Singleton manager class
    core.py             # LoggableObject base class
    decorators.py       # Logging decorators
    logger.py           # Logging setup
    utils.py            # Helper functions
    handlers/           # Storage backends
      __init__.py
      handlers.py       # Base handler interface
      hdf5.py          # HDF5 persistence
      memory.py        # In-memory storage
      dummy.py         # No-op handler
```

### Error Handling Strategy
- File Copy Errors: Verify source exists, check permissions
- Import Transform Errors: Create backup, use safe regex patterns
- Test Failures: Rollback changes, investigate specific failures
- Missing Dependencies: Install required packages before proceeding

## Phase 1: Preparation & Vendoring
**Goal**: Vendor LabChronicle package and update dependencies
**Validation**: Package copied correctly, dependencies updated

### Parallel Tasks:
- Task 1.1: Vendor LabChronicle package
  - Clone/access LabChronicle repository
  - Copy labchronicle/ to leeq/chronicle/
  - Verify all files copied correctly
  - Ensure __init__.py maintains same exports

- Task 1.2: Update dependency configuration
  - Add markdownify to requirements.txt
  - Remove labchronicle git dependency
  - Update pyproject.toml if present
  - Verify dependency resolution works

- Task 1.3: Create import transformation script
  - Write script to automate import updates
  - Include backup functionality
  - Add dry-run mode for testing
  - Create import mapping configuration

### Phase Validation:
```bash
# Verify chronicle module exists and imports work
ls -la leeq/chronicle/
python -c "from leeq.chronicle import LoggableObject, log_and_record, Chronicle"

# Check dependencies updated
grep -v "labchronicle @ git" requirements.txt
grep "markdownify" requirements.txt

# Test import script in dry-run mode
python scripts/transform_imports.py --dry-run
```

## Phase 2: Core Import Migration
**Goal**: Update all imports from labchronicle to leeq.chronicle
**Validation**: All imports transformed, no external references remain

### Parallel Tasks:
- Task 2.1: Update core module imports
  - Transform imports in leeq/core/base.py
  - Update imports in leeq/experiments/
  - Update imports in leeq/setups/
  - Verify module loading works

- Task 2.2: Update test imports
  - Transform imports in tests/
  - Update mock paths in test files
  - Fix any pytest fixtures using labchronicle
  - Ensure test discovery works

- Task 2.3: Update notebook and benchmark imports
  - Transform imports in notebooks/
  - Transform imports in benchmark/
  - Update any example scripts
  - Check interactive imports work

### Phase Validation:
```bash
# Verify no external labchronicle imports remain
grep -r "from labchronicle\|import labchronicle" leeq/ tests/ notebooks/ benchmark/ --exclude-dir=chronicle
# Should return nothing

# Check core imports work
python -c "from leeq.core.base import LeeQObject"
python -c "from leeq.experiments.experiments import Experiment"

# Quick smoke test
python -c "
from leeq.chronicle import LoggableObject
from leeq.core.base import LeeQObject
obj = LeeQObject('test')
assert isinstance(obj, LoggableObject)
print('Core inheritance works!')
"
```

## Phase 3: Testing & Integration Verification
**Goal**: Ensure all functionality works with vendored package
**Validation**: All tests pass, logging functionality verified

### Parallel Tasks:
- Task 3.1: Run unit tests
  - Execute chronicle-specific tests
  - Run core module tests
  - Verify decorator functionality
  - Check handler operations

- Task 3.2: Run integration tests
  - Execute experiment workflow tests
  - Verify logging persistence
  - Test chronicle singleton behavior
  - Validate data serialization

- Task 3.3: Run E2E validation
  - Test complete experiment execution
  - Verify labchronicle features work
  - Test with real quantum elements

### Phase Validation:
```bash
# Run all tests with coverage
pytest tests/ -v --cov=leeq --cov-report=term-missing

# Run specific integration tests
pytest tests/integration_tests/ -v
pytest tests/epii/test_leeq_backend_integration.py -v

# Test decorator functionality
python -c "
from leeq.chronicle import log_and_record, log_event
@log_and_record
class TestClass:
    @log_event
    def test_method(self):
        return 'success'
TestClass().test_method()
print('Decorators work!')
"

# Run linting
bash ./ci_scripts/lint.sh
```

## Phase 4: Documentation & Cleanup
**Goal**: Update documentation and finalize migration
**Validation**: Documentation accurate, no legacy references

### Parallel Tasks:
- Task 4.1: Update documentation
  - Update CLAUDE.md dependencies section
  - Update installation instructions
  - Fix import examples in docs/
  - Update developer documentation

- Task 4.2: Clean up
  - Remove any temporary files
  - Add module-level documentation
  - Create migration notes

- Task 4.3: Final validation
  - Fresh virtualenv installation test
  - Build and verify documentation
  - Create rollback documentation

### Phase Validation:
```bash
# Build documentation
mkdocs build

# Test in fresh environment
python -m venv test_env
source test_env/bin/activate
pip install -e .
python -c "from leeq.chronicle import LoggableObject"
pytest tests/core/test_base.py -v

# Final import check
find . -type f -name "*.py" -exec grep -l "labchronicle" {} \; | grep -v "/chronicle/"
# Should return nothing

# Verify all validation gates pass
grep -r "from labchronicle\|import labchronicle" leeq/ tests/ --exclude-dir=chronicle
pytest tests/ -v
bash ./ci_scripts/lint.sh
```

## Execution Strategy
1. Create feature branch: `git checkout -b feature/merge-labchronicle`
2. Execute Phase 1 tasks in parallel (vendoring, dependencies, script creation)
3. Validate Phase 1 before proceeding
4. Execute Phase 2 tasks in parallel (transform all imports)
5. Validate Phase 2 - ensure no external imports remain
6. Execute Phase 3 tasks in parallel (comprehensive testing)
7. Validate Phase 3 - all tests must pass
8. Execute Phase 4 tasks in parallel (documentation and cleanup)
9. Final validation across all components
10. Create PR with detailed migration report

## Success Metrics
- ✅ All 30+ files updated with new imports
- ✅ 100% of existing tests passing
- ✅ No external labchronicle references
- ✅ Documentation builds successfully
- ✅ Linting passes without errors
- ✅ Fresh installation works correctly
- ✅ Chronicle functionality preserved

## Rollback Strategy
If any phase fails:
1. Revert all file changes: `git checkout -- .`
2. Restore requirements.txt: `git checkout -- requirements.txt`
3. Remove leeq/chronicle/ directory: `rm -rf leeq/chronicle/`
4. Re-run tests to verify original state
5. Document failure reason for retry

## Import Transformation Script
Save as `scripts/transform_imports.py`:
```python
#!/usr/bin/env python3
import os
import re
import shutil
from pathlib import Path
import argparse

def transform_imports(file_path, dry_run=False):
    """Transform labchronicle imports to leeq.chronicle"""
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    original = content
    
    # Transform patterns
    patterns = [
        (r'from labchronicle import', 'from leeq.chronicle import'),
        (r'from labchronicle\.', 'from leeq.chronicle.'),
        (r'import labchronicle\b', 'import leeq.chronicle as labchronicle'),
        (r'"labchronicle\.', '"leeq.chronicle.'),  # String references
        (r"'labchronicle\.", "'leeq.chronicle."),  # String references
    ]
    
    for pattern, replacement in patterns:
        content = re.sub(pattern, replacement, content)
    
    if content != original:
        if not dry_run:
            # Backup original
            shutil.copy2(file_path, f"{file_path}.bak")
            with open(file_path, 'w') as f:
                f.write(content)
        print(f"{'[DRY-RUN] ' if dry_run else ''}Updated: {file_path}")
        return True
    return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    
    dirs = ['leeq/', 'tests/', 'notebooks/', 'benchmark/']
    updated = 0
    
    for dir_path in dirs:
        if os.path.exists(dir_path):
            for py_file in Path(dir_path).rglob('*.py'):
                if 'chronicle' not in str(py_file):
                    if transform_imports(py_file, args.dry_run):
                        updated += 1
    
    print(f"\n{'[DRY-RUN] ' if args.dry_run else ''}Total files updated: {updated}")

if __name__ == '__main__':
    main()
```

## Timeline Estimate
- Phase 1: 1-2 hours (vendoring and setup)
- Phase 2: 1-2 hours (import transformation)
- Phase 3: 2-3 hours (testing and validation)
- Phase 4: 1-2 hours (documentation and cleanup)

**Total: 5-9 hours** (with parallel execution)