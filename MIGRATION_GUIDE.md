# Migration Guide: Experiment Aliases Removed

## Overview

This guide covers the breaking changes in LeeQ EPII v0.2.0 where all experiment aliases have been removed. All code must now use canonical experiment names.

## Breaking Changes

### Removed Experiment Aliases

All experiment aliases have been completely removed. The following table shows the mapping from old aliases to new canonical names:

| Old Alias | New Canonical Name |
|-----------|-------------------|
| `rabi` | `calibrations.NormalisedRabi` |
| `t1` | `characterizations.SimpleT1` |
| `ramsey` | `calibrations.SimpleRamseyMultilevel` |
| `echo` | `characterizations.SpinEchoMultiLevel` |
| `spin_echo` | `characterizations.SpinEchoMultiLevel` |
| `drag` | `calibrations.DragCalibrationSingleQubitMultilevel` |
| `randomized_benchmarking` | `characterizations.RandomizedBenchmarkingTwoLevelSubspaceMultilevelSystem` |
| `multi_qubit_rabi` | `calibrations.MultiQubitRabi` |
| `multi_qubit_t1` | `characterizations.MultiQubitT1` |
| `multi_qubit_ramsey` | `calibrations.MultiQubitRamseyMultilevel` |
| `qubit_spectroscopy_frequency` | `calibrations.QubitSpectroscopyFrequency` |

### Removed Features

- `ExperimentRouter._add_backward_compatibility_aliases()` method
- `ExperimentRouter._initialize_parameter_map()` method  
- `ExperimentRouter.parameter_map` attribute
- All alias-related functionality

### Experiment Count Change

The total number of available experiments has changed from 89 to approximately 78, as the 11 aliases are no longer counted as separate experiments.

## Migration Instructions

### Automatic Migration

Use the provided migration script to automatically update your code:

```bash
# Dry run (shows what would change)
python scripts/migrate_aliases.py

# Apply changes
python scripts/migrate_aliases.py --apply
```

### Manual Migration Examples

#### Before (Using Aliases)
```python
# Client usage
result = client.run_experiment("rabi", {
    "qubit": "q1",
    "amplitudes": [0.1, 0.2, 0.3],
    "num_shots": 1000
})

# Test code
router = ExperimentRouter()
exp_class = router.get_experiment("t1")

# Configuration
experiments_to_run = ["rabi", "t1", "ramsey"]
```

#### After (Using Canonical Names)
```python
# Client usage
result = client.run_experiment("calibrations.NormalisedRabi", {
    "qubit": "q1", 
    "amplitudes": [0.1, 0.2, 0.3],
    "num_shots": 1000
})

# Test code
router = ExperimentRouter()
exp_class = router.get_experiment("calibrations.NormalisedRabi")

# Configuration
experiments_to_run = [
    "calibrations.NormalisedRabi",
    "characterizations.SimpleT1", 
    "calibrations.SimpleRamseyMultilevel"
]
```

### Common Migration Patterns

1. **String literals in code**: Replace `"rabi"` with `"calibrations.NormalisedRabi"`
2. **Configuration files**: Update experiment names in JSON/YAML configs
3. **Test fixtures**: Update expected experiment lists
4. **Example scripts**: Update all demonstration code

### Files That Need Updates

The migration affects these types of files:
- Python source code using EPII experiments
- Test files
- Configuration files  
- Example scripts
- Documentation

## Validation After Migration

### Check Your Migration

1. **Verify no aliases remain**:
```python
from leeq.epii.experiments import ExperimentRouter
router = ExperimentRouter()
aliases = ['rabi', 't1', 'ramsey', 'echo', 'drag']
for alias in aliases:
    assert alias not in router.experiment_map, f"Alias {alias} still exists!"
print("✅ All aliases removed successfully")
```

2. **Verify canonical names work**:
```python
assert "calibrations.NormalisedRabi" in router.experiment_map
assert router.get_experiment("calibrations.NormalisedRabi") is not None
print("✅ Canonical names working")
```

3. **Check experiment count**:
```python
count = len(router.experiment_map)
assert 78 <= count <= 80, f"Expected ~78 experiments, got {count}"
print(f"✅ {count} canonical experiments available")
```

### Run Tests

```bash
# Run EPII tests
pytest tests/epii/ -v

# Run integration tests  
pytest tests/integration_tests/test_epii_daemon.py -v

# Check for import errors
python -c "from leeq.epii.experiments import ExperimentRouter"
```

## Troubleshooting

### Common Issues

1. **Import errors after migration**:
   - Ensure all string references to experiments use canonical names
   - Check configuration files for old aliases

2. **Tests failing**:
   - Update test assertions for new experiment counts
   - Update expected experiment name lists
   - Remove tests that specifically checked for aliases

3. **Client code not working**:
   - Update all `run_experiment()` calls to use canonical names
   - Update any hardcoded experiment name lists

### Getting Help

If you encounter issues during migration:

1. Run the migration script in dry-run mode first: `python scripts/migrate_aliases.py`
2. Check the test suite passes before making changes: `pytest tests/epii/ -v`
3. Use git to track changes and revert if needed: `git checkout -- filename.py`

## Additional Resources

- See `RELEASE_NOTES_v0.2.0.md` for complete change details
- Check the `scripts/migrate_aliases.py` script for migration logic
- Review updated examples in `examples/epii/` directory

## Summary

This migration removes complexity by eliminating the dual naming system. While it requires code changes, it results in a cleaner, more predictable API where experiment names are consistent and discoverable.

The canonical naming convention follows the pattern `category.ExperimentClassName` which makes it easier to understand the experiment's purpose and find it in the codebase.