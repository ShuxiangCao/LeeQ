# Release Notes v0.2.0

**Release Date**: TBD  
**Type**: Major Release (Breaking Changes)

## Overview

This release removes all experiment aliases from the LeeQ EPII system, requiring all code to use canonical experiment names. This change simplifies the codebase by eliminating dual naming conventions and improves code clarity.

## ⚠️ Breaking Changes

### Experiment Aliases Removed

**Impact**: All existing code using experiment aliases must be updated.

All experiment aliases have been completely removed from the `ExperimentRouter`. Code must now use full canonical names:

| Removed Alias | Required Canonical Name |
|---------------|------------------------|
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

### API Changes

**Removed Methods**:
- `ExperimentRouter._add_backward_compatibility_aliases()` 
- `ExperimentRouter._initialize_parameter_map()`

**Removed Attributes**:
- `ExperimentRouter.parameter_map`

**Changed Behavior**:
- `ExperimentRouter.experiment_map` now contains only canonical names (~78 experiments vs previous 89 with aliases)
- `ExperimentRouter.get_experiment()` only accepts canonical names
- `ExperimentRouter.list_experiments()` returns only canonical names

### Command Line Changes

**EPII Daemon**:
- `--test-experiment` parameter help text updated to reference canonical names
- Default test experiment remains `calibrations.NormalisedRabi`

## Migration

### Automatic Migration

Use the provided migration script:

```bash
# Preview changes
python scripts/migrate_aliases.py

# Apply migration
python scripts/migrate_aliases.py --apply
```

### Manual Updates Required

Update your code from:
```python
# OLD - Will no longer work
client.run_experiment("rabi", parameters)
router.get_experiment("t1")
```

To:
```python
# NEW - Required format
client.run_experiment("calibrations.NormalisedRabi", parameters) 
router.get_experiment("characterizations.SimpleT1")
```

See `MIGRATION_GUIDE.md` for complete migration instructions.

## What's Changed

### Core Changes
- **Simplified ExperimentRouter**: Removed alias support completely
- **Cleaner API**: Single naming convention throughout the system
- **Reduced complexity**: Eliminated dual name resolution logic

### Updated Files
- **Core Implementation**: 4 files updated in `leeq/epii/`
- **Test Suite**: 18 test files migrated to canonical names
- **Examples**: 3 example files updated
- **Documentation**: Updated help text and examples

### Performance Impact
- **Positive**: Slight improvement in experiment lookup performance
- **Memory**: Reduced memory usage by removing alias mappings
- **No Impact**: Experiment execution performance unchanged

## Validation

This release has been validated with:
- ✅ Full test suite passing (pytest tests/ -v)
- ✅ Style checks passing (ruff check leeq/epii/ tests/epii/)
- ✅ No remaining alias references in codebase
- ✅ All examples working with canonical names
- ✅ Integration tests passing

## Developer Impact

### For EPII Users
- **Action Required**: Update all experiment names to canonical format
- **Migration Time**: ~15 minutes with provided script
- **Testing**: Verify your code works with canonical names

### For Contributors  
- **Code Reviews**: Ensure new code uses only canonical names
- **Documentation**: Update any references to old aliases
- **Tests**: Write tests using canonical names only

## Technical Details

### Experiment Discovery
- Auto-discovery mechanism unchanged
- Still finds ~78 canonical experiments
- Naming follows `category.ExperimentClassName` pattern

### Backward Compatibility
- **None**: This is a breaking change with no backward compatibility
- **Reason**: Simplifies codebase and eliminates confusion
- **Alternative**: Use canonical names throughout

## Future Considerations

### Next Release (v0.2.1)
- Performance optimizations for experiment discovery
- Enhanced error messages for invalid experiment names
- Additional validation tooling

### Planned Features
- Experiment parameter validation improvements
- Enhanced documentation generation
- Better IDE support for experiment names

## Troubleshooting

### Common Migration Issues

1. **Tests failing after migration**:
   ```bash
   # Check for remaining aliases
   grep -r '"rabi"' tests/ examples/ --include="*.py"
   ```

2. **Import errors**:
   ```python
   # Verify router works
   from leeq.epii.experiments import ExperimentRouter
   router = ExperimentRouter()
   print(f"Available experiments: {len(router.experiment_map)}")
   ```

3. **Client code not working**:
   - Update all `run_experiment()` calls to use canonical names
   - Check configuration files for old aliases

### Getting Support

- Review `MIGRATION_GUIDE.md` for detailed migration steps
- Run migration script in dry-run mode first
- Check test suite before and after migration

## Credits

This release was implemented following the PRP architecture with:
- Comprehensive migration tooling
- Extensive test coverage
- Clear documentation and migration path

## Compatibility

| Component | Compatibility |
|-----------|--------------|
| **LeeQ Core** | ✅ Fully compatible |
| **EPII Client** | ⚠️ Requires migration |
| **Test Suite** | ⚠️ Requires migration |
| **Examples** | ⚠️ Requires migration |
| **Documentation** | ✅ Updated |

---

**For detailed migration instructions, see `MIGRATION_GUIDE.md`**

**For technical implementation details, see the PRP documentation**