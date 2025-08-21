# Tutorial Validation Framework

## Overview

The Tutorial Validation Framework (`validate_tutorial_steps.py`) provides comprehensive validation for LeeQ tutorial content to ensure accuracy, completeness, and proper learning progression.

## Features

### 1. Step-by-Step Verification
- Validates tutorial structure and organization
- Checks for required sections and logical progression
- Ensures proper introduction and flow

### 2. Code Example Validation
- Syntax checking for all code blocks
- Import statement validation
- Pattern checking based on code type
- Completeness verification
- Classification of code examples (DUT creation, configuration, experiments, etc.)

### 3. Concept Explanation Validation
- Ensures key concepts are properly explained
- Checks for required explanations and terminology
- Validates concept introduction and detail level
- Verifies technical accuracy

### 4. Learning Progression Checkpoints
- Validates prerequisite knowledge ordering
- Checks skill demonstrability through examples
- Ensures concepts build upon each other
- Validates checkpoint completion

## Usage

### Basic Usage
```bash
# Validate entire tutorial
python scripts/validate_tutorial_steps.py docs/tutorial.md

# Validate with verbose output
python scripts/validate_tutorial_steps.py docs/tutorial.md --verbose
```

### Targeted Validation
```bash
# Validate only code examples
python scripts/validate_tutorial_steps.py docs/tutorial.md --section code

# Validate only concept explanations  
python scripts/validate_tutorial_steps.py docs/tutorial.md --section concepts

# Validate only learning progression
python scripts/validate_tutorial_steps.py docs/tutorial.md --section progression

# Validate only structure
python scripts/validate_tutorial_steps.py docs/tutorial.md --section structure
```

### Custom Tutorial File
```bash
# Validate different tutorial file
python scripts/validate_tutorial_steps.py path/to/custom_tutorial.md
```

## Exit Codes

- `0`: All validations passed
- `1`: Tutorial structure issues
- `2`: Code example failures  
- `3`: Concept validation failures
- `4`: Learning progression issues
- `5`: Unexpected error occurred

## Validation Categories

### Structure Validation
Checks for:
- Required sections presence
- Logical section ordering
- Proper introduction
- Section progression patterns

### Code Example Validation
Validates:
- **Syntax**: Python syntax correctness
- **Imports**: Required import statements
- **Patterns**: Proper coding patterns by type
- **Completeness**: Complete, functional examples
- **Documentation**: Docstrings and comments

### Concept Validation
Ensures:
- **Key Concepts**: DUT Object, Collections, Measurement Primitives, etc.
- **Explanations**: Required terminology and explanations
- **Detail Level**: Adequate depth of explanation
- **Technical Accuracy**: Correct technical information

### Learning Progression Validation
Verifies:
- **Prerequisites**: Proper ordering of concepts
- **Checkpoints**: Defined learning milestones
- **Skills**: Demonstrable skills through examples
- **Progression**: Logical learning flow

## Required Sections

The framework expects these core sections:
- Parameter Storage and Update
- Orchestrating Pulses
- Adjusting Runtime Parameters  
- Customizing Your Setup
- Creating a Customized Experiment
- Data Persistence

## Learning Progression Checkpoints

1. **Basic Concepts**
   - DUT Object, Collections, Measurement Primitives
   - Skills: Create TransmonElement, Define collections

2. **Pulse Orchestration**  
   - Logical Primitives, LPBs, Pulse scheduling
   - Skills: Build pulse sequences, Use operators

3. **Single Qubit Operations**
   - Single qubit gates, Measurement primitives
   - Skills: Access gates, Run measurements

4. **Runtime Control**
   - Parameter updates, Calibration persistence
   - Skills: Update parameters, Save/load configurations

5. **Custom Experiments**
   - Experiment class, Data visualization
   - Skills: Create experiments, Plot results

## Code Example Classification

The framework automatically classifies code examples:
- `dut_creation`: TransmonElement creation
- `configuration`: LPB collections and measurement primitives
- `experiment_definition`: Experiment class definitions
- `function_definition`: Function definitions
- `experiment_execution`: Running experiments
- `data_persistence`: Chronicle integration
- `general_example`: Other code examples

## Report Format

The validation generates a comprehensive report including:
- Overall pass/fail summary
- Detailed results by category
- Specific issues and recommendations
- Failed test explanations

## Integration with CI/CD

The framework can be integrated into continuous integration:

```yaml
# GitHub Actions example
- name: Validate Tutorial
  run: python scripts/validate_tutorial_steps.py docs/tutorial.md
```

The exit codes allow for proper CI/CD integration and failure handling.

## Extending the Framework

### Adding New Concepts
To validate new concepts, update the `concepts` dictionary in `validate_concept_explanations()`:

```python
concepts = {
    'New Concept': {
        'section': 'Section Name',
        'required_explanations': [
            'explanation1',
            'explanation2'
        ]
    }
}
```

### Adding New Code Patterns
Update `_check_code_patterns()` to include new validation patterns:

```python
if example_type == 'new_type':
    if 'required_pattern' not in code:
        issues.append("Missing required pattern")
```

### Adding New Checkpoints
Update the `checkpoints` list in `validate_learning_progression()`:

```python
checkpoints.append({
    'name': 'New Checkpoint',
    'prerequisite': 'Previous Checkpoint',
    'concepts': ['concept1', 'concept2'],
    'skills': ['skill1', 'skill2']
})
```

## Best Practices

1. **Run Validation Regularly**: Validate tutorial content after any changes
2. **Use Verbose Mode**: Enable verbose output for detailed debugging
3. **Target Specific Sections**: Use section-specific validation for focused fixes
4. **Review All Failures**: Address all validation failures before finalizing content
5. **Update Framework**: Keep validation rules current with LeeQ changes

## Troubleshooting

### Common Issues

**Missing Imports**
- Add required import statements to code examples
- Consider the tutorial's intended audience and import patterns

**Incomplete Code Examples**
- Remove `...` placeholders with actual code
- Add missing docstrings and comments

**Concept Explanations**
- Ensure concepts are clearly introduced and explained
- Add technical details and context

**Section Organization**
- Review section ordering for logical progression
- Add missing required sections

### Getting Help

- Use `--verbose` flag for detailed output
- Check exit codes for specific failure categories
- Review generated recommendations
- Examine specific validation issues in the report

## Framework Maintenance

The validation framework should be updated when:
- LeeQ API changes affect tutorial content
- New concepts are added to the tutorial
- Learning progression changes
- Code patterns evolve

Regular maintenance ensures the framework remains effective for tutorial quality assurance.