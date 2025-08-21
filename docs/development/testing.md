# Testing Guide

This guide covers testing practices and procedures for LeeQ development.

## Test Structure

LeeQ uses pytest for testing, with tests organized as follows:

```
tests/
├── unit/                  # Unit tests for individual components
│   ├── core/             # Core module tests
│   ├── experiments/      # Experiment tests
│   └── utils/            # Utility tests
├── integration/          # Integration tests
└── fixtures/            # Shared test fixtures
```

## Running Tests

### Basic Test Execution

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/unit/core/test_elements.py

# Run specific test
pytest tests/unit/core/test_elements.py::test_qubit_initialization
```

### Test Coverage

```bash
# Run with coverage report
pytest --cov=leeq

# Generate HTML coverage report
pytest --cov=leeq --cov-report=html

# View coverage report
open htmlcov/index.html
```

### Test Categories

```bash
# Run only unit tests
pytest tests/unit/

# Run only integration tests
pytest tests/integration/

# Run tests matching pattern
pytest -k "rabi"
```

## Writing Tests

### Unit Test Example

```python
import pytest
from leeq.core.elements import Qubit

def test_qubit_frequency():
    """Test qubit frequency setting."""
    qubit = Qubit(name="q0")
    qubit.set_frequency(5.0e9)
    assert qubit.frequency == 5.0e9

def test_qubit_invalid_frequency():
    """Test invalid frequency raises error."""
    qubit = Qubit(name="q0")
    with pytest.raises(ValueError):
        qubit.set_frequency(-1.0)
```

### Integration Test Example

```python
import pytest
from leeq.experiments import RabiExperiment
from leeq.setups import create_virtual_setup

@pytest.fixture
def virtual_setup():
    """Create virtual setup for testing."""
    return create_virtual_setup(num_qubits=2)

def test_rabi_experiment_workflow(virtual_setup):
    """Test complete Rabi experiment workflow."""
    qubit = virtual_setup.get_qubit(0)
    
    experiment = RabiExperiment(
        qubit=qubit,
        amplitude_range=(0, 1),
        num_points=20
    )
    
    result = experiment.run()
    assert result.success
    assert result.optimal_amplitude is not None
```

## Test Fixtures

### Common Fixtures

Located in `tests/fixtures/`:

```python
# conftest.py
import pytest
from leeq.core.elements import Qubit

@pytest.fixture
def sample_qubit():
    """Provide a sample qubit for testing."""
    return Qubit(
        name="test_qubit",
        frequency=5.0e9,
        anharmonicity=-200e6
    )

@pytest.fixture
def mock_hardware():
    """Mock hardware interface."""
    from unittest.mock import Mock
    hardware = Mock()
    hardware.execute.return_value = {"success": True}
    return hardware
```

## Mocking and Patching

### Mocking Hardware

```python
from unittest.mock import patch, Mock

@patch('leeq.hardware.quantum_device')
def test_with_mock_hardware(mock_device):
    """Test with mocked hardware."""
    mock_device.execute.return_value = {
        "counts": {"0": 500, "1": 500}
    }
    
    # Your test code here
    result = run_experiment()
    assert mock_device.execute.called
```

## Test Best Practices

### 1. Test Organization
- One test file per module
- Group related tests in classes
- Use descriptive test names

### 2. Test Independence
- Tests should not depend on each other
- Clean up resources after tests
- Use fixtures for shared setup

### 3. Assertions
- Use specific assertions
- Test both success and failure cases
- Include edge cases

### 4. Documentation
- Add docstrings to test functions
- Explain complex test scenarios
- Document expected behaviors

## Continuous Integration

Tests run automatically on:
- Pull requests
- Commits to main branch
- Nightly builds

### CI Configuration

```yaml
# .github/workflows/tests.yml
name: Tests
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -r requirements-dev.txt
      - name: Run tests
        run: pytest --cov=leeq
```

## Performance Testing

### Benchmarking

```python
import pytest
import time

@pytest.mark.benchmark
def test_performance():
    """Benchmark critical operations."""
    start = time.time()
    
    # Operation to benchmark
    result = expensive_operation()
    
    duration = time.time() - start
    assert duration < 1.0  # Should complete in under 1 second
```

## Debugging Tests

### Running with debugger

```bash
# Run with pdb on failure
pytest --pdb

# Run with verbose traceback
pytest --tb=long

# Run with print statements visible
pytest -s
```

### Using pytest markers

```python
@pytest.mark.slow
def test_slow_operation():
    """Mark slow tests."""
    pass

# Run excluding slow tests
pytest -m "not slow"
```

## Test Data

Store test data in `tests/data/`:
- Sample configuration files
- Expected output files
- Mock response data

```python
import os
import json

def test_with_data():
    """Test using external data file."""
    data_path = os.path.join(
        os.path.dirname(__file__),
        "data",
        "sample_results.json"
    )
    
    with open(data_path) as f:
        expected = json.load(f)
    
    result = process_data()
    assert result == expected
```