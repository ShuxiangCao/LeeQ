# Implementation Plan: LeeQ EPII (Experiment Platform Intelligence Interface)

## Implementation Blueprint

### Approach (Pseudocode)
```python
# Main EPII gRPC Service Implementation
class EPIIService(ExperimentPlatformServiceServicer):
    def __init__(self, setup_config):
        self.setup = load_leeq_setup(setup_config)
        self.experiment_router = ExperimentRouter()
        self.parameter_manager = ParameterManager(self.setup)
    
    def RunExperiment(self, request, context):
        # 1. Validate experiment request
        experiment_name = request.experiment_name
        parameters = deserialize_parameters(request.parameters)
        
        # 2. Route to LeeQ experiment
        experiment_class = self.experiment_router.get_experiment(experiment_name)
        experiment = experiment_class(self.setup, **parameters)
        
        # 3. Execute experiment
        experiment.run()
        
        # 4. Serialize results
        response = ExperimentResponse()
        response.data.CopyFrom(serialize_numpy_array(experiment.data))
        response.fit_params.update(experiment.fit_params)
        
        return response
    
    def GetParameter(self, request, context):
        # Map EPII parameter to LeeQ setup parameter
        value = self.parameter_manager.get_parameter(request.name)
        return ParameterResponse(value=serialize_value(value))
    
    def SetParameter(self, request, context):
        # Validate and set LeeQ setup parameter
        self.parameter_manager.set_parameter(request.name, request.value)
        return StatusResponse(success=True)

# Daemon Entry Point
def main():
    config = load_config_from_file()
    setup = initialize_leeq_setup(config)
    service = EPIIService(setup)
    
    server = grpc.server(ThreadPoolExecutor(max_workers=10))
    add_ExperimentPlatformServiceServicer_to_server(service, server)
    server.add_insecure_port(f'[::]:{config.port}')
    
    server.start()
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        server.stop(grace=30)
```

### File Structure
```
leeq/
  epii/
    __init__.py           # Package initialization
    proto/
      epii.proto         # Protocol buffer definitions
      epii_pb2.py        # Generated protobuf messages
      epii_pb2_grpc.py   # Generated gRPC service stubs
    service.py           # Main gRPC service implementation
    daemon.py            # Daemon CLI and process management
    experiments.py       # Experiment router and mapping
    parameters.py        # Parameter management
    serialization.py     # NumPy/data serialization utilities
    config.py            # Configuration loading and validation
    utils.py             # Helper functions
tests/
  epii/
    test_service.py      # gRPC service tests
    test_experiments.py  # Experiment router tests
    test_parameters.py   # Parameter management tests
    test_serialization.py # Serialization tests
    test_daemon.py       # Daemon process tests
    fixtures/
      simulation_2q.json # Test configuration files
  integration_tests/
    test_epii_daemon.py  # End-to-end integration tests
configs/
  epii/
    simulation_2q.json   # 2-qubit simulation setup
    hardware_lab1.json   # Hardware setup example
scripts/
  systemd/
    leeq-epii@.service   # Systemd service template
```

### Error Handling Strategy
- **gRPC Errors**: Use grpc.StatusCode for standard error responses
- **Parameter Errors**: Validate types and ranges, return clear error messages  
- **Experiment Errors**: Catch LeeQ exceptions, log details, return execution errors
- **Serialization Errors**: Handle numpy/protobuf conversion failures gracefully
- **Configuration Errors**: Validate JSON configs on startup, fail fast with clear messages

## Phase 1: Foundation Setup
**Goal**: Establish gRPC infrastructure and core service skeleton
**Validation**: gRPC server starts, protobuf messages work, basic service responds

### Parallel Tasks:

#### Task 1.1: Protocol Buffer Setup
- Create `leeq/epii/proto/epii.proto` based on EPII v1.0 specification
- Generate Python stubs: `python -m grpc_tools.protoc --python_out=. --grpc_python_out=. epii.proto`
- Create basic message validation tests

#### Task 1.2: Project Structure
- Create `leeq/epii/` directory structure
- Initialize `__init__.py` files with proper imports
- Create `tests/epii/` test structure with pytest fixtures

#### Task 1.3: Base Service Implementation
- Create `leeq/epii/service.py` with ExperimentPlatformServiceServicer skeleton
- Implement Ping() and GetCapabilities() methods
- Add basic gRPC server startup/shutdown in `leeq/epii/daemon.py`

### Phase Validation:
```bash
# Check structure exists
ls -la leeq/epii/
ls -la tests/epii/

# Check protobuf generation
python -c "from leeq.epii.proto import epii_pb2, epii_pb2_grpc"

# Test basic service
pytest tests/epii/test_service.py::test_ping -v
pytest tests/epii/test_service.py::test_capabilities -v

# Start daemon (should not crash)
python -m leeq.epii.daemon --config tests/epii/fixtures/minimal.json --port 50051 &
sleep 2
kill %1
```

## Phase 2: Core Implementation
**Goal**: Implement experiment execution and parameter management
**Validation**: Core experiments work, parameters can be get/set, data serializes

### Parallel Tasks:

#### Task 2.1: Experiment Router
- Create `leeq/epii/experiments.py` with experiment name mapping
- Implement experiment discovery from LeeQ experiment classes
- Map EPII experiment names to LeeQ classes (rabi, t1, ramsey, echo, drag, randomized_benchmarking)
- Add parameter mapping and validation

#### Task 2.2: Data Serialization
- Create `leeq/epii/serialization.py` for numpy array ↔ protobuf conversion
- Implement serialization using `numpy.tobytes()` method
- Add plotly figure serialization for plot data
- Handle different numpy dtypes and array shapes

#### Task 2.3: Parameter Management  
- Create `leeq/epii/parameters.py` for LeeQ setup parameter interface
- Map LeeQ SetupStatusParameters to EPII parameter operations
- Implement type conversion and validation for different parameter types
- Add parameter listing and discovery

#### Task 2.4: Configuration System
- Create `leeq/epii/config.py` for JSON configuration loading
- Implement setup factory pattern for simulation vs hardware
- Add configuration validation and environment variable support

### Phase Validation:
```bash
# Test experiment router
pytest tests/epii/test_experiments.py -v

# Test parameter operations
pytest tests/epii/test_parameters.py -v

# Test configuration loading
pytest tests/epii/test_config.py -v
```

## Phase 3: Integration & Service Methods
**Goal**: Complete gRPC service implementation with all methods working
**Validation**: All EPII service methods functional, experiments execute end-to-end

### Parallel Tasks:

#### Task 3.1: Complete Service Implementation
- Implement RunExperiment() method with full error handling
- Implement GetParameter(), SetParameter(), ListParameters() methods  
- Add proper gRPC error handling and status codes
- Integrate all components in service layer

#### Task 3.2: LeeQ Backend Integration
- Test with existing LeeQ experiment classes in simulation mode
- Validate experiment parameter passing and result collection
- Ensure compatibility with LeeQ setup management
- Add experiment execution timeout and cancellation

#### Task 3.3: End-to-End Testing
- Create integration tests with real LeeQ simulation setups
- Test complete gRPC client-server workflows
- Validate data serialization
- Test experiment execution

### Phase Validation:
```bash
# Test complete service methods
pytest tests/epii/test_service.py -v

# Run integration tests
pytest tests/integration_tests/test_epii_daemon.py -v

# Test with simulation setup
python -m leeq.epii.daemon --config configs/epii/simulation_2q.json &
python tests/epii/manual_client_test.py
```

## Phase 4: Daemon & Production Features
**Goal**: Production-ready daemon with proper process management and monitoring
**Validation**: Daemon runs as systemd service, handles errors gracefully, logs properly

### Parallel Tasks:

#### Task 4.1: Daemon Process Management
- Complete `leeq/epii/daemon.py` with CLI interface (argparse)
- Add proper signal handling (SIGTERM, SIGINT) for graceful shutdown
- Implement PID file management and process monitoring
- Add daemon startup validation and health checks

#### Task 4.2: Systemd Integration
- Create systemd service template in `scripts/systemd/leeq-epii@.service`
- Add installation scripts for service deployment
- Configure proper logging to syslog/journald
- Test service start/stop/restart operations

#### Task 4.3: Logging & Debugging
- Implement request/response logging
- Create debugging utilities and troubleshooting tools

#### Task 4.4: Documentation & Examples
- Create configuration examples for different setups
- Add deployment guide and systemd setup instructions
- Document gRPC client usage patterns
- Create troubleshooting guide

### Phase Validation:
```bash
# Test daemon CLI
python -m leeq.epii.daemon --help
python -m leeq.epii.daemon --config configs/epii/simulation_2q.json --validate

# Test systemd service (requires sudo)
sudo systemctl enable scripts/systemd/leeq-epii@simulation_2q.service
sudo systemctl start leeq-epii@simulation_2q
sudo systemctl status leeq-epii@simulation_2q

# Full test suite
pytest tests/ -v --cov=leeq.epii --cov-report=term-missing
```

## Phase 5: Final Validation & Polish
**Goal**: Complete implementation with full test coverage and documentation
**Validation**: All quality gates pass, ready for production deployment

### Parallel Tasks:

#### Task 5.1: Comprehensive Testing
- Achieve >90% test coverage for all EPII modules
- Test with different LeeQ setup configurations

#### Task 5.2: Code Quality & Style
- Run linting and style checks: `bash ./ci_scripts/lint.sh`
- Add type hints to all public interfaces
- Ensure docstring coverage for all modules
- Code review and refactoring

#### Task 5.3: Production Readiness
- Test deployment in different environments
- Create operational documentation

### Phase Validation:
```bash
# Style and type checking
bash ./ci_scripts/lint.sh

# Full test suite with coverage
pytest tests/ -v --cov=leeq.epii --cov-report=term-missing --cov-fail-under=90

# Integration with external EPII client
python tests/integration_tests/external_client_test.py

```

## Execution Strategy

### Sequential Phase Execution
1. **Foundation → Core → Integration → Daemon → Validation**
2. Each phase must pass validation before proceeding
3. Parallel tasks within phases can run simultaneously  
4. Failed validations require re-execution of failed tasks
5. Continuous integration runs on each phase completion

### Dependency Management
- **Phase 1 → Phase 2**: Protobuf definitions needed for serialization
- **Phase 2 → Phase 3**: All core components needed for service integration
- **Phase 3 → Phase 4**: Working service needed for daemon implementation
- **Phase 4 → Phase 5**: Complete implementation needed for final validation

## Success Metrics

### Functional Criteria
- [ ] All 6 required experiments working (rabi, t1, ramsey, echo, drag, randomized_benchmarking)
- [ ] Parameter management fully functional (get/set/list operations)
- [ ] gRPC server handles requests correctly
- [ ] Daemon integrates with systemd successfully

### Quality Criteria  
- [ ] Test coverage > 90% for leeq.epii modules
- [ ] No linting or type checking errors
- [ ] Integration tests pass with simulation setups
- [ ] Documentation complete with examples

### Production Criteria
- [ ] Systemd service starts/stops reliably
- [ ] Graceful shutdown handles running experiments
- [ ] Error handling provides useful feedback
- [ ] Configuration validation prevents runtime issues
- [ ] Logging works correctly