# Reset Commands

## Overview

QERIS provides two types of reset commands to handle error recovery and maintenance:

1. **Hardware Reset** (`reset_hardware`) - Resets the quantum hardware state
2. **Server Reset** (`reset_server`) - Resets the MCP server state

## Hardware Reset

### Reset Types

- **`full`** - Complete hardware reset including qubits, instruments, and calibrations
- **`qubits`** - Reset only qubit parameters to defaults
- **`instruments`** - Reset instrument connections and configurations
- **`calibrations`** - Clear all calibration data

### Implementation Example

```python
async def reset_hardware(self, reset_type: str = "full") -> dict:
    """Reset hardware to default state"""
    results = {"reset_type": reset_type, "status": "success", "details": {}}
    
    try:
        if reset_type in ["full", "qubits"]:
            # Reset all qubit parameters to their default values
            for qubit_name, qubit in self.setup.qubits.items():
                if hasattr(qubit, 'reset_to_defaults'):
                    qubit.reset_to_defaults()
                elif hasattr(qubit, 'initialize'):
                    qubit.initialize()
            results["details"]["qubits"] = f"Reset {len(self.setup.qubits)} qubits to defaults"
            
        if reset_type in ["full", "instruments"]:
            # Reset instruments
            if hasattr(self.setup, 'disconnect_devices'):
                self.setup.disconnect_devices()
            if hasattr(self.setup, 'connect_devices'):
                self.setup.connect_devices()
            results["details"]["instruments"] = "Disconnected and reconnected all instruments"
            
        if reset_type in ["full", "calibrations"]:
            # Clear calibration data from qubits
            for qubit_name, qubit in self.setup.qubits.items():
                if hasattr(qubit, 'clear_calibrations'):
                    qubit.clear_calibrations()
                # Clear common calibration parameters
                for param in ['pi_amp', 'pi2_amp', 'pi_len', 'readout_amp']:
                    if hasattr(qubit, f'set_{param}'):
                        try:
                            getattr(qubit, f'set_{param}')(None)
                        except:
                            pass
            results["details"]["calibrations"] = "Cleared all calibration data"
            
        # Reset setup status parameters
        if hasattr(self.setup, '_status'):
            self.setup._status.set_parameters(
                Shot_Number=2000,
                Shot_Period=500.0,
                Measurement_Basis='<z>',
                Acquisition_Type='IQ'
            )
            results["details"]["setup_status"] = "Reset to default parameters"
            
    except Exception as e:
        results["status"] = "partial"
        results["error"] = str(e)
        
    return results
```

## Server Reset

### Parameters

- **`keep_hardware_state`** (boolean) - Whether to preserve the current hardware state

### Implementation Example

```python
async def reset_server(self, keep_hardware_state: bool = True) -> dict:
    """Reset MCP server state"""
    results = {"status": "success", "details": {}}
    
    try:
        # Clear experiment tracking
        self.current_experiment_id = None
        self.experiment_manager._active = None
        results["details"]["experiments"] = "Cleared active experiment"
        
        # Stop live monitor if running
        if hasattr(self.experiment_manager, 'stop_live_monitor'):
            self.experiment_manager.stop_live_monitor()
            results["details"]["monitor"] = "Stopped live monitor"
        
        # Re-discover experiments
        self._discover_experiments()
        results["details"]["discovery"] = f"Re-discovered {len(self.experiments)} experiments"
        
        # Clear any caches
        if hasattr(self, '_parameter_cache'):
            self._parameter_cache = {}
        if hasattr(self, '_result_cache'):
            self._result_cache = {}
        results["details"]["cache"] = "Cleared all caches"
        
        if not keep_hardware_state:
            # Also reset hardware
            hw_reset = await self.reset_hardware("full")
            results["details"]["hardware"] = hw_reset
        else:
            results["details"]["hardware"] = "Preserved"
            
        # Restart live monitor
        if hasattr(self.experiment_manager, 'start_live_monitor'):
            self.experiment_manager.start_live_monitor()
            results["details"]["monitor"] = "Restarted live monitor"
            
    except Exception as e:
        results["status"] = "failed"
        results["error"] = str(e)
        
    return results
```

## Usage Examples

### Error Recovery

```python
async def handle_hardware_issues():
    """Example of using reset commands to handle hardware issues"""
    client = Client()
    await client.connect('http://localhost:8765')
    
    # Check device status
    device_info = await client.call_tool('get_device_info')
    print(f"Device: {device_info['device_name']}")
    
    # Try to run an experiment
    try:
        exp_id = await client.call_tool('run_experiment', {
            'experiment_name': 'RabiExperiment',
            'qubits': ['q0'],
            'parameters': {'start': 0, 'stop': 1, 'step': 0.05}
        })
    except Exception as e:
        print(f"Experiment failed: {e}")
        
        # Reset hardware to recover
        print("Attempting hardware reset...")
        reset_result = await client.call_tool('reset_hardware', {
            'reset_type': 'full'
        })
        print(f"Reset result: {reset_result}")
        
        # Retry experiment
        exp_id = await client.call_tool('run_experiment', {
            'experiment_name': 'RabiExperiment',
            'qubits': ['q0'],
            'parameters': {'start': 0, 'stop': 1, 'step': 0.05}
        })
        print(f"Experiment started successfully: {exp_id}")
    
    return exp_id
```

### Maintenance Routine

```python
async def maintenance_routine():
    """Regular maintenance using reset commands"""
    client = Client()
    await client.connect('http://localhost:8765')
    
    # Clear old calibrations
    print("Clearing calibration data...")
    cal_reset = await client.call_tool('reset_hardware', {
        'reset_type': 'calibrations'
    })
    print(f"Calibrations cleared: {cal_reset['details']}")
    
    # Reset server state while keeping hardware
    print("Resetting server state...")
    server_reset = await client.call_tool('reset_server', {
        'keep_hardware_state': True
    })
    print(f"Server reset: {server_reset['details']}")
    
    # Run calibration sequence
    calibration_experiments = [
        'QubitSpectroscopy',
        'RabiExperiment', 
        'T1Experiment',
        'T2Experiment'
    ]
    
    for exp_name in calibration_experiments:
        print(f"Running {exp_name}...")
        exp_info = await client.call_tool('get_experiment_info', {
            'experiment_name': exp_name
        })
        
        exp_id = await client.call_tool('run_experiment', {
            'experiment_name': exp_name,
            'qubits': ['q0'],
            'parameters': exp_info['example']['parameters']
        })
        
        # Wait for completion
        while True:
            status = await client.call_tool('get_status')
            if status['experiment']['state'] == 'completed':
                break
            await asyncio.sleep(1)
    
    print("Calibration routine completed")
```

### Progressive Troubleshooting

```python
async def troubleshooting_workflow():
    """Advanced troubleshooting with different reset types"""
    client = Client()
    await client.connect('http://localhost:8765')
    
    # Try different reset levels
    reset_sequence = [
        ('calibrations', "Clearing calibration data only"),
        ('qubits', "Resetting qubit parameters to defaults"),
        ('instruments', "Resetting instrument connections"),
        ('full', "Full hardware reset")
    ]
    
    for reset_type, description in reset_sequence:
        print(f"\n{description}...")
        result = await client.call_tool('reset_hardware', {
            'reset_type': reset_type
        })
        
        if result['status'] == 'success':
            print(f"✓ {reset_type} reset successful")
            print(f"  Details: {result['details']}")
            
            # Test if issue is resolved
            try:
                test_status = await client.call_tool('get_status')
                if test_status['experiment']['state'] != 'error':
                    print("Issue resolved!")
                    break
            except:
                continue
        else:
            print(f"✗ {reset_type} reset failed: {result.get('error')}")
    
    # If hardware resets didn't help, try server reset
    if result['status'] != 'success':
        print("\nAttempting server reset without hardware changes...")
        server_result = await client.call_tool('reset_server', {
            'keep_hardware_state': True
        })
        print(f"Server reset: {server_result}")
```

### Clean Session

```python
async def clean_start_session():
    """Start a fresh session with clean state"""
    client = Client()
    await client.connect('http://localhost:8765')
    
    # Full reset to ensure clean state
    print("Initializing clean session...")
    
    # Reset server and hardware
    reset_result = await client.call_tool('reset_server', {
        'keep_hardware_state': False
    })
    
    print("System reset complete:")
    for component, status in reset_result['details'].items():
        print(f"  - {component}: {status}")
    
    # Verify clean state
    device = await client.call_tool('get_device_info')
    params = await client.call_tool('get_qubit_parameters', {
        'qubits': ['q0'],
        'parameter_categories': ['hamiltonian', 'coherence']
    })
    
    print(f"\nDevice ready: {device['device_name']}")
    print("Default parameters loaded:")
    for param_name, param_data in params['q0']['parameters'].items():
        print(f"  - {param_name}: {param_data['value']} {param_data.get('unit', '')}")
    
    return True
```

## Best Practices

1. **Always confirm destructive operations** - Especially for full hardware resets
2. **Use progressive reset levels** - Start with minimal reset and escalate as needed
3. **Preserve hardware state when possible** - Server resets can often fix issues without touching hardware
4. **Document reset reasons** - Log why resets were performed for troubleshooting
5. **Test after reset** - Always verify the system is functional after a reset
6. **Backup calibrations** - Save important calibration data before clearing