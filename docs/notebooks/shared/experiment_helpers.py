"""
Common experiment patterns and helper functions for LeeQ notebooks.

This module provides reusable experiment patterns, visualization helpers,
and analysis functions used across tutorial, example, and workflow notebooks.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from leeq.experiments.builtin.basic.calibrations import (
    NormalisedRabi, SimpleRamseyMultilevel, AmpPingpongCalibrationSingleQubitMultilevel,
    CrossAllXYDragMultiRunSingleQubitMultilevel
)
from leeq.experiments.builtin.basic.characterizations import SimpleT1, SpinEchoMultiLevel


def run_basic_calibration_sequence(dut, update_params=True):
    """
    Run standard single-qubit calibration sequence.
    
    Args:
        dut: TransmonElement device under test
        update_params (bool): Whether to update device parameters
        
    Returns:
        dict: Results from calibration experiments
    """
    results = {}
    
    # 1. Rabi amplitude calibration
    print("Running Rabi amplitude calibration...")
    rabi = NormalisedRabi(
        dut_qubit=dut, 
        step=0.01, 
        stop=0.5, 
        amp=0.2, 
        update=update_params
    )
    results['calibrations.NormalisedRabi'] = rabi
    
    # 2. Ramsey frequency calibration (multiple scales)
    print("Running Ramsey frequency calibration...")
    ramsey_coarse = SimpleRamseyMultilevel(
        dut=dut, 
        set_offset=10, 
        stop=0.3, 
        step=0.005
    )
    ramsey_medium = SimpleRamseyMultilevel(
        dut=dut, 
        set_offset=1, 
        stop=3, 
        step=0.05
    )
    ramsey_fine = SimpleRamseyMultilevel(
        dut=dut, 
        set_offset=0.1, 
        stop=30, 
        step=0.5
    )
    results['calibrations.SimpleRamseyMultilevel'] = {
        'coarse': ramsey_coarse,
        'medium': ramsey_medium, 
        'fine': ramsey_fine
    }
    
    # 3. Phase calibration (pingpong)
    print("Running phase calibration...")
    pingpong = AmpPingpongCalibrationSingleQubitMultilevel(dut=dut)
    results['pingpong'] = pingpong
    
    # 4. DRAG calibration
    print("Running DRAG calibration...")
    drag = CrossAllXYDragMultiRunSingleQubitMultilevel(dut=dut)
    results['calibrations.DragCalibrationSingleQubitMultilevel'] = drag
    
    return results


def run_coherence_characterization(dut):
    """
    Run standard coherence measurement sequence (T1, T2_echo, T2_ramsey).
    
    Args:
        dut: TransmonElement device under test
        
    Returns:
        dict: Results from coherence experiments
    """
    results = {}
    
    # T1 measurement
    print("Running T1 measurement...")
    t1 = SimpleT1(
        qubit=dut, 
        time_length=300, 
        time_resolution=5
    )
    results['characterizations.SimpleT1'] = t1
    
    # T2 Echo measurement  
    print("Running T2 Echo measurement...")
    t2_echo = SpinEchoMultiLevel(
        dut=dut, 
        free_evolution_time=200, 
        time_resolution=5
    )
    results['t2_echo'] = t2_echo
    
    # T2 Ramsey measurement
    print("Running T2 Ramsey measurement...")
    t2_ramsey = SimpleRamseyMultilevel(
        dut=dut, 
        stop=50, 
        step=0.25, 
        set_offset=0.2
    )
    results['t2_ramsey'] = t2_ramsey
    
    return results


def plot_rabi_oscillation(rabi_experiment, title="Rabi Oscillation"):
    """
    Create standardized Rabi oscillation plot.
    
    Args:
        rabi_experiment: NormalisedRabi experiment result
        title (str): Plot title
        
    Returns:
        plotly.graph_objects.Figure: Interactive plot
    """
    try:
        # Extract data from experiment
        data = rabi_experiment.get_results()
        x_data = data.get('amplitudes', [])
        y_data = data.get('populations', [])
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x_data,
            y=y_data,
            mode='markers+lines',
            name='Population',
            marker=dict(size=6),
            line=dict(width=2)
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Drive Amplitude',
            yaxis_title='Excited State Population',
            template='plotly_white',
            showlegend=True
        )
        
        return fig
    except Exception as e:
        print(f"Warning: Could not create Rabi plot - {e}")
        return None


def plot_coherence_measurements(coherence_results):
    """
    Create subplot figure with T1 and T2 measurements.
    
    Args:
        coherence_results (dict): Results from run_coherence_characterization
        
    Returns:
        plotly.graph_objects.Figure: Multi-panel plot
    """
    try:
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=("T1 Decay", "T2 Echo", "T2 Ramsey"),
            specs=[[{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # T1 plot
        if 'characterizations.SimpleT1' in coherence_results:
            t1_data = coherence_results['characterizations.SimpleT1'].get_results()
            fig.add_trace(
                go.Scatter(
                    x=t1_data.get('times', []),
                    y=t1_data.get('populations', []),
                    mode='markers+lines',
                    name='T1 Data',
                    marker=dict(size=4)
                ),
                row=1, col=1
            )
        
        # T2 Echo plot
        if 't2_echo' in coherence_results:
            t2_echo_data = coherence_results['t2_echo'].get_results()
            fig.add_trace(
                go.Scatter(
                    x=t2_echo_data.get('times', []),
                    y=t2_echo_data.get('populations', []),
                    mode='markers+lines',
                    name='T2 Echo Data',
                    marker=dict(size=4)
                ),
                row=1, col=2
            )
        
        # T2 Ramsey plot
        if 't2_ramsey' in coherence_results:
            t2_ramsey_data = coherence_results['t2_ramsey'].get_results()
            fig.add_trace(
                go.Scatter(
                    x=t2_ramsey_data.get('times', []),
                    y=t2_ramsey_data.get('populations', []),
                    mode='markers+lines',
                    name='T2 Ramsey Data',
                    marker=dict(size=4)
                ),
                row=1, col=3
            )
        
        fig.update_layout(
            title="Coherence Characterization",
            template='plotly_white',
            showlegend=False,
            height=400
        )
        
        # Update axis labels
        fig.update_xaxes(title_text="Time (μs)", row=1, col=1)
        fig.update_xaxes(title_text="Time (μs)", row=1, col=2)
        fig.update_xaxes(title_text="Time (μs)", row=1, col=3)
        fig.update_yaxes(title_text="Population", row=1, col=1)
        
        return fig
    except Exception as e:
        print(f"Warning: Could not create coherence plots - {e}")
        return None


def analyze_calibration_results(calibration_results):
    """
    Analyze and summarize calibration experiment results.
    
    Args:
        calibration_results (dict): Results from run_basic_calibration_sequence
        
    Returns:
        dict: Summary of calibrated parameters
    """
    summary = {}
    
    try:
        # Extract Rabi results
        if 'calibrations.NormalisedRabi' in calibration_results:
            rabi = calibration_results['calibrations.NormalisedRabi']
            summary['pi_amplitude'] = getattr(rabi, 'calibrated_amplitude', 'N/A')
            summary['rabi_frequency'] = getattr(rabi, 'rabi_frequency', 'N/A')
        
        # Extract Ramsey results  
        if 'calibrations.SimpleRamseyMultilevel' in calibration_results:
            ramsey_fine = calibration_results['calibrations.SimpleRamseyMultilevel']['fine']
            summary['frequency_offset'] = getattr(ramsey_fine, 'frequency_offset', 'N/A')
            summary['t2_ramsey'] = getattr(ramsey_fine, 't2_ramsey', 'N/A')
        
        # Extract phase calibration
        if 'pingpong' in calibration_results:
            pingpong = calibration_results['pingpong']
            summary['phase_offset'] = getattr(pingpong, 'phase_offset', 'N/A')
        
        # Extract DRAG parameter
        if 'calibrations.DragCalibrationSingleQubitMultilevel' in calibration_results:
            drag = calibration_results['calibrations.DragCalibrationSingleQubitMultilevel']
            summary['drag_coefficient'] = getattr(drag, 'drag_coefficient', 'N/A')
            
    except Exception as e:
        print(f"Warning: Could not analyze some calibration results - {e}")
    
    return summary


def create_experiment_summary_table(summary_data):
    """
    Create a formatted table summarizing experiment parameters.
    
    Args:
        summary_data (dict): Parameter summary from analyze_calibration_results
        
    Returns:
        str: HTML formatted table
    """
    html = """
    <table style='border-collapse: collapse; width: 100%; border: 1px solid #ddd;'>
    <thead>
        <tr style='background-color: #f2f2f2;'>
            <th style='border: 1px solid #ddd; padding: 8px; text-align: left;'>Parameter</th>
            <th style='border: 1px solid #ddd; padding: 8px; text-align: left;'>Value</th>
        </tr>
    </thead>
    <tbody>
    """
    
    parameter_names = {
        'pi_amplitude': 'π Pulse Amplitude',
        'rabi_frequency': 'Rabi Frequency (MHz)',
        'frequency_offset': 'Frequency Offset (MHz)',
        't2_ramsey': 'T₂ Ramsey (μs)',
        'phase_offset': 'Phase Offset (rad)',
        'drag_coefficient': 'DRAG Coefficient'
    }
    
    for key, value in summary_data.items():
        param_name = parameter_names.get(key, key)
        html += f"""
        <tr>
            <td style='border: 1px solid #ddd; padding: 8px;'>{param_name}</td>
            <td style='border: 1px solid #ddd; padding: 8px;'>{value}</td>
        </tr>
        """
    
    html += """
    </tbody>
    </table>
    """
    
    return html


def safe_experiment_execution(experiment_func, *args, **kwargs):
    """
    Safely execute experiment with error handling and logging.
    
    Args:
        experiment_func: Function that creates and runs experiment
        *args: Arguments for experiment function
        **kwargs: Keyword arguments for experiment function
        
    Returns:
        Experiment result or None if failed
    """
    try:
        result = experiment_func(*args, **kwargs)
        print(f"✓ Successfully completed {experiment_func.__name__}")
        return result
    except Exception as e:
        print(f"✗ Failed to execute {experiment_func.__name__}: {e}")
        return None


def validate_dut_configuration(dut):
    """
    Validate that DUT is properly configured for experiments.
    
    Args:
        dut: TransmonElement device under test
        
    Returns:
        bool: True if configuration is valid
    """
    try:
        # Check basic configuration
        config = dut.parameters
        if not config:
            print("Error: DUT has no configuration parameters")
            return False
        
        # Check for required collections
        if 'lpb_collections' not in config:
            print("Error: Missing lpb_collections in configuration")
            return False
        
        if 'f01' not in config['lpb_collections']:
            print("Error: Missing f01 collection in configuration")
            return False
        
        # Check measurement primitives
        if 'measurement_primitives' not in config:
            print("Error: Missing measurement_primitives in configuration")
            return False
        
        print("✓ DUT configuration validation passed")
        return True
        
    except Exception as e:
        print(f"Error validating DUT configuration: {e}")
        return False