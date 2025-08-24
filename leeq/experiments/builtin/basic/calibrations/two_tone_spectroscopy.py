from typing import Any, Optional

import numpy as np
import plotly.graph_objects as go
from k_agents.inspection.decorator import visual_inspection

from leeq import Experiment, ExperimentManager, Sweeper, SweepParametersSideEffectFactory
from leeq.chronicle import log_and_record, register_browser_function
from leeq.setups.built_in.setup_simulation_high_level import HighLevelSimulationSetup

__all__ = ['TwoToneQubitSpectroscopy']


def _simulate_two_tone_point_global(drives, readout_params, readout_channel, simulator):
    """
    Global worker function for parallel two-tone spectroscopy point calculation.
    This function must be at module level to be pickle-able for multiprocessing.
    """
    iq_response = simulator.simulate_spectroscopy_iq(drives, readout_params)
    return iq_response[readout_channel]


class TwoToneQubitSpectroscopy(Experiment):
    """
    Two-tone spectroscopy with dual frequency sweeps and optional noise-free simulation.
    
    This experiment applies two frequency-swept tones simultaneously to probe:
    - Multi-photon transitions
    - Sideband effects  
    - Qubit-resonator coupling
    - AC Stark shifts
    
    In simulation mode, the disable_noise parameter allows generation of clean data
    without Gaussian noise for validation and benchmarking purposes.
    
    Attributes
    ----------
    trace : ndarray
        A 2D numpy array holding the complex IQ data from the measurement.
    result : dict
        A dictionary with 'Magnitude' and 'Phase' of the measured response.
    freq1_arr : ndarray
        Array of frequencies for tone 1.
    freq2_arr : ndarray
        Array of frequencies for tone 2.
        
    Examples
    --------
    Standard two-tone spectroscopy with noise:
    
    >>> exp = TwoToneQubitSpectroscopy(
    ...     dut_qubit=qubit,
    ...     tone1_start=4950.0, tone1_stop=5050.0, tone1_step=10.0,
    ...     tone2_start=4750.0, tone2_stop=4850.0, tone2_step=10.0,
    ...     num_avs=1000
    ... )
    
    Clean simulation for theoretical comparison:
    
    >>> exp_clean = TwoToneQubitSpectroscopy(
    ...     dut_qubit=qubit,
    ...     tone1_start=4950.0, tone1_stop=5050.0, tone1_step=10.0,
    ...     tone2_start=4750.0, tone2_stop=4850.0, tone2_step=10.0,
    ...     num_avs=1000,
    ...     disable_noise=True
    ... )
    
    Cross-channel measurement (tone2 on f12):
    
    >>> exp_cross = TwoToneQubitSpectroscopy(
    ...     dut_qubit=qubit,
    ...     tone1_start=4950.0, tone1_stop=5050.0, tone1_step=10.0,
    ...     tone2_start=4750.0, tone2_stop=4850.0, tone2_step=10.0,
    ...     same_channel=False,
    ...     num_avs=1000,
    ...     disable_noise=True
    ... )
    """
    
    @log_and_record
    def run(
        self,
        dut_qubit: Any,
        tone1_start: float = 4950.0,
        tone1_stop: float = 5050.0,
        tone1_step: float = 10.0,
        tone1_amp: float = 0.1,
        tone2_start: float = 4750.0,
        tone2_stop: float = 4850.0,
        tone2_step: float = 10.0,
        tone2_amp: float = 0.1,
        same_channel: bool = False,
        num_avs: int = 1000,
        rep_rate: float = 0.0,
        mp_width: float = 1.0,
        set_qubit: str = 'f01',
        disable_noise: bool = False,
        use_parallel: bool = True,
        num_workers: Optional[int] = None,
    ):
        """
        Run two-tone spectroscopy experiment on hardware.
        
        Parameters
        ----------
        dut_qubit : Any
            The qubit element to perform spectroscopy on.
        tone1_start : float
            Start frequency for tone 1 in MHz.
        tone1_stop : float
            Stop frequency for tone 1 in MHz.
        tone1_step : float
            Step size for tone 1 frequency sweep in MHz.
        tone1_amp : float
            Amplitude for tone 1.
        tone2_start : float
            Start frequency for tone 2 in MHz.
        tone2_stop : float
            Stop frequency for tone 2 in MHz.
        tone2_step : float
            Step size for tone 2 frequency sweep in MHz.
        tone2_amp : float
            Amplitude for tone 2.
        same_channel : bool
            If True, both tones on same channel. If False, tone 2 on f12 channel.
        num_avs : int
            Number of averages.
        rep_rate : float
            Repetition rate in Hz.
        mp_width : float
            Measurement pulse width in microseconds.
        set_qubit : str
            Qubit transition to probe ('f01' or 'f12').
        disable_noise : bool
            If True, disable noise in simulation mode. Ignored in hardware mode.
        use_parallel : bool
            If True, use CPU parallelization for faster processing. Ignored in hardware mode.
        num_workers : Optional[int]
            Number of worker processes. If None, uses all CPU cores. Ignored in hardware mode.
        """
        self.dut_qubit = dut_qubit
        self.tone1_amp = tone1_amp
        self.tone2_amp = tone2_amp
        self.same_channel = same_channel
        self.num_avs = num_avs
        
        # Setup frequency arrays
        self.freq1_arr = np.arange(tone1_start, tone1_stop + tone1_step/2, tone1_step)
        self.freq2_arr = np.arange(tone2_start, tone2_stop + tone2_step/2, tone2_step)
        
        # Build pulse sequence
        lpb = self._build_pulse_sequence(
            dut_qubit, same_channel, set_qubit, 
            tone1_amp, tone2_amp, mp_width
        )
        
        # Setup sweepers
        swp = self._setup_sweepers(
            tone1_start, tone1_stop, tone1_step,
            tone2_start, tone2_stop, tone2_step
        )
        
        # Run experiment
        ExperimentManager().run(lpb, swp)
        
        # Process results
        self._process_results()
    
    @log_and_record(overwrite_func_name='TwoToneQubitSpectroscopy.run')
    def run_simulated(
        self,
        dut_qubit: Any,
        tone1_start: float = 4950.0,
        tone1_stop: float = 5050.0,
        tone1_step: float = 10.0,
        tone1_amp: float = 0.1,
        tone2_start: float = 4750.0,
        tone2_stop: float = 4850.0,
        tone2_step: float = 10.0,
        tone2_amp: float = 0.1,
        same_channel: bool = False,
        num_avs: int = 1000,
        rep_rate: float = 0.0,
        mp_width: float = 1.0,
        set_qubit: str = 'f01',
        disable_noise: bool = False,
        use_parallel: bool = True,
        num_workers: Optional[int] = None,
    ):
        """
        Run two-tone spectroscopy experiment in simulation mode with optional noise-free data and CPU parallelization.
        
        Uses CWSpectroscopySimulator for efficient multi-tone simulation. Supports both
        same-channel (combined amplitude) and cross-channel configurations.
        
        Parameters
        ----------
        dut_qubit : Any
            The qubit element to perform spectroscopy on.
        tone1_start : float
            Start frequency for tone 1 in MHz.
        tone1_stop : float
            Stop frequency for tone 1 in MHz.
        tone1_step : float
            Step size for tone 1 frequency sweep in MHz.
        tone1_amp : float
            Amplitude for tone 1.
        tone2_start : float
            Start frequency for tone 2 in MHz.
        tone2_stop : float
            Stop frequency for tone 2 in MHz.
        tone2_step : float
            Step size for tone 2 frequency sweep in MHz.
        tone2_amp : float
            Amplitude for tone 2.
        same_channel : bool
            If True, both tones on same channel. If False, tone 2 on f12 channel.
        num_avs : int
            Number of averages.
        rep_rate : float
            Repetition rate in Hz.
        mp_width : float
            Measurement pulse width in microseconds.
        set_qubit : str
            Qubit transition to probe ('f01' or 'f12').
        disable_noise : bool, optional
            If True, skip Gaussian noise addition for clean simulation data.
            Useful for physics validation and benchmarking. Default is False.
        use_parallel : bool, optional
            If True, use CPU parallelization for faster 2D processing.
            Default is True for automatic speedup.
        num_workers : Optional[int], optional
            Number of worker processes for parallelization. If None, uses all CPU cores.
            Only effective when use_parallel=True.
            
        Notes
        -----
        When disable_noise=True:
        - No Gaussian noise (normally scaled by 10.0/sqrt(num_avs))
        - Results are deterministic and repeatable across the entire 2D map
        - Ideal for comparing against theoretical models
        - Note: This experiment uses simpler noise model (Gaussian only, no baseline dropout)
        
        Channel Configuration:
        - same_channel=True: Both tones on same drive channel, amplitudes combine
        - same_channel=False: Tone 1 on primary channel, tone 2 on secondary (f01/f12)
        
        Examples
        --------
        Standard two-tone simulation with noise:
        
        >>> exp = TwoToneQubitSpectroscopy(
        ...     dut_qubit=qubit,
        ...     tone1_start=4950.0, tone1_stop=5050.0, tone1_step=10.0,
        ...     tone2_start=4750.0, tone2_stop=4850.0, tone2_step=10.0,
        ...     same_channel=True, num_avs=1000
        ... )
        
        Clean cross-channel analysis:
        
        >>> exp_clean = TwoToneQubitSpectroscopy(
        ...     dut_qubit=qubit,
        ...     tone1_start=4950.0, tone1_stop=5050.0, tone1_step=10.0,
        ...     tone2_start=4750.0, tone2_stop=4850.0, tone2_step=10.0,
        ...     same_channel=False, num_avs=1000,
        ...     disable_noise=True
        ... )
        """
        from leeq.theory.simulation.numpy.cw_spectroscopy import CWSpectroscopySimulator
        
        self.dut_qubit = dut_qubit
        self.tone1_amp = tone1_amp
        self.tone2_amp = tone2_amp
        self.same_channel = same_channel
        self.num_avs = num_avs
        
        # Setup frequency arrays
        self.freq1_arr = np.arange(tone1_start, tone1_stop + tone1_step/2, tone1_step)
        self.freq2_arr = np.arange(tone2_start, tone2_stop + tone2_step/2, tone2_step)
        
        # Get setup and simulator
        setup = ExperimentManager().get_default_setup()
        simulator = CWSpectroscopySimulator(setup)
        
        # Get channels
        c1 = dut_qubit.get_c1(set_qubit)
        channel1 = c1.channel
        if same_channel:
            channel2 = channel1
        else:
            if set_qubit == 'f01':
                c2 = dut_qubit.get_c1('f12')
            else:
                c2 = dut_qubit.get_c1('f01')
            channel2 = c2.channel
        
        # Get readout parameters
        mp = dut_qubit.get_default_measurement_prim_int()
        readout_channel = mp.channel
        readout_freq = mp.freq
        readout_amp = mp.amp
        
        # 2D sweep simulation with optional parallelization
        if use_parallel:
            try:
                import time
                from concurrent.futures import ProcessPoolExecutor
                import multiprocessing
                
                start_time = time.time()
                
                if num_workers is None:
                    num_workers = multiprocessing.cpu_count()
                
                # Create parameter combinations for parallel processing
                param_combinations = []
                for f1 in self.freq1_arr:
                    for f2 in self.freq2_arr:
                        # Setup drives for this parameter combination
                        if same_channel and np.isclose(f1, f2):
                            drives = [(channel1, f1, tone1_amp + tone2_amp)]
                        else:
                            drives = [(channel1, f1, tone1_amp), (channel2, f2, tone2_amp)]
                        
                        readout_params = {
                            readout_channel: {
                                'frequency': readout_freq,
                                'amplitude': readout_amp
                            }
                        }
                        param_combinations.append((drives, readout_params, readout_channel))
                
                # Process in parallel
                with ProcessPoolExecutor(max_workers=num_workers) as executor:
                    futures = [executor.submit(_simulate_two_tone_point_global, 
                                             drives, readout_params, readout_channel, simulator)
                             for drives, readout_params, readout_channel in param_combinations]
                    result_flat = [f.result() for f in futures]
                
                # Reshape to 2D array
                result = np.array(result_flat).reshape(len(self.freq1_arr), len(self.freq2_arr))
                
                parallel_time = time.time() - start_time
                print(f"Two-tone parallel processing completed in {parallel_time:.2f}s using {num_workers} workers")
                
            except Exception as e:
                print(f"Two-tone parallel processing failed ({e}), falling back to sequential")
                use_parallel = False
        
        if not use_parallel:
            # Sequential processing (original implementation)
            result = []
            for f1 in self.freq1_arr:
                row = []
                for f2 in self.freq2_arr:
                    # Setup drives
                    if same_channel and np.isclose(f1, f2):
                        # Combined amplitude when same frequency on same channel
                        drives = [(channel1, f1, tone1_amp + tone2_amp)]
                    else:
                        drives = [(channel1, f1, tone1_amp), (channel2, f2, tone2_amp)]
                    
                    # Simulate
                    readout_params = {
                        readout_channel: {
                            'frequency': readout_freq,
                            'amplitude': readout_amp
                        }
                    }
                    iq_response = simulator.simulate_spectroscopy_iq(drives, readout_params)
                    row.append(iq_response[readout_channel])
                result.append(row)
        
        # Store results
        self.trace = np.array(result)
        
        # Add realistic noise (only if noise is enabled)
        if not disable_noise:
            self.trace = self._add_realistic_noise(self.trace, num_avs)
        
        # Process results
        self._process_results()
    
    def _build_pulse_sequence(self, dut_qubit, same_channel, set_qubit, 
                              tone1_amp, tone2_amp, mp_width):
        """Build logical primitive blocks for the pulse sequence."""
        # Get drive primitives
        c1 = dut_qubit.get_c1(set_qubit)
        drive1 = c1['X'].clone()
        drive1.update_pulse_args(amp=tone1_amp, width=mp_width)
        
        if same_channel:
            drive2 = c1['X'].clone()
        else:
            # Use different transition
            if set_qubit == 'f01':
                c2 = dut_qubit.get_c1('f12')
            else:
                c2 = dut_qubit.get_c1('f01')
            drive2 = c2['X'].clone()
        drive2.update_pulse_args(amp=tone2_amp, width=mp_width)
        
        # Store for sweeper access
        self.drive1 = drive1
        self.drive2 = drive2
        
        # Get measurement primitive
        mp = dut_qubit.get_default_measurement_prim_int()
        
        # Build sequence: parallel drives followed by measurement
        lpb = (drive1 * drive2) + mp
        
        return lpb
    
    def _setup_sweepers(self, tone1_start, tone1_stop, tone1_step,
                       tone2_start, tone2_stop, tone2_step):
        """Setup 2D frequency sweepers."""
        # Frequency sweep for tone 1
        swp_freq1 = Sweeper(
            np.arange,
            n_kwargs={
                'start': tone1_start,
                'stop': tone1_stop + tone1_step/2,
                'step': tone1_step
            },
            params=[
                SweepParametersSideEffectFactory.func(
                    self.drive1.update_freq, {}, 'freq'
                )
            ]
        )
        
        # Frequency sweep for tone 2
        swp_freq2 = Sweeper(
            np.arange,
            n_kwargs={
                'start': tone2_start,
                'stop': tone2_stop + tone2_step/2,
                'step': tone2_step
            },
            params=[
                SweepParametersSideEffectFactory.func(
                    self.drive2.update_freq, {}, 'freq'
                )
            ]
        )
        
        # Chain sweepers for 2D sweep (freq2 is inner loop)
        return swp_freq1 + swp_freq2
    
    def _add_realistic_noise(self, data, num_avs):
        """Add realistic noise to simulated data."""
        noise_scale = 10.0 / np.sqrt(num_avs)
        noise = (np.random.normal(0, noise_scale, data.shape) + 
                 1j * np.random.normal(0, noise_scale, data.shape))
        return data + noise
    
    def _process_results(self):
        """Process raw trace data into magnitude and phase."""
        if not hasattr(self, 'trace'):
            # For hardware execution, retrieve from setup
            setup = ExperimentManager().get_default_setup()
            raw_data = setup.get_sweep_result()
            # Reshape to 2D
            self.trace = raw_data.reshape(len(self.freq1_arr), len(self.freq2_arr))
        
        self.result = {
            'Magnitude': np.abs(self.trace),
            'Phase': np.angle(self.trace)
        }
    
    @register_browser_function()
    @visual_inspection("""
    Determine if the two-tone spectroscopy shows clear resonances.
    Look for:
    1. Clear peaks or dips in the 2D spectrum
    2. Coupling patterns between the two tones
    3. Power-dependent shifts or splitting
    Return the peak locations if visible.
    """)
    def plot(self):
        """Plot 2D magnitude heatmap."""
        fig = go.Figure(data=go.Heatmap(
            x=self.freq2_arr,
            y=self.freq1_arr,
            z=self.result['Magnitude'],
            colorscale='Viridis',
            colorbar=dict(title='Magnitude')
        ))
        
        fig.update_layout(
            title='Two-Tone Qubit Spectroscopy',
            xaxis_title='Tone 2 Frequency (MHz)',
            yaxis_title='Tone 1 Frequency (MHz)',
            width=800,
            height=600
        )
        
        return fig
    
    @register_browser_function()
    def plot_phase(self):
        """Plot 2D phase heatmap."""
        # Unwrap phase in both dimensions
        phase_unwrapped = np.unwrap(np.unwrap(self.result['Phase'], axis=0), axis=1)
        
        fig = go.Figure(data=go.Heatmap(
            x=self.freq2_arr,
            y=self.freq1_arr,
            z=phase_unwrapped,
            colorscale='RdBu',
            colorbar=dict(title='Phase (rad)')
        ))
        
        fig.update_layout(
            title='Two-Tone Spectroscopy - Phase',
            xaxis_title='Tone 2 Frequency (MHz)',
            yaxis_title='Tone 1 Frequency (MHz)',
            width=800,
            height=600
        )
        
        return fig
    
    def find_peaks(self):
        """Find resonance peaks in 2D spectrum."""
        magnitude = self.result['Magnitude']
        peak_idx = np.unravel_index(np.argmax(magnitude), magnitude.shape)
        
        return {
            'peak_freq1': self.freq1_arr[peak_idx[0]],
            'peak_freq2': self.freq2_arr[peak_idx[1]],
            'peak_magnitude': magnitude[peak_idx],
            'peak_index': peak_idx
        }
    
    def get_cross_section(self, axis='freq1', value=None):
        """Get 1D cross-section of 2D spectrum.
        
        Parameters
        ----------
        axis : str
            'freq1' or 'freq2' - which axis to slice along
        value : float
            Frequency value to slice at. If None, uses peak location.
        
        Returns
        -------
        dict
            Cross-section data with frequencies and magnitude
        """
        if value is None:
            peaks = self.find_peaks()
            if axis == 'freq1':
                idx = peaks['peak_index'][0]
                return {
                    'frequencies': self.freq2_arr,
                    'magnitude': self.result['Magnitude'][idx, :],
                    'slice_freq': self.freq1_arr[idx]
                }
            else:
                idx = peaks['peak_index'][1]
                return {
                    'frequencies': self.freq1_arr,
                    'magnitude': self.result['Magnitude'][:, idx],
                    'slice_freq': self.freq2_arr[idx]
                }
        else:
            if axis == 'freq1':
                idx = np.argmin(np.abs(self.freq1_arr - value))
                return {
                    'frequencies': self.freq2_arr,
                    'magnitude': self.result['Magnitude'][idx, :],
                    'slice_freq': self.freq1_arr[idx]
                }
            else:
                idx = np.argmin(np.abs(self.freq2_arr - value))
                return {
                    'frequencies': self.freq1_arr,
                    'magnitude': self.result['Magnitude'][:, idx],
                    'slice_freq': self.freq2_arr[idx]
                }