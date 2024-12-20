from k_agents.inspection.agents import VisualInspectionAgent
from leeq.experiments.builtin import *
from k_agents.inspection.decorator import visual_inspection
from plotly import graph_objects as go


def get_partial_ai_inspection_results(exp, inspection_method='full', ignore_cache=False):

    assert inspection_method in ['full', 'visual_only', 'fitting_only'], \
        f"inspection_method must be 'full', 'visual_only' or 'fitting_only', got {inspection_method}"

    agents = exp._get_inspection_agents()
    visual_agents = []
    other_agents = []
    for agent in agents:
        if isinstance(agent, VisualInspectionAgent):
            visual_agents.append(agent)
        else:
            other_agents.append(agent)

    agents = []
    # Add the visual inspection results to the AI inspection results.
    if inspection_method != 'fitting_only':
        agents = visual_agents

    # Add the fitting results to the AI inspection results.
    if inspection_method != 'visual_only':
        agents += other_agents

    if ignore_cache:
        exp._ai_inspection_summary = None

    summary = exp._get_ai_inspection_results_from_some_agents(agents)

    if ignore_cache:
        exp._ai_inspection_summary = None

    return summary


def extract_results_from_experiment(exp):
    analyze_results = {
        'full': get_partial_ai_inspection_results(exp, inspection_method='full',
                                                  ignore_cache=True),
        'fitting_only': get_partial_ai_inspection_results(exp,
                                                          inspection_method='fitting_only',
                                                          ignore_cache=True),
        'visual_only': get_partial_ai_inspection_results(exp,
                                                         inspection_method='visual_only',
                                                         ignore_cache=True),
    }
    exp._execute_plot_functions(build_static_image=True)
    return analyze_results


class NormalisedRabiDataValidityCheckRaw(NormalisedRabi):

    @register_browser_function()
    @visual_inspection("""
    Analyze this quantum mechanics Rabi oscillation experiment plot in the Fourier frequency domain. A successful experiment should have a significant peak in the figure.
    """)
    def plot_fft(self) -> go.Figure:
        """
        Plots frequency spectrum of Rabi oscillations using data from an experiment run.

        This method retrieves arguments from the 'run' object, processes the data,
        and then creates a plot using Plotly. The plot features a line plot representing
        the magnitude of the frequency components of the data.
        """

        args = self._get_run_args_dict()
        t = np.arange(args['start'], args['stop'], args['step'])
        delta_t = t[1] - t[0]  # Time step

        # Compute the FFT of the data
        data_fft = np.fft.fft(self.data)
        freqs = np.fft.fftfreq(len(self.data), delta_t)

        # Only take the positive frequencies and their magnitudes
        positive_freqs = freqs[:len(freqs) // 2]
        magnitude = np.abs(data_fft[:len(freqs) // 2])

        # Create the plot
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=positive_freqs,
                y=magnitude,
                mode='lines',
                line=dict(color='Blue'),
                name='Magnitude'
            )
        )

        # Update layout to suit frequency spectrum visualization
        fig.update_layout(
            title='Frequency Spectrum of Rabi Oscillations',
            xaxis_title='Frequency (Hz)',
            yaxis_title='Magnitude',
            legend_title='Legend',
            font=dict(
                family='Courier New, monospace',
                size=12,
                color='Black'
            ),
            plot_bgcolor='white'
        )

        return fig

    #@visual_inspection("""
    #Analyze this quantum mechanics Rabi oscillation experiment plot. Determine if it shows a successful or failed experiment by evaluating:
    #    1. Oscillation behaviour in the figure. It may not be perfect, but it needs to be distinguished from random noise data.
    #    2. Amplitude and frequency consistency. inconsistent oscillation is considered a failure.
    #For example, the following Image is a successful Rabi oscillation experiment plot:
    #Image("openai_rabi_success_cases_0_NormalisedRabiDataValidityCheck.plot.png")
    #the following Image is a failure case for the Rabi experiment:
    #Image("openai_rabi_failure_cases_0_NormalisedRabiDataValidityCheck.plot.png")
    #""")
    def plot(self) -> go.Figure:
        """
        Plots Rabi oscillations using data from an experiment run.

        This method retrieves arguments from the 'run' object, processes the data,
        and then creates a plot using Plotly. The plot features scatter points
        representing the original data and a sine fit for each qubit involved in the
        experiment.
        """

        args = self._get_run_args_dict()
        t = np.arange(args['start'], args['stop'], args['step'])
        t_interpolate = np.arange(
            args['start'],
            args['stop'],
            args['step'] / 5)

        # Create subplots: each qubit's data gets its own plot
        fig = go.Figure()
        # Scatter plot of the actual data
        fig.add_trace(
            go.Scatter(
                x=t,
                y=self.data,
                mode='markers',
                marker=dict(
                    color='Blue',
                    size=7,
                    opacity=0.5,
                    line=dict(color='Black', width=2),
                ),
                name=f'data'
            )
        )

        # Fit data
        f = self.fit_params['Frequency']
        a = self.fit_params['Amplitude']
        p = self.fit_params['Phase'] - 2.0 * np.pi * f * args['start']
        o = self.fit_params['Offset']
        fit = a * np.sin(2.0 * np.pi * f * t_interpolate + p) + o

        # Line plot of the fit
        fig.add_trace(
            go.Scatter(
                x=t_interpolate,
                y=fit,
                mode='lines',
                line=dict(color='Red'),
                name=f'fit',
                visible='legendonly'
            )
        )

        # Update layout for better visualization
        fig.update_layout(
            title='Time Rabi',
            xaxis_title='Time (µs)',
            yaxis_title='<z>',
            legend_title='Legend',
            font=dict(
                family='Courier New, monospace',
                size=12,
                color='Black'
            ),
            plot_bgcolor='white'
        )

        return fig

    @text_inspection
    def fitting(self):
        """
        Get the prompt to analyze the result.

        Returns:
        str: The prompt to analyze the result.
        """

        oscillation_freq = self.fit_params['Frequency']
        experiment_time_duration = self._get_run_args_dict()['stop'] - \
                                   self._get_run_args_dict()['start']
        oscillation_count = (experiment_time_duration * oscillation_freq)

        return {
            "analysis": f"The fitting result of the Rabi oscillation suggest the amplitude of {self.fit_params['Amplitude']}, "
                        f"the frequency of {self.fit_params['Frequency']}, the phase of {self.fit_params['Phase']}. The offset of"
                        f" {self.fit_params['Offset']}. The suggested new driving amplitude is {self.guess_amp}."
                        f"From the fitting results, the plot should exhibit {oscillation_count} oscillations.",
        }


class NormalisedRabiDataValidityCheckImageFewShot(NormalisedRabiDataValidityCheckRaw):
    @register_browser_function()
    @visual_inspection("""
    Analyze this quantum mechanics Rabi oscillation experiment plot in the Fourier frequency domain. A successful experiment should have a significant peak in the figure.
    
    For example, the following Image is a successful Rabi oscillation experiment plot:
    Image("image_refs/rabi_success_NormalisedRabiDataValidityCheck.plot_fft.png")
    the following Image is a failure case for the Rabi experiment:
    Image("image_refs/rabi_failure_NormalisedRabiDataValidityCheck.plot_fft.png")
    """)
    def plot_fft(self) -> go.Figure:
        """
        Plots frequency spectrum of Rabi oscillations using data from an experiment run.

        This method retrieves arguments from the 'run' object, processes the data,
        and then creates a plot using Plotly. The plot features a line plot representing
        the magnitude of the frequency components of the data.
        """

        args = self._get_run_args_dict()
        t = np.arange(args['start'], args['stop'], args['step'])
        delta_t = t[1] - t[0]  # Time step

        # Compute the FFT of the data
        data_fft = np.fft.fft(self.data)
        freqs = np.fft.fftfreq(len(self.data), delta_t)

        # Only take the positive frequencies and their magnitudes
        positive_freqs = freqs[:len(freqs) // 2]
        magnitude = np.abs(data_fft[:len(freqs) // 2])

        # Create the plot
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=positive_freqs,
                y=magnitude,
                mode='lines',
                line=dict(color='Blue'),
                name='Magnitude'
            )
        )

        # Update layout to suit frequency spectrum visualization
        fig.update_layout(
            title='Frequency Spectrum of Rabi Oscillations',
            xaxis_title='Frequency (Hz)',
            yaxis_title='Magnitude',
            legend_title='Legend',
            font=dict(
                family='Courier New, monospace',
                size=12,
                color='Black'
            ),
            plot_bgcolor='white'
        )

        return fig


class ResonatorSweepTransmissionRaw(ResonatorSweepTransmissionWithExtraInitialLPB):
    """
    Class representing a resonator sweep transmission experiment with extra initial LPB.
    Inherits from a generic "experiment" class.
    """

    _experiment_result_analysis_instructions = """
Inspect the plot to detect the resonator's presence. If present:
1. Consider the resonator linewidth (typically sub-MHz to a few MHz).
2. If the step size is much larger than the linewidth:
   a. Focus on the expected resonator region.
   b. Reduce the step size for better accuracy.
3. If linewidth < 0.1 MHz, it's likely not a resonator; the experiment fails.
4. If there are reports from the inspection of the plot and it indicate there is no resonator, believe it and drop the fitting results.
The experiment is considered successful if a resonator is detected. Otherwise, it is considered unsuccessful and suggest a new sweeping range and step size.
"""

    @register_browser_function()
    @visual_inspection("""
Analyze the resonator spectroscopy magnitude plot to determine if it exhibits characteristics indicative of a resonator. 
Specifically, look for a sharp dip or peak in the plot, which would signal the presence of a resonator. 
The rest of the plot should remain relatively stable, without significant fluctuations.
Provide a detailed analysis of the magnitude and frequency data. Identifying a resonator indicates a successful experiment.
    """)
    def plot_magnitude(self):
        args = self._get_run_args_dict()
        f = np.arange(args["start"], args["stop"], args["step"])

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=f,
                y=self.result["Magnitude"],
                mode="lines",
                name="Magnitude"))

        fig.update_layout(
            title="Resonator spectroscopy magnitude",
            xaxis_title="Frequency [MHz]",
            yaxis_title="Magnitude",
            plot_bgcolor="white",
        )

        return fig


class ResonatorSweepTransmissionImageFewShot(ResonatorSweepTransmissionRaw):
    """
    Class representing a resonator sweep transmission experiment with extra initial LPB.
    Inherits from a generic "experiment" class.
    """

    @register_browser_function()
    @visual_inspection("""
Analyze the resonator spectroscopy magnitude plot to determine if it exhibits characteristics indicative of a resonator. 
Specifically, look for a sharp dip or peak in the plot, which would signal the presence of a resonator. 
The rest of the plot should remain relatively stable, without significant fluctuations.
Provide a detailed analysis of the magnitude and frequency data. Identifying a resonator indicates a successful experiment.
For example, the following Image is a successful experiment plot:
Image("image_refs/resonator_spec_success_ResonatorSweepTransmissionWithExtraInitialLPB.plot_magnitude.png")
the following Image is a failure case for the experiment: 
Image("image_refs/resonator_spec_failure_ResonatorSweepTransmissionWithExtraInitialLPB.plot_magnitude.png")
""")
    def plot_magnitude(self):
        args = self._get_run_args_dict()
        f = np.arange(args["start"], args["stop"], args["step"])

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=f,
                y=self.result["Magnitude"],
                mode="lines",
                name="Magnitude"))

        fig.update_layout(
            title="Resonator spectroscopy magnitude",
            xaxis_title="Frequency [MHz]",
            yaxis_title="Magnitude",
            plot_bgcolor="white",
        )

        return fig


class MeasurementCalibrationMultilevelGMMRaw(MeasurementCalibrationMultilevelGMM):
    pass


class MeasurementCalibrationMultilevelGMMImageFewShot(
    MeasurementCalibrationMultilevelGMMRaw):
    @register_browser_function()
    @visual_inspection("""
        Analyze a plot of collected signal data to determine experiment success:
        1. Identify clusters: The signal represents hidden system states, with each state generating a 2D Gaussian distribution (spherical blobs).
        2. Count and evaluate distributions:
           - Treat partially overlapped clusters with two visible density centers as separate distributions.
           - Consider elliptical distributions with only one visible density center as a single distribution.
           - Compare densities of observed distributions.
           - If three or more distributions are present, but only two have major density, consider only the two high-density distributions and ignore the low-density ones.
        3. Experiment outcome:
           - Success: Exactly two major distributions observed (after accounting for density).
           - Failure: Any other outcome (e.g., one distribution, or more than two major distributions).
        For example, the following Image is a successful experiment plot:
        Image("image_refs/gmm_success_MeasurementCalibrationMultilevelGMM.plot_hexbin.png")
        the following Image is a failure case for the experiment: 
        Image("image_refs/gmm_failure_MeasurementCalibrationMultilevelGMM.plot_hexbin.png")
        """)
    def plot_hexbin(self, result_data=None) -> 'matplotlib.figure.Figure':
        """
        Plot the IQ data with the fitted Gaussian Mixture Model using hexbin.

        Parameters:
        result_data (Optional[np.ndarray]): The result data to plot. Defaults to the result data from the experiment.

        Returns:
            Figure: The matplotlib figure.
        """

        if result_data is None:
            data = self.result.flatten()
        else:
            data = result_data.flatten()

        fig = plt.figure(figsize=(3, 3))
        ax = fig.add_subplot(111)

        xs = data.real
        ys = data.imag

        # Create a hexbin map
        hb = ax.hexbin(xs, ys, gridsize=50, cmap='viridis', bins='log')

        # Add a color bar
        # cb = plt.colorbar(hb)
        # cb.set_label('log10(N)')

        # Add labels and title
        ax.set_xlabel('I channel')
        ax.set_ylabel('Q channel')
        ax.set_aspect('equal', adjustable='box')

        return fig


class DragCalibrationSingleQubitMultilevelRaw(DragCalibrationSingleQubitMultilevel):
    pass


class DragCalibrationSingleQubitMultilevelImageFewShot(
    DragCalibrationSingleQubitMultilevelRaw):
    @register_browser_function()
    @visual_inspection(
        """
        Analyze the scatter plot with blue and red data points and trend lines:
            1.Compare the slopes of the trend lines.
            2.Assess how well data points fit their trend lines, noting outliers or patterns.
            3.Evaluate data point distribution along the DRAG coefficient axis.
            4.Determine if trend lines accurately represent their datasets.
            5.Compare trends between the two datasets.
            6.Estimate the fitting residuals.
        Success criteria:
            1.Distinct trends for each color
            2.Appropriate line fitting, with the blue and red lines has significant difference in distribution.
            3.Lines intersect near the plot's center region, small shifts away from the center is acceptable.
            4. Residuals are within acceptable range.
        If criteria aren't met, mark the experiment as failed and suggest a new range for the sweep.
        For example, the following Image is a successful experiment plot:
        Image("image_refs/drag_success_cases_DragCalibrationSingleQubitMultilevel.plot.png")
        the following Image is a failure case for the experiment: 
        Image("image_refs/drag_failure_DragCalibrationSingleQubitMultilevel.plot.png")
        """
    )
    def plot(self):
        self.linear_fit()
        fig = plt.figure()

        plt.plot(self.sweep_values, self.result[:, 0], 'ro', alpha=0.5)
        plt.plot(
            self.sweep_values,
            self.fit_xp[0] *
            self.sweep_values +
            self.fit_xp[1],
            'r-')
        plt.plot(self.sweep_values, self.result[:, 1], 'bo', alpha=0.5)
        plt.plot(
            self.sweep_values,
            self.fit_xm[0] *
            self.sweep_values +
            self.fit_xm[1],
            'b-')
        plt.xlabel(u"DRAG coefficient")
        plt.ylabel(u"<z>")
        # plt.legend()
        return fig
