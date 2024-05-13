from sklearn.pipeline import Pipeline
from typing import Optional, Union, List, Dict, Any
from plotly import graph_objects as go
from plotly import subplots
from sklearn import mixture
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from labchronicle import register_browser_function, log_and_record
from leeq import *
from leeq.core.primitives.logical_primitives import LogicalPrimitiveBlockSweep, LogicalPrimitiveBlockParallel, \
    LogicalPrimitiveBlockSerial
from leeq.setups.built_in.setup_simulation_high_level import HighLevelSimulationSetup
from leeq.theory.simulation.numpy.dispersive_readout.simulator import DispersiveReadoutSimulatorSyntheticData
from leeq.utils import setup_logging
from leeq.utils.prompt import visual_analyze_prompt

logger = setup_logging(__name__)


class CustomRescaler(BaseEstimator, TransformerMixin):
    def __init__(self):
        super().__init__()

    def fit(self, X, y=None):
        # Reshaping the data to be flattened
        data_flattened = X.reshape(-1)

        # Calculating the standard deviation and the rescale factor
        self.scale_ = 1 / np.abs(data_flattened).std()
        return self

    def transform(self, X, y=None):
        return X * self.scale_


def fit_gmm_model(data: np.ndarray, n_components: int,
                  initial_means: Optional[np.ndarray] = None) -> Pipeline:
    """
    Fits a Gaussian Mixture Model Pipeline to the provided data.

    This function standardizes the data using CustomRescaler and then applies GaussianMixture to fit the model.
    Note: The function assumes that the GaussianMixture is the final step in the pipeline and uses it for fitting the model.

    Parameters:
    data (np.ndarray): The input data to fit. Should be a 2D array of shape (n_samples, n_features).
    n_components (int): The number of mixture components for the GaussianMixture.
    initial_means (Optional[np.ndarray]): The user-provided initial means for the GaussianMixture. Default is None.

    Returns:
    GaussianMixture: The fitted Gaussian Mixture model.

    Raises:
    ValueError: If input data is not a 2D array or if n_components is less than 1.
    """

    from sklearn.mixture import GaussianMixture

    if not np.iscomplexobj(data):
        msg = "Input data should be a complex array."
        logger.error(msg)
        raise ValueError(msg)

    data = data.flatten()
    # Transform complex data to real
    data = np.vstack([data.real, data.imag]).T

    if data.ndim != 2:
        msg = "Input data should be a 2D array of shape (n_samples, n_features)."
        logger.error(msg)
        raise ValueError(msg)

    if n_components < 1:
        msg = f"Number of components (n_components) must be greater than or equal to 1, but got {n_components}."
        logger.error(msg)
        raise ValueError(msg)

    # Standardize the data
    pipeline = Pipeline([('scaler', CustomRescaler()),
                         ('gmm', GaussianMixture(n_components=n_components,
                                                 covariance_type='spherical',
                                                 means_init=initial_means))])

    pipeline.fit(data)
    return pipeline


def measurement_transform_gmm(
        data: np.ndarray,
        basis: str,
        clf: Pipeline,
        output_map: Dict[int, int],
        z_threshold: int = 1
) -> Union[np.ndarray, float]:
    """
    Transforms measurement data based on the specified basis using Gaussian Mixture Model (GMM).

    Parameters:
    data (np.ndarray): The input data to be transformed.
    basis (str): The basis used for transformation. It can be '', 'raw', '<zs>', 'bin', 'prob', 'p(0)', 'p(1)', or '<z>'.
    clf (GaussianMixture): The GMM classifier used for prediction.
    output_map (Dict[int, int]): A dictionary mapping the output of the GMM to specific integers.
    z_threshold (int): The threshold used for determining the zero point in the '<z>' basis. Defaults to 1.

    Returns:
    Union[np.ndarray, float]: The transformed data based on the specified basis.

    Raises:
    RuntimeError: If an unknown basis string is provided.
    """

    # If basis is empty or 'raw', return the original data
    if basis in ('', 'raw'):
        return data

    original_shape = data.shape

    # The input shape should be
    # (1, measurement_id, n_samples)

    data_flat = data.flatten()
    # Transform complex data to real
    data_complex_to_real = np.vstack([data_flat.real, data_flat.imag]).T

    # Predict using the Gaussian Mixture Model classifier
    output = clf.predict(data_complex_to_real)
    # Map the output using output_map, ignore unmapped values
    output_mapped = np.asarray([output_map[x]
                                for x in output if x in output_map])

    output_reshaped = output_mapped.reshape(original_shape)

    max_output = max(output_map.values())

    if basis == '<zs>':
        return output_reshaped

    # Count the occurrences of each unique value in output_reshaped

    bins = np.asarray([np.sum((output_reshaped == i).astype(int))
                       for i in range(max_output + 1)])

    if basis == 'bin':
        return bins[np.newaxis, :]

    if basis == 'prob':
        # Normalize the bin counts to get probabilities
        return (bins / np.sum(bins))[np.newaxis, :]

    zero_count = np.sum((output_reshaped < z_threshold).astype(int), axis=-2)
    one_count = np.sum((output_reshaped >= z_threshold).astype(int), axis=-2)

    z = (zero_count - one_count) / (zero_count + one_count)

    # Calculate probabilities or return z based on basis
    if basis == 'p(0)':
        return (z[np.newaxis, :] + 1) / 2
    elif basis == 'p(1)':
        return (z[np.newaxis, :] - 1) / -2
    elif basis == '<z>':
        return z[np.newaxis, :]
    else:
        msg = f"Unknown basis {basis}"
        logger.error(msg)
        raise RuntimeError(msg)


def find_output_map(data: np.ndarray, clf: Pipeline) -> Dict[int, int]:
    """
    Constructs an outcome map based on the predictions of a classifier.

    This function primarily serves to map the states predicted by a classifier
    to an outcome map. The process first assigns states based on population, from lower to higher energy levels.
    If states remain unassigned, it then uses higher populations for lower energy levels.

    Parameters:
    data (np.ndarray): The input data for prediction, with shape (n_samples, n_features).
    n_components (int): The total number of distinct states/components.
    clf (PipeLine): A classifier pipeline instance that should have a 'predict' method.

    Returns:
    Dict[int, int]: A dictionary mapping the predicted state to an outcome index.

    Raises:
    RuntimeError: If wrong classification leads to unhandled states.
    """

    outcome_map: Dict[int, int] = {}
    used_state: List[int] = []
    n_components: int = clf.named_steps['gmm'].n_components

    # Iterate over the data to classify states and build the outcome map.
    for i in range(data.shape[1]):
        # Prepare data for prediction by stacking real and imaginary parts.
        prediction_data = np.vstack(
            [np.real(data[:, i]), np.imag(data[:, i])]).transpose()
        prediction_for_this_state = clf.predict(prediction_data)

        # Order states by their population count in descending order.
        indexes = np.argsort(np.bincount(prediction_for_this_state))[::-1]

        # Assign the state to the outcome map if it's not already used.
        for j in indexes:
            if j not in used_state:
                outcome_map[j] = i
                used_state.append(j)
                break

    # Check if there are any states left unassigned.
    if len(used_state) < n_components:
        # Handle the missing states.
        prediction_data_last = np.vstack(
            [np.real(data[:, -1]), np.imag(data[:, -1])]).transpose()
        prediction_for_last_state = clf.predict(prediction_data_last)
        used_state_array = np.array(used_state)
        unused_state = [x for x in range(n_components) if x not in used_state]

        # Raise an error if there's a classification issue leading to unhandled
        # states.
        for i in unused_state:
            if i not in used_state and (i < used_state_array).any():
                msg = "Wrong classification leads to unhandled states."
                logger.error(msg)
                raise RuntimeError(msg)

        # Assign the remaining states to the outcome map.
        indexes = np.argsort(np.bincount(prediction_for_last_state))[::-1]
        for j in indexes:
            if j not in used_state:
                outcome_map[j] = len(outcome_map)

    # Sanity check: the outcome map should have entries equal to n_components.
    assert len(
        outcome_map) == n_components, "Outcome map size does not match the number of components."

    return outcome_map


def calculate_signal_to_noise_ratio(
        clf: Pipeline, outcome_map: List[int]) -> Dict[tuple, float]:
    """
    Calculate the signal-to-noise ratio (SNR) for a Gaussian Mixture Model.

    Parameters:
    clf (Pipeline): Classifier pipeline containing the GMM.
    outcome_map (List[int]): List of outcomes for the GMM components.

    Returns:
    Dict[tuple, float]: Dictionary with tuples of component indices as keys and their corresponding SNR as values.

    Raises:
    ValueError: If 'gmm' key is not present in the clf dictionary.
    """
    # Ensure 'gmm' key exists in the classifier dictionary
    if 'gmm' not in clf.named_steps:
        raise ValueError("Key 'gmm' not found in the classifier dictionary.")

    # Extract GMM properties
    cov = clf.named_steps['gmm'].covariances_
    means = clf.named_steps['gmm'].means_
    n_components = clf.named_steps['gmm'].n_components

    # Initialize the SNR dictionary
    snr = {}

    # Compare the number of outcomes with the number of GMM components
    if len(outcome_map) < n_components:
        # If there are fewer outcomes, set SNR to 0 for all component pairs
        for i in range(n_components):
            for j in range(i, n_components):
                snr[(i, j)] = 0
    else:
        # Otherwise, calculate the SNR for each pair of components
        for i in range(n_components):
            for j in range(
                    i +
                    1,
                    n_components):  # Start from i+1 to avoid self-comparison

                # Calculate standard deviations and distances
                std_i = np.sqrt(cov[i])
                means_i = means[i, :]
                std_j = np.sqrt(cov[j])
                means_j = means[j, :]
                dist = (means_i - means_j)

                # Compute SNR
                snr_value = np.linalg.norm(dist, ord=2) * 2 / (std_i + std_j)

                # Order outcomes
                a, b = min(
                    outcome_map[i], outcome_map[j]), max(
                    outcome_map[i], outcome_map[j])

                # Update SNR dictionary
                snr[(a, b)] = snr_value

    return snr


class MeasurementCalibrationMultilevelGMM(Experiment):
    @log_and_record
    def run(self,
            dut: 'TransmonElement',
            # Replace with actual class
            sweep_lpb_list: List['LogicalPrimitiveBlock'],
            mprim_index: int,
            freq: Optional[float] = None,
            amp: Optional[float] = None,
            update: bool = False,
            extra_readout_duts: Optional[List['DeviceUnderTest']] = None,
            z_threshold: Optional[int] = None) -> None:
        """
        Run the measurement process on a transmon qubit, potentially
        altering frequency and amplitude, and calculate the signal-to-noise ratio.

        Parameters:
        dut (TransmonElement): The qubit instance.
        sweep_lpb_list (List[LogicalPrimitiveBlock]): List of LPBs to be included in the sweep.
        mprim_index (int): Index of the measurement primitive in use.
        freq (Optional[float]): New frequency to set, if any.
        amp (Optional[float]): New amplitude to set, if any.
        update (bool): Flag indicating if the original frequency/amplitude should be restored.
        extra_readout_duts (Optional[List[DeviceUnderTest]]): Additional DUTs for readout.
        z_threshold (Optional[int]): Threshold for measurement, defaults to mprim_index + 1 or mprim_index.

        Returns:
        None: This method updates class attributes with the results.
        """

        self.result = None

        # Retrieve the measurement primitive by index
        mprim = dut.get_measurement_prim_intlist(str(mprim_index))

        # Set default z_threshold if not provided
        if z_threshold is None:
            z_threshold = mprim_index + 1 if mprim_index < 3 else mprim_index

        # Preserve original frequency and amplitude
        original_freq = mprim.freq
        original_amp = mprim.get_pulse_args("amp")

        # Reset any existing transform function
        mprim.set_transform_function(None)

        # Update frequency and amplitude if new values are provided
        if freq is not None:
            mprim.update_freq(freq)
        if amp is not None:
            mprim.update_pulse_args(amp=amp)

        # Initialize LPB sweep with provided list
        sweep_lpb = LogicalPrimitiveBlockSweep(children=sweep_lpb_list)

        # If there are additional DUTs for readout, prepare them
        if extra_readout_duts is not None:
            mprims = [
                d.get_measurement_prim_intlist(
                    str(mprim_index)).clone() for d in extra_readout_duts]
            for m in mprims:
                # Reset transform function for extra readouts
                m.set_transform_function(None)
            mprims.append(mprim)  # Append the original mprim to the list
        else:
            mprims = [mprim]

        # Create logical primitive block with parallel processing
        lpb = sweep_lpb + LogicalPrimitiveBlockParallel(mprims)

        # Prepare the sweeper with LPB sweep
        swp = Sweeper.from_sweep_lpb(sweep_lpb)

        # Execute the experiment
        ExperimentManager().run(lpb, swp)

        # Format the result and update the class attribute
        result = np.squeeze(mprim.result()).transpose()
        self.result = result

        self.analyze_gmm_result(dut, mprim_index, result)

        # Apply the GMM transform function with the fitted model
        mprim.set_transform_function(measurement_transform_gmm,
                                     clf=self.clf,
                                     output_map=self.output_map,
                                     z_threshold=z_threshold)

        # If not updating, restore original frequency and amplitude
        if not update:
            if freq is not None:
                mprim.update_freq(original_freq)
            if amp is not None:
                mprim.update_pulse_args(amp=original_amp)

    def analyze_gmm_result(self,
                           dut: 'TransmonElement',
                           mprim_index: int,
                           result: np.ndarray,
                           ):
        """
        Analyze the result data using a Gaussian Mixture Model (GMM) and update the measurement primitive.
        Parameters
        ----------
        dut : TransmonElement
            The transmon qubit instance.
        mprim_index : int
            The index of the measurement primitive in use.
        result : np.ndarray
            The result data to analyze.
        Returns
        """

        mprim = dut.get_measurement_prim_intlist(str(mprim_index))

        # Retrieve calibration data for the measurement primitive
        mprim_params = dut.get_calibrations()['measurement_primitives']
        mprim_param = mprim_params[str(mprim_index)]
        n_components = len(
            mprim_param['distinguishable_states'])  # result.shape[1]

        # Attempt to get an initial guess for GMM means if available
        initial_gmm_mean_guess = mprim_param.get('gmm_mean_guess', None)
        if initial_gmm_mean_guess is None:
            transform_func, transform_kwargs = mprim.get_transform_function()
            if transform_func == measurement_transform_gmm:
                initial_gmm_mean_guess = transform_kwargs['clf'].means_

        # Fit the GMM model to the result data
        clf = fit_gmm_model(result, n_components, initial_gmm_mean_guess)

        # Determine the output map from the fitted model
        outcome_map = find_output_map(result, clf)

        # Update class attributes with model and parameters
        self.clf = clf
        self.clf_params = clf.get_params()
        self.output_map = outcome_map

        # Calculate and store the signal-to-noise ratio
        self.snr = calculate_signal_to_noise_ratio(clf, outcome_map)

    @log_and_record(overwrite_func_name='MeasurementCalibrationMultilevelGMM.run')
    def run_simulated(self,
                      dut: 'DeviceUnderTest',
                      # Replace with actual class
                      sweep_lpb_list: List['LogicalPrimitiveBlock'],
                      mprim_index: int,
                      freq: Optional[float] = None,
                      amp: Optional[float] = None,
                      update: bool = False,
                      extra_readout_duts: Optional[List['DeviceUnderTest']] = None,
                      z_threshold: Optional[int] = None) -> None:
        """
        Run the measurement process on a transmon qubit, potentially altering frequency and amplitude,
         and calculate the signal-to-noise ratio. Note that the sweep_lpb_list only used to indicate the number of
            distinguishable states, and the content of lpbs are not used.

         Same as the run method, but uses simulated data instead of actual data.

        Returns
        -------
        """

        assert len(sweep_lpb_list) <= 4, ("Only less than 4 LPBs are allowed for simulated data, which represent"
                                          "to prepare to the |0>, |1>, |2>, |3> state.")

        simulator_setup: HighLevelSimulationSetup = setup().get_default_setup()
        virtual_transmon = simulator_setup.get_virtual_qubit(dut)
        mp = dut.get_measurement_prim_intlist(str(mprim_index))

        if freq is not None:
            mp.update_freq(freq)

        if amp is None:
            amp = mp.get_pulse_args("amp")

        baseline = 0
        pulse_args = mp.get_parameters()
        width = pulse_args.get("width", 8)
        rise = pulse_args.get("rise", 0)
        trunc = pulse_args.get("trunc", 1.2)
        sampling_rate = 1e1
        f_r = virtual_transmon.readout_frequency
        kappa = virtual_transmon.readout_linewidth
        chis = np.cumsum([virtual_transmon.readout_dipsersive_shift] * 4)
        quiescent_state_distribution = virtual_transmon.quiescent_state_distribution

        def _get_state_based_on_quiescent_state_distribution(target_state):
            """
            Get the state based on the quiescent state distribution. If target state == 0,
            return a sample of the quiescent state distribution. If it equals to 1, swap 0 and 1
            then return a sample of the quiescent state distribution. If it equals to 2, swap 0 and 2, 1 and 0.
            Parameters
            ----------
            target_state : int
                The target state.

            Returns
            -------
            int
                The state based on the quiescent state distribution.
            """
            if target_state == 0:
                r = np.random.choice([0, 1, 2, 3], p=quiescent_state_distribution)
            elif target_state == 1:
                r = np.random.choice([1, 0, 2, 3], p=quiescent_state_distribution)
            elif target_state == 2:
                r = np.random.choice([1, 2, 0, 3], p=quiescent_state_distribution)
            elif target_state == 3:
                r = np.random.choice([1, 2, 3, 0], p=quiescent_state_distribution)
            else:
                raise ValueError("The target state is not supported.")

            return int(r)

        simulator = DispersiveReadoutSimulatorSyntheticData(
            f_r, kappa, chis, amp, baseline, width,
            rise, trunc, sampling_rate,
        )

        shot_number = setup().status().get_param('Shot_Number')

        data = np.asarray([[simulator._simulate_trace(
            state=_get_state_based_on_quiescent_state_distribution(i), noise_std=0.005, f_prob=mp.freq
        ).sum() for i in range(len(sweep_lpb_list))] for x in range(shot_number)])

        self.result = data

        self.analyze_gmm_result(dut, mprim_index, self.result)

    @register_browser_function(available_after=(run,))
    @visual_analyze_prompt("""
    Describe the Image: Provide a brief description of the image, particularly focusing on the distribution of points,
        difference between two distributions and the overlap of clusters if any.
    Identify Clusters: Note the color and arrangement of the clusters. Are there distinct groups of different colors 
        (e.g., red and blue)?
    Assess Overlap: Look at the central areas of the plot where the clusters might overlap. How much do the clusters 
        mix in these regions?
    Assess Distribution: Look at the distribution 0 and distribution 1. Do they have a clear difference?
    Check Circles/Ellipses: Observe the circles or ellipses drawn around the clusters. Do these shapes encompass
        predominantly one color, or do they contain a significant mixture of both colors?
    Legend Information: If there are percentages or ratios in the legend, they might indicate the extent of separation
        or overlap. Lower values suggest less overlap and better separation.
    Overall Impression: Based on the above factors, decide 1) if the clusters are well separated (little to no overlap,
        clear boundaries) or not well separated (significant overlap, indistinct boundaries). 2) if the two distributions
         have clear difference. If they have difference but not separated means the experiment works but fitting failed.
          If there is no difference means the experiment failed.
    """)
    def gmm_iq(self, result_data=None):
        """
        Plot the IQ data with the fitted Gaussian Mixture Model with matplotlib.

        Parameters:
        result_data (Optional[np.ndarray]): The result data to plot. Defaults to the result data from the experiment.

        Returns:
        """
        from matplotlib import pyplot as plt

        if result_data is None:
            result_data = self.result

        fig = plt.figure(figsize=(result_data.shape[1] * 3.5, 3.5))
        colors = [
            '#1f77b4',
            '#d62728',
            '#2ca02c',
            '#ff7f0e',
            '#9467bd',
            '#8c564b',
            '#e377c2',
            '#7f7f7f',
            '#bcbd22',
            '#17becf']

        for i in range(result_data.shape[1]):
            ax = fig.add_subplot(int(f"1{result_data.shape[1]}{i + 1}"))

            ax.set_title(f"Distribution {i}")
            data = np.vstack([
                np.real(result_data[:, i]),
                np.imag(result_data[:, i])
            ]).transpose()

            state_label = self.clf.predict(data)
            data = self.clf.named_steps['scaler'].transform(data)

            for index in np.unique(state_label):
                percentage = np.average((state_label == index).astype(int))
                ax.scatter(data[:, 0][state_label == index], data[:, 1][state_label == index],
                           alpha=0.5, label=str(self.output_map[index]) + ":" + f"{percentage * 100:.2f}%",
                           color=colors[index], s=3)

                mean = self.clf.named_steps['gmm'].means_[index]
                std = np.sqrt(self.clf.named_steps['gmm'].covariances_[index])
                ax.plot(
                    mean[0],
                    mean[1],
                    "x",
                    markersize=10,
                    color=colors[index])

                x = np.linspace(0, 2 * np.pi, 1001)

                ax.plot(std * 3 * np.cos(x) + mean[0],
                        std * 3 * np.sin(x) + mean[1], color=colors[index])

                ax.set_xlabel('I')
                ax.set_ylabel('Q')
                ax.legend()
                ax.axis('equal')

        plt.tight_layout()

        return fig

    def gmm_iq_plotly(self, result_data=None) -> go.Figure:
        """
        Plot the IQ data with the fitted Gaussian Mixture Model.

        Parameters:
        result_data (Optional[np.ndarray]): The result data to plot. Defaults to the result data from the experiment.

        Returns:
        """

        print({"Means": self.clf.named_steps['gmm'].means_,
               "Cov": self.clf.named_steps['gmm'].covariances_})

        colors = [
            '#1f77b4',
            '#d62728',
            '#2ca02c',
            '#ff7f0e',
            '#9467bd',
            '#8c564b',
            '#e377c2',
            '#7f7f7f',
            '#bcbd22',
            '#17becf']

        if result_data is None:
            result_data = self.result

        # Create subplots: the layout is 1 x number of features, and it shares
        # x and y axes
        fig = subplots.make_subplots(
            rows=1,
            cols=result_data.shape[1],
            shared_xaxes=True,
            shared_yaxes=True)

        for i in range(result_data.shape[1]):
            data = np.vstack([
                np.real(result_data[:, i]),
                np.imag(result_data[:, i])
            ]).transpose()

            state_label = self.clf.predict(data)
            data = self.clf.named_steps['scaler'].transform(data)

            for index in np.unique(state_label):
                percentage = np.average((state_label == index).astype(int))

                # Scatter plot for the data points
                fig.add_trace(
                    go.Scatter(
                        x=data[:, 0][state_label == index],
                        y=data[:, 1][state_label == index],
                        mode='markers',
                        marker=dict(color=colors[index], size=3, opacity=0.3),
                        name=str(
                            self.output_map[index]) + ":" + f"{percentage * 100:.2f}%",
                        legendgroup=int(index),  # to tie legend items together
                    ),
                    row=1,  # since we are using 1 row
                    col=i + 1  # column is the current feature index + 1
                )

                mean = self.clf['gmm'].means_[index]
                std = np.sqrt(self.clf['gmm'].covariances_[index])

                # Plot the means as 'X'
                fig.add_trace(
                    go.Scatter(
                        x=[mean[0]],
                        y=[mean[1]],
                        mode='markers',
                        marker_symbol='x',
                        marker=dict(color=colors[index], size=10),
                        legendgroup=int(index),
                        showlegend=False
                    ),
                    row=1,
                    col=i + 1
                )

                # Draw the standard deviation circle
                x = np.linspace(0, 2 * np.pi, 1001)
                fig.add_trace(
                    go.Scatter(
                        x=std * 3 * np.cos(x) + mean[0],
                        y=std * 3 * np.sin(x) + mean[1],
                        mode='lines',
                        line=dict(color=colors[index]),
                        legendgroup=int(index),
                        showlegend=False
                    ),
                    row=1,
                    col=i + 1
                )

            # Update axes labels and title
            fig.update_xaxes(title_text="I", row=1, col=i + 1)
            fig.update_yaxes(
                title_text="Q", row=1, col=i + 1,
                scaleanchor="x",
                scaleratio=1,
            )

            fig.update_layout(
                title_text=f"Distribution {i}",
                title_x=0.5,
                # title_xanchor='center',
                title_yanchor='top',
                plot_bgcolor='white'
            )

        return fig

    def live_plots(self, step_no):
        """
        Plot the IQ data with the fitted Gaussian Mixture Model.

        Parameters:
            step_no (tuple[int]): The step number to plot.
        """
        if self.result is None:
            return go.Figure()
