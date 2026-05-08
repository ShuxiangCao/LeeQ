from .common import *
from .continuous import ConditionalStarkShiftContinuous
from .repeated import ConditionalStarkShiftRepeatedGate

class ConditionalStarkEchoTuneUpAI(Experiment):
    """
    Class for performing Conditional Stark Echo Tune-Up experiments.
    """

    _experiment_result_analysis_instructions = """
The Conditional Stark Echo Tune-Up experiment has been completed. Please read the following report to analyze the
if this is a successful experiment. Make the analysis concise and clear in one short sentence describing the reason.
"""

    def run_simulated(self, *args, **kwargs):
        return self.run(*args, **kwargs)

    @log_and_record
    def run(
            self,
            duts: List[Any],
            params: Dict[str, Any] = None,
            frequency: float = None,
            amplitude: float = None,
            phase_diff: float = 0,
            rise: float = 0.015,
            t_start: float = 0,
            t_stop: float = 15,
            sweep_points: int = 30,
            n_start: int = 0,
            n_stop: int = 32,
            update_iz: bool = False,
            update_zz: bool = True,
            n_max_iteration: int = 1,
            zz_accuracy_threshold: float = 0.001,
            zz_uncertainty_threshold: float = 0.03,
            iz_accuracy_threshold: float = 0.001,
            iz_uncertainty_threshold: float = 0.03,
            ai_inspection: bool = False
    ) -> None:
        """
        Run the Conditional Stark Echo Tune-Up experiment to calibrate the siZZel two qubit gate parameters for a
        pair of qubits.

        Args:
            duts (List[Any]): Devices under test.
            params (Dict[str, Any], optional): Parameters for the experiment. Defaults to None.
            frequency (float, optional): Frequency for the experiment. Defaults to None.
            amplitude (float, optional): Amplitude control for the experiment. Defaults to None.
            phase_diff (float, optional): Phase difference for the experiment. Defaults to 0.
            rise (float, optional): Rise time for the experiment. Defaults to 0.01.
            t_start (float, optional): Start time for the sweep. Defaults to 0.
            t_stop (float, optional): Stop time for the sweep. Defaults to 20.
            sweep_points (int, optional): Number of points in the sweep. Defaults to 30.
            n_start (int, optional): Start number for gate iteration. Defaults to 0.
            n_stop (int, optional): Stop number for gate iteration. Defaults to 32.
            update_iz (bool, optional): Flag to update IZ. Defaults to False.
            update_zz (bool, optional): Flag to update ZZ. Defaults to True.
            n_max_iteration (int, optional): Maximum number of iterations. Defaults to 20.
            zz_accuracy_threshold (float, optional): Accuracy threshold for ZZ. Defaults to 0.001.
            zz_uncertainty_threshold (float, optional): Uncertainty threshold for ZZ. Defaults to 0.03.
            iz_accuracy_threshold (float, optional): Accuracy threshold for IZ. Defaults to 0.001.
            iz_uncertainty_threshold (float, optional): Uncertainty threshold for IZ. Defaults to 0.03.
            ai_inspection (bool, optional): Flag for AI inspection. Defaults to False. Please set it to True if you
                want to use the AI inspection feature, or you are an AI writing the code.
        """
        self.duts = duts
        self.n_max_iteration = n_max_iteration
        self.zz_accuracy_threshold = zz_accuracy_threshold
        self.zz_uncertainty_threshold = zz_uncertainty_threshold
        self.iz_accuracy_threshold = iz_accuracy_threshold
        self.iz_uncertainty_threshold = iz_uncertainty_threshold

        self.ai_inspection = True

        if update_iz:
            raise ValueError("update_iz must be False.")

        if params is None:
            amp_rabi_control = duts[0].get_c1('f01')['X'].amp
            amp_rabi_target = duts[1].get_c1('f01')['X'].amp

            area_control = amp_rabi_control * duts[0].get_c1('f01')['X'].width
            area_target = amp_rabi_target * duts[1].get_c1('f01')['X'].width

            params = {
                'iz_control': 0,
                'iz_target': 0,
                'frequency': frequency,
                'amp_control': amplitude,
                'amp_target': amplitude * area_target / area_control,
                'rise': rise,
                'width': 0,
                'phase_diff': phase_diff,
                'zz_interaction_positive': True,
                'echo': True
            }

        self.current_params = params
        self.params_list = [params]

        try:
            iz_rate, zz_rate, self._xy_hamiltonian_tomography_inspection_results = self.run_sizzel_xy_hamiltonian_tomography(
                t_start=t_start, t_stop=t_stop, sweep_points=sweep_points
            )
        except Exception as e:
            self._xy_hamiltonian_tomography_inspection_results = {'success': False,
                                                                  'analysis': f'Exception occurred: {e}'
                                                                  }
            self._repeated_gate_inspection_results = {'success': False,
                                                      'analysis': f'Exception occurred: {e}'
                                                      }
            return

        if not self._xy_hamiltonian_tomography_inspection_results['success']:
            self._repeated_gate_inspection_results = {'success': False,
                                                      'analysis': (
                                                          'Skipped due to the failure of '
                                                          'Hamiltonian tomography experiment.')
                                                      }
            return

        if self.current_params['width'] > 0.4:
            self._repeated_gate_inspection_results = {'success': True,
                                                      'analysis': (
                                                          'Skipped due the width estimation evaluated from hamiltonian tomography is too long.')
                                                      }
            return

        self.current_params['zz_interaction_positive'] = zz_rate.nominal_value > 0

        try:
            self._repeated_gate_inspection_results = self.run_repeated_gate_hamiltonian_tomography(
                zz_rate=zz_rate, n_start=n_start, n_stop=n_stop, update_iz=False,
                update_zz=True
            )
        except Exception as e:
            self._repeated_gate_inspection_results = {'success': False,
                                                      'analysis': f'Exception occurred: {e}'
                                                      }

    @text_inspection
    def fitting(self) -> Union[str, None]:

        res_dict = {
            "Inspection results from hamiltonian_tomography": self._xy_hamiltonian_tomography_inspection_results,
            "Inspection results from repeated_gate_hamiltonian_tomography": self._repeated_gate_inspection_results,
            "fitted parameters": self.current_params
        }

        return res_dict

    def _check_data_validity_using_ai(self, experiment: Experiment,
                                      additional_information: str, show=True) -> dict[
            str, str]:
        if not self.ai_inspection:
            return {
                'analysis': 'AI inspection is not enabled. Always assumes the data is valid.',
                'success': True,
            }

        inspection_results = experiment.get_ai_inspection_summary()

        prompt = f"""
        You are asked to read the report of the data inspection AI and look at the results reported from a fitting code.
        Please check the inspection results and confirm whether the experiment data is valid from the inspection.
        Also check the validity of the fitted results and ensure they are reasonable and physical. If the data is invalid
        or the fitting results are invalid, the experiment is considered a failure. Otherwise, the experiment is
        considered a success.

        <Inspection results>
        {inspection_results}
        </Inspection results>

        <Fitting results>
        {additional_information}
        </Fitting results>

        """ + """
        <Return format>
        {
            'analysis': str,
            'success': bool,
        }
        </Return format>
        """

        chat = Chat(prompt,
                    "You are a very smart and helpful assistant who only reply in JSON dict. "
                    + "Keep everything in a same line in the response.")
        res = chat.complete(parse="dict", expensive=True, cache=True)

        html = dict_to_html(res)
        display_chat(agent_name="Inspection AI", content='<br>' + html,
                     background_color='#f0f8ff')

        return res

    def run_sizzel_xy_hamiltonian_tomography(
            self, t_start: float, t_stop: float, sweep_points: int = 60
    ) -> Tuple[Any, Any, dict]:
        """
        Run the SiZZle XY Hamiltonian tomography.

        Args:
            t_start (float): Start time for the sweep.
            t_stop (float): Stop time for the sweep.
            sweep_points (int, optional): Number of points in the sweep. Defaults to 60.

        Returns:
            Tuple[Any, Any, dict]: Measured IZ rate and ZZ rate and the inspection results.
        """
        setup().status().set_param("Shot_Period", 500)
        setup().status().set_param("Shot_Number", 500)

        (t_stop - t_start) / sweep_points

        if self.ai_inspection:

            sizzel_xy = ConditionalStarkShiftContinuous(duts=self.duts, frequency=self.current_params['frequency'],
                                                        amp_control=self.current_params['amp_control'],
                                                        amp_target=self.current_params['amp_target'],
                                                        rise=self.current_params['rise'],
                                                        start=t_start, stop=t_stop, sweep_points=sweep_points,
                                                        phase_diff=self.current_params['phase_diff'], echo=True)

            inspection_results = sizzel_xy.get_ai_inspection_summary()
        else:
            sizzel_xy = ConditionalStarkShiftContinuous(
                qubits=self.duts,
                frequency=self.current_params['frequency'],
                amp_control=self.current_params['amp_control'],
                amp_target=self.current_params['amp_target'],
                rise=self.current_params['rise'],
                start=t_start,
                stop=t_stop,
                sweep_points=sweep_points,
                phase_diff=self.current_params['phase_diff'],
                iz_rate_cancel=0,
                iz_rise_drop=0,
                echo=True
            )

            inspection_results = {
                'analysis': 'AI inspection is not enabled. Always assumes the data is valid.',
                'success': True,
            }

        result = sizzel_xy.analyze_results_with_errs()

        try:
            iz_rate = result['iz_rate']
            zz_rate = result['zz_rate']

            new_params = self.current_params.copy()
            new_params['width'] = np.abs(0.125 / zz_rate.nominal_value) / 2

            f'Estimated IZ = {iz_rate} MHz, ZZ = {zz_rate} MHz, width = {new_params["width"]} us'

            self.params_list.append(new_params)
            self.current_params = new_params
        except Exception:
            iz_rate = None
            zz_rate = None
            inspection_results['success'] = False
            inspection_results['error'] = "Failed to analyze the results. Fitting error"

        return iz_rate, zz_rate, inspection_results

    def run_repeated_gate_hamiltonian_tomography(
            self,
            zz_rate: Any,
            n_start: int = 0,
            n_stop: int = 32,
            update_iz: bool = False,
            update_zz: bool = True
    ) -> None:
        """
        Run repeated gate Hamiltonian tomography.

        Args:
            duts (List[Any]): Devices under test.
            zz_rate (Any): Measured ZZ rate.
            n_start (int, optional): Start number for gate iteration. Defaults to 0.
            n_stop (int, optional): Stop number for gate iteration. Defaults to 32.
            update_iz (bool, optional): Flag to update IZ. Defaults to False.
            update_zz (bool, optional): Flag to update ZZ. Defaults to True.
        """
        iz_target = self.current_params['iz_target']
        width = self.current_params['width']
        iz_check_pass = False
        zz_check_pass = False

        measured_iz_list = []
        measured_zz_list = []

        estimated_iz_list = []
        estimated_zz_list = []

        kalman_iz = None
        kalman_zz = None

        for _i in range(self.n_max_iteration):

            try_count = 0
            while try_count < 1:
                if self.ai_inspection:

                    prompt = f"""
                        Please implement the ConditionalStarkShiftRepeatedGate experiment with the provided parameters.

                        The experiment should be run with the following parameters:
                        - duts: duts in the available variable
                        - frequency: {self.current_params['frequency']}
                        - amp_control: {self.current_params['amp_control']}
                        - amp_target: {self.current_params['amp_target']}
                        - rise: {self.current_params['rise']}
                        - start_gate_number: {n_start}
                        - gate_count: {n_stop}
                        - width: {width}
                        - phase_diff: {self.current_params['phase_diff']}
                        - echo: True
                    """

                    """
                    next_stage_guide = "Go to Complete if success. Otherwise Fail."
                    ai_experiment = OneInstExecutionAgent(
                        prompt,
                        duts=self.duts,
                        next_stage_guide=next_stage_guide)
                    repeated_gate = ai_experiment.get_last_experiment()
                    """

                    ai_experiment_var_table = execute_experiment_from_instruction(
                        prompt=prompt, duts=self.duts,
                    )
                    repeated_gate = get_exp_from_var_table(ai_experiment_var_table)
                else:
                    repeated_gate = ConditionalStarkShiftRepeatedGate(
                        duts=self.duts,
                        iz_control=0,
                        iz_target=iz_target,
                        frequency=self.current_params['frequency'],
                        amp_control=self.current_params['amp_control'],
                        amp_target=self.current_params['amp_target'],
                        rise=self.current_params['rise'],
                        width=width,
                        start_gate_number=n_start,
                        gate_count=n_stop,
                        echo=True,
                    )

                iz_target_measured = iz_target + repeated_gate.iz_rate.nominal_value * np.pi * 2
                zz_measured = repeated_gate.zz_rate.nominal_value

                fitted_results_str = f'Estimated pgc  IZ = {iz_target_measured}, ZZ = {zz_measured} MHz, width = {width} us'

                inspection_results = self._check_data_validity_using_ai(repeated_gate,
                                                                        fitted_results_str)

                if inspection_results['success']:
                    break
                else:
                    try_count += 1

            if try_count == 3:
                return inspection_results

            measured_iz_list.append(iz_target_measured)
            measured_zz_list.append(zz_measured)

            if kalman_iz is None:
                kalman_iz = KalmanFilter1D(
                    initial_position=iz_target_measured,
                    position_variance=(repeated_gate.iz_rate.std_dev * np.pi * 2) ** 2
                )
                kalman_zz = KalmanFilter1D(
                    initial_position=zz_measured,
                    position_variance=(repeated_gate.zz_rate.std_dev * np.pi * 2) ** 2
                )
            else:
                kalman_iz.update(
                    measurement=iz_target_measured,
                    measurement_variance=(repeated_gate.iz_rate.std_dev * np.pi * 2) ** 2
                )

                kalman_zz.update(
                    measurement=zz_measured,
                    measurement_variance=(repeated_gate.zz_rate.std_dev * np.pi * 2) ** 2
                )


            if update_iz:
                iz_target = kalman_iz.x
                iz_check_pass = kalman_iz.P < self.iz_uncertainty_threshold
                if not update_zz:
                    estimated_iz_list.append(ufloat(kalman_iz.x, np.sqrt(kalman_iz.P)))

            if update_zz:
                zz_pgc = kalman_zz.x

                target_zz = np.sign(zz_pgc) * 0.125
                zz_diff = target_zz - zz_pgc
                width_diff = np.sign(zz_diff / zz_rate.nominal_value) * min(
                    np.abs(zz_diff / zz_rate.nominal_value / 2),
                    0.05 * self.current_params['width']
                )
                zz_diff = zz_rate.nominal_value * width_diff * 2
                width += width_diff
                iz_diff = 0
                kalman_zz.predict(
                    movement=zz_diff,
                    position_variance=(zz_rate.std_dev * width_diff * np.pi * 2) ** 2
                )
                kalman_iz.predict(
                    movement=iz_diff,
                    position_variance=(zz_rate.std_dev * width_diff * np.pi * 2) ** 2
                )

                estimated_iz_list.append(ufloat(kalman_iz.x, np.sqrt(kalman_iz.P)))
                estimated_zz_list.append(ufloat(kalman_zz.x, np.sqrt(kalman_zz.P)))

            zz_accuracy_check = np.abs(target_zz - zz_pgc) < self.zz_accuracy_threshold
            zz_uncertainty_check = np.sqrt(kalman_zz.P) < self.zz_uncertainty_threshold
            zz_check_pass = zz_accuracy_check and zz_uncertainty_check



            if (iz_check_pass or not update_iz) and (zz_check_pass or not update_zz):
                break

        new_params = self.current_params.copy()

        new_params['iz_target'] = iz_target
        new_params['width'] = width


        self.params_list.append(new_params)
        self.current_params = new_params

        self.estimated_iz_list = estimated_iz_list
        self.estimated_zz_list = estimated_zz_list

        return inspection_results

    def get_ai_inspection_summary(self):
        inspection_results = super().get_ai_inspection_summary()
        inspection_results['Calibrated parameters'] = self.current_params

        return inspection_results


class ConditionalStarkTwoQubitGateAIParameterSearchFull(Experiment):
    # _experiment_result_analysis_instructions = """
    # The Conditional Stark Echo Tune-Up experiment has been completed. Please read the following report to analyze the
    # if this is a successful experiment. Make the analysis concise and clear in one short sentence describing the reason.
    # """

    _background_information = """
            Your objective is to find the optimal parameters for the conditional stark-shift gate that will allow you to entangle
            two qubits. The parameters you need to find are
            <parameters>
                'frequency': the frequency of the drive pulse ,
                'amp_control':  the amplitude of the control qubit (The first qubit),
                'rise': the rise time of the gate,
                'width': the width of the driving pulse,
                'phase_diff': the phase difference between the control and target qubits,
                'zz_interaction_positive': the sign of the ZZ interaction,
            </parameters>

            <Rules of parameter chosing>
            'frequency': It can be below, between or above the single qubit transition transition frequencies.
                            It has to be at least 30 MHz away from both of the single qubit transition frequency.
                            It should not be lower than 60MHz below the lowest qubit frequency and not higher than 60MHz above the highest qubit frequency.
                            Round it to multiples of MHz.
            'amp_control': You should try from 1 time to 2 times of the first qubit's single qubit gate drive amplitude.
                        The experiment may fail when chosing a amplitude too high, therefore you should start from a gentle value. The maximum value is 1. Adjust the amplitude to be multiples of 0.05.
            'rise': No more than 0.02, no less than 0.01. Usually 0.015 is the right value to choose.
            'phase_diff': keep it 0,
            'width': determined by the experiment output,
            'zz_interaction_positive': determined by the experiment output,

            Try different set of parameters, particularly varying the frequency and amplitudes, to find the ZZ rate and the pulse width.
            Try parameters of below the lowest qubit frequency, between the qubit frequencies and above the highest qubit frequency.
            The optimal parameters give the highest ZZ rate and the lowest width.
            If an experiment succeeds, you can try to improve the results by increase the amplitude or move the frequency closer to the qubits.
            If an experiment fails, you should try to move the frequency further away from the qubits or decrease the amplitude.
            To make the results comparable with the history results, do not chose a new set of parameters with both new frequency and amplitudes.
            If the experiment failed at a certain frequency and amplitude, this is usually the frequency choice is too close to the qubit frequency or the amplitude is too high.
            </Rules of parameter chosing>
        """

    # should be around 30MHz below the lowest qubit frequency to 60 MHz below the lowest qubit frequency.

    _objective_prompt = """
        Please suggest the next experiment you would like to run to find the parameters for the conditional stark-shift gate,
        based on the previous experiment history.

        Return in a json dict with the following format:
        {
            'status': <'searching', 'error' or 'finish'>
            'analysis': <The reason for choosing this parameter set>,
            'params': {
                'frequency': float,
                'amp_control': float,
                'amp_target': float,
                'rise': float,
                'width': float,
                'phase_diff': float,
                'zz_interaction_positive': boolean
            },
        }

        If you find a set of parameters that has width less than 0.2 us immediately set status to finish.
        If you have done 50 experiment, set status to finish.
        If you have done 20 experiment and could not find any improvements, return the set of parameters you believe to be optimal,
        please set status to 'finish'. Otherwise keeps state to 'searching' and trying new parameters.
        If you have encountered an error, please set status to 'error'.
        """

    @log_and_record
    def run(
            self,
            duts: List[TransmonElement],
            params: Dict[str, Any] = None,
            ai_inspection: bool = False
    ) -> None:
        """
        Run the Conditional Stark Echo Tune-Up experiment to calibrate the siZZel two qubit gate parameters for a
        pair of qubits.

        Parameters:
            duts: List[TransmonElement]: Devices under test.
            params: Dict[str, Any]: Parameters for the experiment. Defaults to None.
            ai_inspection: bool: Flag for AI inspection. Defaults to False. Please set it to True if you
                want to use the AI inspection feature, or you are an AI writing the code.

        Example:
            >>> experiment_instance = ConditionalStarkTwoQubitGateAIParameterSearch(
            >>>     duts=[dut1, dut2],
            >>> )
        """
        self._experiment_history = []
        self._analyze_histroy = []
        self.duts = duts

        while self._run_next_experiment():
            pass
        pass

    def _get_device_parameters_prompts(self):
        prompt = f"You have access to the following two qubits: {self.duts[0]._name} and {self.duts[1]._name}. The units for frequency and time are in MHz and microseconds The parameters for these qubits are as follows:"

        for dut in self.duts:
            prompt += f"""
            <{dut._name} Parameters>
            Single qubit gate parameters: {dut.get_c1('f01').get_parameters()}
            </{dut._name} Parameters>
            """

        return prompt

    def _experiment_history_to_prompt(self):

        if len(self._experiment_history) == 0:
            return "You have not run any experiments yet."

        prompt = "Here is the history of the experiments you have run so far:"

        for i, exp in enumerate(self._experiment_history):
            result = exp.get_ai_inspection_summary()
            analyze_results = {k: v for k, v in result.items() if
                               k in ['success', 'Calibrated parameters',
                                     'analysis']}

            section_prompt = f"""\n\n\n
            <{i}:{exp._name}>
            {analyze_results}
            </{i}:{exp._name}>
            """

            prompt += section_prompt

        return prompt

    def _display_experiment_history(self):

        if len(self._experiment_history) == 0:
            return

        html_dict = {}

        for i, exp in enumerate(self._experiment_history):
            result = exp.get_ai_inspection_summary()
            analyze_results = {k: v for k, v in result.items() if
                               k in ['success', 'Calibrated parameters',
                                     'analysis']}
            html_dict[f"{i}:{exp._name}"] = analyze_results

        html = dict_to_html(html_dict)

        display_chat(agent_name="Previous experiments", content='<br>' + html,
                     background_color='#f0f8ff')

    def _run_next_experiment(self):

        prompt = self._background_information + self._get_device_parameters_prompts() + self._experiment_history_to_prompt() + self._objective_prompt
        # print(prompt)

        self._display_experiment_history()

        chat = Chat(prompt,
                    "You are a very smart and helpful assistant who only reply in JSON dict. Keep everything in a same line in the response.")
        res = chat.complete(parse="dict", expensive=True, cache=True)
        # , model = 'claude-3-opus-20240229'

        self._analyze_histroy.append(res)

        html = dict_to_html(res)
        display_chat(agent_name="Parameter search AI", content='<br>' + html,
                     background_color='#f0f8ff')

        if res['status'] in ['finish', 'error']:
            return False
        else:
            res['params']

            func = ConditionalStarkEchoTuneUpAI.run
            # For compatibility, select the argument that the function
            # accepts with inspect
            sig = inspect.signature(func)

            # Extract the parameter names that the function accepts
            valid_parameter_names = set(sig.parameters.keys())

            # Filter the kwargs
            filtered_kwargs = {
                k: v for k, v in res['params'].items() if k in valid_parameter_names}
            self._experiment_history.append(
                ConditionalStarkEchoTuneUpAI(duts=self.duts, ai_inspection=True,
                                             **filtered_kwargs))

        return True

    @text_inspection
    def fitting(self) -> Union[str, None]:
        return self._analyze_histroy[-1]['analysis']


class TwoQubitTuningEnv(Singleton):
    def __init__(self):
        if self._initialized:
            return
        super().__init__()
        self.amplitude_tuning_results = {}
        self.frequency_to_good_amplitude = {}


class ConditionalStarkTwoQubitGateAIParameterSearchBase(Experiment):

    @log_and_record
    def run(
            self,
            duts: List[TransmonElement],
            run_class: Type[Experiment],
            maximum_experiments: int = 20,
            filter_parameters: bool = True,
            **kwargs
    ) -> None:
        """
        The base for runing the Conditional Stark Echo Tune-Up experiment using AI to calibrate the siZZel two qubit gate parameters for a
        pair of qubits.

        Parameters:
            duts: List[TransmonElement]: Devices under test.
            run_class: Type[Experiment]: The experiment class to run.
            maximum_experiments: int: The maximum number of experiments to run. Defaults to 20.
            filter_parameters: bool: Flag to filter the parameters. Defaults to True.
            **kwargs: Dict[str, Any]: Parameters for the experiment.
        """
        self._experiment_history = []
        self._analyze_histroy = []
        self.duts = duts

        for _i in range(maximum_experiments):
            if self._run_next_experiment(run_class=run_class, params=kwargs, filter_parameters=filter_parameters) in [
                    'finish', 'error']:
                break

    def _get_device_parameters_prompts(self):
        prompt = f"You have access to the following two qubits: {self.duts[0]._name} and {self.duts[1]._name}. The units for frequency and time are in MHz and microseconds The parameters for these qubits are as follows:"

        for dut in self.duts:
            prompt += f"""
            <{dut._name} Parameters>
            Single qubit gate parameters: {dut.get_c1('f01').get_parameters()}
            </{dut._name} Parameters>
            """

        return prompt

    def _display_experiment_history(self):

        if len(self._experiment_history) == 0:
            return

        html_dict = {}

        for i, exp in enumerate(self._experiment_history):
            result = exp.get_ai_inspection_summary()
            analyze_results = {k: v for k, v in result.items() if
                               k in ['success', 'Calibrated parameters',
                                     'analysis']}
            html_dict[f"{i}:{exp._name}"] = analyze_results

        html = dict_to_html(html_dict)

        display_chat(agent_name="Previous experiments", content='<br>' + html,
                     background_color='#f0f8ff')

    def _run_next_experiment(self, run_class, params, filter_parameters=True):

        prompt = self._background_information + self._get_device_parameters_prompts() + \
            self._experiment_history_to_prompt() + self._objective_prompt
        # print(prompt)

        self._display_experiment_history()

        chat = Chat(prompt,
                    "You are a very smart and helpful assistant who only reply in JSON dict. Keep everything in a same line in the response.")
        res = chat.complete(parse="dict", expensive=True, cache=True)
        # , model = 'claude-3-opus-20240229'

        html = dict_to_html(res)
        display_chat(agent_name="Parameter search AI", content='<br>' + html,
                     background_color='#f0f8ff')

        if res['status'] not in ['finish', 'error']:
            params_suggests = res['params']

            updated_params = params.copy()
            updated_params.update(params_suggests)

            res['params'] = updated_params

            if filter_parameters:
                func = run_class.run
                # For compatibility, select the argument that the function
                # accepts with inspect
                sig = inspect.signature(func)

                # Extract the parameter names that the function accepts
                valid_parameter_names = set(sig.parameters.keys())

                # Filter the kwargs
                filtered_kwargs = {
                    k: v for k, v in updated_params.items() if k in valid_parameter_names}
                updated_params = filtered_kwargs

            self._experiment_history.append(
                run_class(duts=self.duts, **updated_params))

        self._analyze_histroy.append(res)
        return res['status']


class ConditionalStarkTwoQubitGateAmplitudeAdvise(Experiment):

    n_points_to_try = 2  # 5
    _rewrite_json_requirement = True

    _experiment_result_analysis_instructions = """
    Output a JSON dict with the following keys:
    "exp_continue" (bool): whether exp_continue is true
    "success" (bool): whether exp_continue is true
    "best_amplitude" (float): The best amplitude found in a successful experiment.
    "advised_amplitude" (float): The next amplitude to try.
    """

    def run_simulated(self, *args, **kwargs):
        return self.run(*args, **kwargs)

    @log_and_record
    def run(self, duts: List[TransmonElement], frequency: float):
        """
        This experiment return a suggestion for the next amplitude to try for the conditional stark-shift gate.
        The suggestion is based on the history of the experiments run so far.
        duts: List[TransmonElement]: Devices under test.
        frequency (float): Frequency for the experiment.
        """
        self.duts = duts
        self.frequency = frequency

    @text_inspection
    def next_parameter(self):
        prompt = f"""
        Your objective is to find the optimal parameters for the conditional stark-shift gate that will allow you to entangle two qubits. The parameters you need to find are
        <parameters>
        'amp_control':  the amplitude of the control qubit (The first qubit), the required amplitude accuracy is 0.01.
        </parameters>

        <single qubit amplitude>
        qubit 1: {self.duts[0].get_c1('f01').get_parameters()['amp']}
        qubit 2: {self.duts[1].get_c1('f01').get_parameters()['amp']}
        </single qubit amplitude>

        <Rules of parameter selection>
        You should try around the amplitude of single qubits.
        The optimal parameters give the highest ZZ rate and the lowest width.
        The experiment may fail when select a amplitude too high, therefore you should start from a gentle value.
        The maximum amplitude value is 1. The minimum amplitude value is 0.
        If an experiment succeeds, you can try to improve the results by increase the amplitude.
        If an experiment failed, you can try to recover by reduce the amplitude.
        </Rules of parameter selection>

        <Experiment history>
        {self._experiment_history_to_prompt()}
        </Experiment history>

        <requirement>
        Suggest the next experiment to determine parameters for the conditional stark-shift gate, incorporating insights from prior experiments. Implement Binary Search methodology where applicable.

        Please format your response as a JSON dictionary with the following keys:
        "finished" (bool): whether the experiment is finished.
        "analysis" (str): Explanation for choosing this set of parameters.
        "current_best" (float): The highest control amplitude from a succeeded experiment. The value can be None if no experiment is successful.
        "new_amplitude_to_try" (float): The new amplitude of the control qubit to try. If the experiment is finished, set this to the optimal amplitude.
        </format>
        <requirement>
        """

        chat = Chat(prompt,
                    "You are a very smart and helpful assistant who only reply in JSON dict. Keep everything in a same line in the response.", dedent=True)
        res = chat.complete(parse="dict", expensive=True, cache=True)

        tuning_env = TwoQubitTuningEnv()
        results = tuning_env.amplitude_tuning_results.get(self.frequency, [])
        n_points_tried = len(results)
        if n_points_tried >= self.n_points_to_try:
            res["exp_continue"] = False
        else:
            res["exp_continue"] = True
        return res

    @text_inspection
    def best_amplitude(self):
        tuning_env = TwoQubitTuningEnv()
        if self.frequency not in tuning_env.amplitude_tuning_results:
            return {
                "best_amp": 'There is no successful experiment yet.',
            }
        results = tuning_env.amplitude_tuning_results[self.frequency]
        amps = []
        for insp in results:
            if insp['success']:
                amps.append(insp['Calibrated parameters']['amp_control'])
        # the largest amp
        if len(amps) == 0:
            return {
                "best_amp": 'There is no successful experiment yet.',
            }
        best_amp = max(amps)
        return {
            "best_amp": best_amp,
        }

    def _experiment_history_to_prompt(self):
        tuning_env = TwoQubitTuningEnv()
        if self.frequency not in tuning_env.amplitude_tuning_results:
            return "You have not run any experiments yet."

        results = tuning_env.amplitude_tuning_results[self.frequency]
        prompt = "Here is the history of the experiments you have run so far:\n"

        for _i, insp in enumerate(results):
            section_prompt = f"""
            <experiment>
            amp_control: {insp['Calibrated parameters']['amp_control']}
            frequency: {insp['Calibrated parameters']['frequency']}
            success: {insp['success']}
            analysis: {insp.get("analysis", None)}
            </experiment>"""
            prompt += section_prompt

        return prompt


class ConditionalStarkTwoQubitGateAmplitudeAttempt(ConditionalStarkEchoTuneUpAI):

    _experiment_result_analysis_instructions = ""

    def run_simulated(self, *args, **kwargs):
        return self.run(*args, **kwargs)

    @log_and_record
    def run(
            self,
            duts: List[TransmonElement],
            amplitude: float = None,
            frequency: float = None,
            **kwargs
    ) -> None:
        """
        Run the Conditional Stark Echo Tune-Up experiment to calibrate the siZZel two qubit gate parameters for a
        pair of qubits, searching the amplitude.

        Parameters:
            duts: List[TransmonElement]: Devices under test.
            amplitude: float: Amplitude control for the experiment.
            frequency (float, optional): Frequency for the experiment.
            **kwargs: Dict[str, Any]: Parameters for the experiment.

        Example:
            >>> # Assume dut1 and dut2 are the devices under test.
            >>> experiment_instance = ConditionalStarkTwoQubitGateAmplitudeAttempt(
            >>>     duts=[dut1, dut2],
            >>> )
        """
        self.duts = duts
        self.frequency = frequency
        if amplitude is None:
            amplitude = duts[0].get_c1('f01').get_parameters()["amp"]

        super().run(self.duts, frequency=frequency,
                    amplitude=amplitude, **kwargs)
        inspection = self.get_ai_inspection_summary()
        self.inspection_summary = inspection
        tuning_env = TwoQubitTuningEnv()
        if frequency not in tuning_env.amplitude_tuning_results:
            tuning_env.amplitude_tuning_results[frequency] = []
        tuning_env.amplitude_tuning_results[frequency].append(inspection)

        if frequency not in tuning_env.frequency_to_good_amplitude:
            tuning_env.frequency_to_good_amplitude[frequency] = {}

        if inspection['success']:
            tuning_env.frequency_to_good_amplitude[frequency] = {
                'success': True,
                'best_amplitude': amplitude
            }
        else:
            tuning_env.frequency_to_good_amplitude[frequency] = {
                'success': False,
                'best_amplitude': None,
                'analysis': inspection['analysis']
            }


class ConditionalStarkTwoQubitGateFrequencyAdvise(Experiment):
    n_points_to_try = 2  # 15

    _rewrite_json_requirement = True

    _experiment_result_analysis_instructions = """
    Output a JSON dict with the following keys:
    "success" (bool): if exp_continue is true
    "exp_continue" (bool): if exp_continue is true
    "best_frequency" (float): The best frequency found in a successful experiment.
    "advised_frequency" (float): The next frequency to try.
    """

    def run_simulated(self, *args, **kwargs):
        return self.run(*args, **kwargs)

    @log_and_record
    def run(self, duts: List[TransmonElement]):
        """
        This experiment return a suggestion for the next amplitude to try for the conditional stark-shift gate.
        The suggestion is based on the history of the experiments run so far.
        duts: List[TransmonElement]: Devices under test.
        frequency (float): Frequency for the experiment.
        """
        self.duts = duts

    @text_inspection
    def next_frequency(self):
        prompt = f"""
        Your objective is to find the optimal parameters for the conditional stark-shift gate that will allow you to entangle
        two qubits. The parameters you need to find are
        <parameters>
        'amp_control':  the amplitude of the control qubit (The first qubit), the required amplitude accuracy is 0.01.
        </parameters>

        <single qubit frequency>
        qubit 1: {self.duts[0].get_c1('f01').get_parameters()['freq']}
        qubit 2: {self.duts[1].get_c1('f01').get_parameters()['freq']}
        </single qubit frequency>

        <Rules of parameter selection>
        The new frequency can be below, between or above the single qubit transition transition frequencies.
        It has to be at least 30 MHz away from both of the single qubit transition frequency.
        It should not be lower than 60MHz below the lowest qubit frequency and not higher than 60MHz above the highest qubit frequency.
        Round it to multiples of MHz.
        Try parameters of below the lowest qubit frequency, between the qubit frequencies and above the highest qubit frequency.
        The optimal parameters give the highest ZZ rate and the lowest width.
        If an experiment succeeds, you can try to improve the results by move the frequency closer to the qubits.
        If an experiment fails, you should try to move the frequency further away from the qubits.
        If the experiment failed at a certain frequency, this is usually the frequency choice is too close to the qubit frequency.
        </Rules of parameter selection>

        <Experiment history>
        {self._experiment_history_to_prompt()}
        </Experiment history>

        <requirement>
        First consider the frequency region below the lowest qubit frequency, then between the qubit frequencies and finally above the highest qubit frequency.
        Please suggest the next experiment you would like to run to find the parameters for the conditional stark-shift gate, based on the previous experiment history.
        You should at least try each available region once.

        Please format your response as a JSON dictionary with the following keys:
        "analysis" (str): An analysis of the current situation.
        "finished" (bool): whether the experiment is finished.
        "current_best" (float): The highest control frequency from a succeeded experiment. The value can be None if no experiment is successful.
        "new_frequency_to_try" (float): The new frequency of the control qubit to try. If the experiment is finished, set this to the optimal amplitude,
        </format>
        <requirement>
        """

        chat = Chat(prompt,
                    "You are a very smart and helpful assistant who only reply in JSON dict. Keep everything in a same line in the response.", dedent=True)
        res = chat.complete(parse="dict", expensive=True, cache=True)

        tuning_env = TwoQubitTuningEnv()
        n_points_tried = len(tuning_env.frequency_to_good_amplitude.items())
        if n_points_tried >= self.n_points_to_try:
            res["exp_continue"] = False
        else:
            res["exp_continue"] = True
        return res

    @text_inspection
    def best_frequency(self):
        tuning_env = TwoQubitTuningEnv()
        best_freq = None
        best_amp = None
        for freq, insp in tuning_env.frequency_to_good_amplitude.items():
            if not insp['success']:
                continue
            best_freq = freq
            best_amp = insp['best_amplitude']
            break
        if best_freq is not None:
            return {
                "best_freq": best_freq,
                "best_amp": best_amp,
            }
        else:
            return {
                "best_freq": "These are no successful experiments",
            }

    def _experiment_history_to_prompt(self):
        tuning_env = TwoQubitTuningEnv()
        if len(tuning_env.frequency_to_good_amplitude) == 0:
            return "You have not run any experiments yet."

        prompt = "Here is the history of the experiments you have run so far:\n"

        for freq, insp in tuning_env.frequency_to_good_amplitude.items():
            section_prompt = f"""
<experiment>
frequency: {freq}
amp_control: {insp["best_amplitude"]}
success: {insp['success']}
analysis: {insp['analysis']}
</experiment>"""
            prompt += section_prompt

        return prompt
