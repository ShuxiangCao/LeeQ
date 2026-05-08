from .common import *
from .rabi_xy import ConditionalStarkTuneUpRabiXY
from .repeated_gate import ConditionalStarkTuneUpRepeatedGateXY

class ConditionalStarkEchoTuneUp(Experiment):
    @log_and_record
    def run(self, duts, params=None, frequency=None, amp_control=None, phase_diff=0, rise=0.01, trunc=1.0,
            t_start=0, t_stop=20, sweep_points=40,
            n_start=0, n_stop=21, update_iz=False, update_zz=True, n_max_iteration=20
            ):
        """
        Execute the experiment on hardware.

        Parameters
        ----------
        duts : list[TransmonElement]
            List of two qubits [control, target].
        params : dict, optional
            Initial gate parameters. Default: None (auto-calculated)
        frequency : float, optional
            Stark drive frequency (MHz). Default: None
        amp_control : float, optional
            Control qubit amplitude. Default: None
        phase_diff : float, optional
            Phase difference between drives. Default: 0
        rise : float, optional
            Pulse rise time. Default: 0.01
        trunc : float, optional
            Pulse truncation. Default: 1.0
        t_start : float, optional
            Start time for sweep. Default: 0
        t_stop : float, optional
            Stop time for sweep. Default: 20
        sweep_points : int, optional
            Number of sweep points. Default: 40
        n_start : int, optional
            Start echo count. Default: 0
        n_stop : int, optional
            Stop echo count. Default: 21
        update_iz : bool, optional
            Update single-qubit Z rates. Default: False
        update_zz : bool, optional
            Update ZZ interaction rate. Default: True
        n_max_iteration : int, optional
            Maximum iterations. Default: 20

        Returns
        -------
        None
            Results are stored in instance attributes.
        """
        self.duts = duts
        self.n_max_iteration = n_max_iteration

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
                'amp_control': amp_control,
                'amp_target': amp_control * area_target / area_control,
                'rise': rise,
                'trunc': trunc,
                'width': 0,
                'phase_diff': phase_diff,
                'zz_interaction_positive': True,
                'echo': True
            }

        # Creating a dataframe
        df = pd.DataFrame(list(params.items()), columns=['Parameter', 'Value'])

        # Formatting the float values to three decimal places
        df['Value'] = df['Value'].apply(lambda x: f"{x:.3f}" if isinstance(x, float) else x)

        # Display the dataframe
        display(df.style.set_properties(**{'text-align': 'center'}).set_table_styles([{
            'selector': 'th',
            'props': [('text-align', 'center')]
        }]))

        self.current_params = params
        self.params_list = [params]

        iz_rate, zz_rate = self.run_sizzel_xy_hamiltonian_tomography(t_start=t_start, t_stop=t_stop,
                                                                     sweep_points=sweep_points)

        self.current_params['zz_interaction_positive'] = zz_rate.nominal_value > 0

        self.run_repeated_gate_hamiltonian_tomography(duts=self.duts, zz_rate=zz_rate, n_start=n_start, n_stop=n_stop,
                                                      update_iz=False, update_zz=True)

    def run_sizzel_xy_hamiltonian_tomography(self, t_start, t_stop, sweep_points=60):

        t_step = (t_stop - t_start) / sweep_points

        sizzel_xy = ConditionalStarkTuneUpRabiXY(
            qubits=self.duts,
            frequency=self.current_params['frequency'],
            amp_control=self.current_params['amp_control'],
            amp_target=self.current_params['amp_target'],
            rise=self.current_params['rise'],
            trunc=self.current_params['trunc'],
            start=t_start,
            stop=t_stop,
            step=t_step,
            phase_diff=self.current_params['phase_diff'],
            iz_rate_cancel=0,
            iz_rise_drop=0,
            echo=True)

        result = sizzel_xy.analyze_results_with_errs()

        iz_rate = result['iz_rate']
        zz_rate = result['zz_rate']

        new_params = self.current_params.copy()
        new_params['width'] = np.abs(0.125 / zz_rate.nominal_value) / 2


        self.params_list.append(new_params)
        self.current_params = new_params

        return iz_rate, zz_rate

    def run_repeated_gate_hamiltonian_tomography(self, duts, zz_rate, n_start=0, n_stop=32, update_iz=False,
                                                 update_zz=True):

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
            repeated_gate = ConditionalStarkTuneUpRepeatedGateXY(
                duts=self.duts,
                iz_control=0,
                iz_target=iz_target,
                frequency=self.current_params['frequency'],
                amp_control=self.current_params['amp_control'],
                amp_target=self.current_params['amp_target'],
                rise=self.current_params['rise'],
                trunc=self.current_params['trunc'],
                width=width,
                start_gate_number=n_start,
                gate_count=n_stop,
                echo=True,
            )

            iz_target_measured = iz_target + repeated_gate.iz_rate.nominal_value * np.pi * 2
            zz_measured = repeated_gate.zz_rate.nominal_value

            measured_iz_list.append(iz_target_measured)
            measured_zz_list.append(zz_measured)

            if kalman_iz is None:
                kalman_iz = KalmanFilter1D(initial_position=iz_target_measured,
                                           position_variance=(repeated_gate.iz_rate.std_dev * np.pi * 2) ** 2)
                kalman_zz = KalmanFilter1D(initial_position=zz_measured,
                                           position_variance=(repeated_gate.zz_rate.std_dev * np.pi * 2) ** 2)
            else:
                kalman_iz.update(measurement=iz_target_measured,
                                 measurement_variance=(repeated_gate.iz_rate.std_dev * np.pi * 2) ** 2)

                kalman_zz.update(measurement=zz_measured,
                                 measurement_variance=(repeated_gate.zz_rate.std_dev * np.pi * 2) ** 2)


            if update_iz:
                iz_target = kalman_iz.x
                iz_check_pass = kalman_iz.P < 1e-2
                if not update_zz:
                    estimated_iz_list.append(unc.ufloat(kalman_iz.x, np.sqrt(kalman_iz.P)))

            if update_zz:
                zz_pgc = kalman_zz.x

                target_zz = np.sign(zz_pgc) * 0.125
                zz_diff = target_zz - zz_pgc
                width_diff = np.sign(zz_diff / zz_rate.nominal_value) * min(np.abs(zz_diff / zz_rate.nominal_value / 2),
                                                                            0.05 * self.current_params['width'])
                zz_diff = zz_rate.nominal_value * width_diff * 2
                width += width_diff
                iz_diff = 0
                # iz_rate_tQ1_cQ2 * width_diff * np.pi * 2
                kalman_zz.predict(movement=zz_diff,
                                  position_variance=(zz_rate.std_dev * width_diff * np.pi * 2) ** 2)
                kalman_iz.predict(movement=iz_diff,
                                  position_variance=(zz_rate.std_dev * width_diff * np.pi * 2) ** 2)

                estimated_iz_list.append(unc.ufloat(kalman_iz.x, np.sqrt(kalman_iz.P)))
                estimated_zz_list.append(unc.ufloat(kalman_zz.x, np.sqrt(kalman_zz.P)))

            zz_accuracy_check = np.abs(target_zz - zz_pgc) < 1e-3
            zz_uncertainty_check = np.sqrt(kalman_zz.P) < 1e-3
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
