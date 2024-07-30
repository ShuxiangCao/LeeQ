from plotly.subplots import make_subplots

from labchronicle import log_and_record, register_browser_function
from leeq import Experiment
from plotly import graph_objects as go
from typing import List, Any
import numpy as np

from leeq.utils.compatibility import *


class ConditionalStarkFineFrequencyTuneUp(Experiment):
    @log_and_record
    def run(self, duts, params=None, phase_diff=0, amp_control=0.2, rise=0.0, trunc=1.0,
            t_start=0, t_stop=20, sweep_points=30,
            frequency_start: float = 4800, frequency_stop: float = 4900, frequency_step: float = 10,
            n_start=0, n_stop=32, update_iz=False, update_zz=True
            ):
        self.duts = duts

        assert update_iz == False

        if params is None:
            amp_rabi_control = duts[0].get_c1('f01')['X'].amp
            amp_rabi_target = duts[1].get_c1('f01')['X'].amp

            area_control = amp_rabi_control * duts[0].get_c1('f01')['X'].width
            area_target = amp_rabi_target * duts[1].get_c1('f01')['X'].width

            params = {
                'iz_control': 0,
                'iz_target': 0,
                'frequency': frequency_start,
                'amp_control': amp_control,
                'amp_target': amp_control * area_target / area_control,
                'rise': rise,
                'trunc': trunc,
                'width': 0,
                'phase_diff': phase_diff,
                'zz_interaction_positive': True,
                'echo': True
            }

        self.current_params = params
        self.params_list = [params]
        self.results = []

        for frequency in np.arange(frequency_start, frequency_stop, frequency_step):
            print(f"frequency: {frequency:.3f} MHz")
            self.current_params['frequency'] = frequency
            iz_rate, zz_rate, width, result, result_target, result_control = self.run_sizzel_xy_hamiltonian_tomography(
                t_start=t_start, t_stop=t_stop, sweep_points=sweep_points
            )

            self.results.append({
                'frequency': frequency,
                'iz_rate': iz_rate,
                'zz_rate': zz_rate,
                'width': width,
                'result': result,
                'result_target': result_target,
                'result_control': result_control,
                't_start': t_start,
                't_stop': t_stop,
                'sweep_points': sweep_points
            })

            self.current_params['zz_interaction_positive'] = zz_rate.nominal_value > 0

    def run_sizzel_xy_hamiltonian_tomography(self, t_start, t_stop, sweep_points=60):
        t_step = (t_stop - t_start) / sweep_points

        from leeq.experiments.builtin.multi_qubit_gates.sizzel import ConditionalStarkTuneUpRabiXY
        sizzel_xy = ConditionalStarkTuneUpRabiXY(
            qubits=self.duts,
            frequency=self.current_params['frequency'],
            amp_control=self.current_params['amp_control'],
            amp_target=self.current_params['amp_target'],
            rise=self.current_params['rise'],
            start=t_start,
            stop=t_stop,
            step=t_step,
            phase_diff=self.current_params['phase_diff'],
            iz_rate_cancel=0,
            iz_rise_drop=0,
            echo=True
        )

        result = sizzel_xy.analyze_results_with_errs()
        iz_rate = result['iz_rate']
        zz_rate = result['zz_rate']

        new_params = self.current_params.copy()
        new_params['width'] = np.abs(0.125 / zz_rate.nominal_value) / 2
        width = new_params['width']

        print(f'Estimated IZ = {iz_rate} MHz, ZZ = {zz_rate} MHz, width = {new_params["width"]: 0.5f} us')

        self.params_list.append(new_params)
        self.current_params = new_params

        self.result_target = sizzel_xy.result  # Store the result of the current iteration
        self.result_control = sizzel_xy.result_control  # Assuming result_control is accessible from sizzel_xy

        return iz_rate, zz_rate, width, result, self.result_target, self.result_control

    @register_browser_function(available_after=(run,))
    def plot_results(self, plot_size_3d=(1000, 2000), fig_size=(600, 2000)):
        # Colors for the plot
        dark_navy = '#000080'
        dark_purple = '#800080'
        light_black = '#D3D3D3'

        # Prepare data for 3D plots
        frequencies = [res['frequency'] for res in self.results]
        times = [np.linspace(res['t_start'], res['t_stop'], res['sweep_points']) for res in self.results]

        # Extract the necessary data for plotting
        results_ground_x = [np.array(res['result_target'])[:, 0, 0] for res in self.results]
        results_excited_x = [np.array(res['result_target'])[:, 1, 0] for res in self.results]

        results_ground_y = [np.array(res['result_target'])[:, 0, 1] for res in self.results]
        results_excited_y = [np.array(res['result_target'])[:, 1, 1] for res in self.results]

        results_control_ground_x = [np.array(res['result_control'])[:, 0, 0] for res in self.results]
        results_control_excited_x = [np.array(res['result_control'])[:, 1, 0] for res in self.results]

        results_control_ground_y = [np.array(res['result_control'])[:, 0, 1] for res in self.results]
        results_control_excited_y = [np.array(res['result_control'])[:, 1, 1] for res in self.results]

        # Create the figure with 3D plots for x data
        fig_3d_x = make_subplots(rows=1, cols=2,
                                 subplot_titles=("Target Qubit - X axis", "Control Qubits - X axis"),
                                 specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]])

        # Plot 3D frequency vs time vs result X for ground and excited states
        for i, frequency in enumerate(frequencies):
            fig_3d_x.add_trace(
                go.Scatter3d(x=[frequency] * len(times[i]), y=times[i], z=results_ground_x[i], mode='lines',
                             name='Ground X', showlegend=(i == 0), line=dict(color=dark_navy)),
                row=1, col=1)
            fig_3d_x.add_trace(
                go.Scatter3d(x=[frequency] * len(times[i]), y=times[i], z=results_excited_x[i], mode='lines',
                             name='Excited X', showlegend=(i == 0), line=dict(color=dark_purple)),
                row=1, col=1)

        # Plot 3D frequency vs time vs result control X for ground and excited states
        for i, frequency in enumerate(frequencies):
            fig_3d_x.add_trace(
                go.Scatter3d(x=[frequency] * len(times[i]), y=times[i], z=results_control_ground_x[i], mode='lines',
                             name='', showlegend=(i == 0), line=dict(color=dark_navy)),
                row=1, col=2)
            fig_3d_x.add_trace(
                go.Scatter3d(x=[frequency] * len(times[i]), y=times[i], z=results_control_excited_x[i], mode='lines',
                             name='', showlegend=(i == 0), line=dict(color=dark_purple)),
                row=1, col=2)

        # Update layout for 3D X figure
        fig_3d_x.update_layout(
            height=plot_size_3d[0], width=plot_size_3d[1],
            scene=dict(
                xaxis=dict(title='Frequency (MHz)', backgroundcolor='white', gridcolor=light_black),
                yaxis=dict(title='Time', backgroundcolor='white', gridcolor=light_black),
                zaxis=dict(title='Result Target X', backgroundcolor='white', gridcolor=light_black)
            ),
            scene2=dict(
                xaxis=dict(title='Frequency (MHz)', backgroundcolor='white', gridcolor=light_black),
                yaxis=dict(title='Time', backgroundcolor='white', gridcolor=light_black),
                zaxis=dict(title='Result Control X', backgroundcolor='white', gridcolor=light_black)
            )
        )

        # Create the figure with 3D plots for y data
        fig_3d_y = make_subplots(rows=1, cols=2,
                                 subplot_titles=("Target Qubit - Y axis", "Control Qubits - Y axis"),
                                 specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]])

        # Plot 3D frequency vs time vs result Y for ground and excited states
        for i, frequency in enumerate(frequencies):
            fig_3d_y.add_trace(
                go.Scatter3d(x=[frequency] * len(times[i]), y=times[i], z=results_ground_y[i], mode='lines',
                             name='Ground Y', showlegend=(i == 0), line=dict(color=dark_navy)),
                row=1, col=1)
            fig_3d_y.add_trace(
                go.Scatter3d(x=[frequency] * len(times[i]), y=times[i], z=results_excited_y[i], mode='lines',
                             name='Excited Y', showlegend=(i == 0), line=dict(color=dark_purple)),
                row=1, col=1)

        # Plot 3D frequency vs time vs result control Y for ground and excited states
        for i, frequency in enumerate(frequencies):
            fig_3d_y.add_trace(
                go.Scatter3d(x=[frequency] * len(times[i]), y=times[i], z=results_control_ground_y[i], mode='lines',
                             name='', showlegend=(i == 0), line=dict(color=dark_navy)),
                row=1, col=2)
            fig_3d_y.add_trace(
                go.Scatter3d(x=[frequency] * len(times[i]), y=times[i], z=results_control_excited_y[i], mode='lines',
                             name='', showlegend=(i == 0), line=dict(color=dark_purple)),
                row=1, col=2)

        # Update layout for 3D Y figure
        fig_3d_y.update_layout(
            height=plot_size_3d[0], width=plot_size_3d[1],
            scene=dict(
                xaxis=dict(title='Frequency (MHz)', backgroundcolor='white', gridcolor=light_black),
                yaxis=dict(title='Time', backgroundcolor='white', gridcolor=light_black),
                zaxis=dict(title='Result Target Y', backgroundcolor='white', gridcolor=light_black)
            ),
            scene2=dict(
                xaxis=dict(title='Frequency (MHz)', backgroundcolor='white', gridcolor=light_black),
                yaxis=dict(title='Time', backgroundcolor='white', gridcolor=light_black),
                zaxis=dict(title='Result Control Y', backgroundcolor='white', gridcolor=light_black)
            )
        )

        # Extract nominal values and uncertainties
        self.iz_rates = [res['iz_rate'].nominal_value for res in self.results]
        self.iz_uncertainties = [res['iz_rate'].std_dev for res in self.results]
        self.zz_rates = [res['zz_rate'].nominal_value for res in self.results]
        self.zz_uncertainties = [res['zz_rate'].std_dev for res in self.results]
        self.widths = [res['width'] for res in self.results]
        self.frequencies = [res['frequency'] for res in self.results]

        # Create the figure with 2D plots
        fig_2d = make_subplots(
            rows=1, cols=3,
            subplot_titles=("ZZ vs Frequency", "IZ vs Frequency", "Width vs Frequency"),
            specs=[[{'type': 'scatter'}, {'type': 'scatter'}, {'type': 'scatter'}]]
        )

        # Update layout for 2D figure
        fig_2d.update_layout(
            height=fig_size[0], width=fig_size[1],
            plot_bgcolor='white',
            xaxis=dict(gridcolor=light_black),
            yaxis=dict(gridcolor=light_black)
        )

        # Plot 2D zz vs frequency
        fig_2d.add_trace(
            go.Scatter(
                x=self.frequencies, y=self.zz_rates, mode='lines+markers', name='ZZ vs Frequency',
                line=dict(color=dark_navy),
                error_y=dict(type='data', array=self.zz_uncertainties, visible=True)
            ),
            row=1, col=1
        )

        # Plot 2D iz vs frequency
        fig_2d.add_trace(
            go.Scatter(
                x=self.frequencies, y=self.iz_rates, mode='lines+markers', name='IZ vs Frequency',
                line=dict(color='gray'),
                error_y=dict(type='data', array=self.iz_uncertainties, visible=True)
            ),
            row=1, col=2
        )

        # Plot 2D width vs frequency
        fig_2d.add_trace(
            go.Scatter(
                x=self.frequencies, y=self.widths, mode='lines+markers', name='Width vs Frequency',
                line=dict(color=dark_purple)
            ),
            row=1, col=3
        )

        # Customize 2D plots
        fig_2d.update_xaxes(title_text="Frequency (MHz)", row=1, col=1)
        fig_2d.update_yaxes(title_text="ZZ Rate (MHz)", row=1, col=1)

        fig_2d.update_xaxes(title_text="Frequency (MHz)", row=1, col=2)
        fig_2d.update_yaxes(title_text="IZ Rate (MHz)", row=1, col=2)

        fig_2d.update_xaxes(title_text="Frequency (MHz)", row=1, col=3)
        fig_2d.update_yaxes(title_text="Width (us)", row=1, col=3)

        return fig_2d.show(), fig_3d_x.show(), fig_3d_y.show()


class ConditionalStarkFineAmpTuneUp(Experiment):
    @log_and_record
    def run(self, duts, params=None, frequency=None, phase_diff=0, rise=0.0, trunc=1.0,
            t_start=0, t_stop=20, sweep_points=30, amp_control=0.2,
            amp_control_start: float = 0.1, amp_control_stop: float = 1.0, amp_control_step: float = 0.1,
            n_start=0, n_stop=32, update_iz=False, update_zz=True
            ):
        self.duts = duts

        assert update_iz == False

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

        self.current_params = params
        self.params_list = [params]
        self.results = []

        for amp_control in np.arange(amp_control_start, amp_control_stop, amp_control_step):
            print(f"amp_control: {amp_control:.3f}")
            self.current_params['amp_control'] = amp_control
            iz_rate, zz_rate, width, result, result_target, result_control = self.run_sizzel_xy_hamiltonian_tomography(
                t_start=t_start, t_stop=t_stop, sweep_points=sweep_points
            )

            self.results.append({
                'amp_control': amp_control,
                'iz_rate': iz_rate,
                'zz_rate': zz_rate,
                'width': width,
                'result': result,
                'result_target': result_target,
                'result_control': result_control,
                't_start': t_start,
                't_stop': t_stop,
                'sweep_points': sweep_points
            })

            self.current_params['zz_interaction_positive'] = zz_rate.nominal_value > 0

    def run_sizzel_xy_hamiltonian_tomography(self, t_start, t_stop, sweep_points=60):
        t_step = (t_stop - t_start) / sweep_points

        from leeq.experiments.builtin.multi_qubit_gates.sizzel import ConditionalStarkTuneUpRabiXY
        sizzel_xy = ConditionalStarkTuneUpRabiXY(
            qubits=self.duts,
            frequency=self.current_params['frequency'],
            amp_control=self.current_params['amp_control'],
            amp_target=self.current_params['amp_target'],
            rise=self.current_params['rise'],
            start=t_start,
            stop=t_stop,
            step=t_step,
            phase_diff=self.current_params['phase_diff'],
            iz_rate_cancel=0,
            iz_rise_drop=0,
            echo=True
        )

        result = sizzel_xy.analyze_results_with_errs()
        iz_rate = result['iz_rate']
        zz_rate = result['zz_rate']

        new_params = self.current_params.copy()
        new_params['width'] = np.abs(0.125 / zz_rate.nominal_value) / 2
        width = new_params['width']

        print(f'Estimated IZ = {iz_rate} MHz, ZZ = {zz_rate} MHz, width = {new_params["width"]: 0.5f} us')

        self.params_list.append(new_params)
        self.current_params = new_params

        self.result_target = sizzel_xy.result  # Store the result of the current iteration
        self.result_control = sizzel_xy.result_control  # Assuming result_control is accessible from sizzel_xy

        return iz_rate, zz_rate, width, result, self.result_target, self.result_control

    @register_browser_function(available_after=(run,))
    def plot_results(self, plot_size_3d=(1000, 2000), fig_size=(600, 2000)):
        # Colors for the plot
        dark_navy = '#000080'
        dark_purple = '#800080'
        light_black = '#D3D3D3'

        # Prepare data for 3D plots
        amp_controls = [res['amp_control'] for res in self.results]
        times = [np.linspace(res['t_start'], res['t_stop'], res['sweep_points']) for res in self.results]

        # Extract the necessary data for plotting
        results_ground_x = [np.array(res['result_target'])[:, 0, 0] for res in self.results]
        results_excited_x = [np.array(res['result_target'])[:, 1, 0] for res in self.results]

        results_ground_y = [np.array(res['result_target'])[:, 0, 1] for res in self.results]
        results_excited_y = [np.array(res['result_target'])[:, 1, 1] for res in self.results]

        results_control_ground_x = [np.array(res['result_control'])[:, 0, 0] for res in self.results]
        results_control_excited_x = [np.array(res['result_control'])[:, 1, 0] for res in self.results]

        results_control_ground_y = [np.array(res['result_control'])[:, 0, 1] for res in self.results]
        results_control_excited_y = [np.array(res['result_control'])[:, 1, 1] for res in self.results]

        # Create the figure with 3D plots for x data
        fig_3d_x = make_subplots(rows=1, cols=2,
                                 subplot_titles=("Target Qubit - X axis", "Control Qubits - X axis"),
                                 specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]])

        # Plot 3D amp_control vs time vs result X for ground and excited states
        for i, amp_control in enumerate(amp_controls):
            fig_3d_x.add_trace(
                go.Scatter3d(x=[amp_control] * len(times[i]), y=times[i], z=results_ground_x[i], mode='lines',
                             name='Ground X', showlegend=(i == 0), line=dict(color=dark_navy)),
                row=1, col=1)
            fig_3d_x.add_trace(
                go.Scatter3d(x=[amp_control] * len(times[i]), y=times[i], z=results_excited_x[i], mode='lines',
                             name='Excited X', showlegend=(i == 0), line=dict(color=dark_purple)),
                row=1, col=1)

        # Plot 3D amp_control vs time vs result control X for ground and excited states
        for i, amp_control in enumerate(amp_controls):
            fig_3d_x.add_trace(
                go.Scatter3d(x=[amp_control] * len(times[i]), y=times[i], z=results_control_ground_x[i],
                             mode='lines',
                             name='', showlegend=(i == 0), line=dict(color=dark_navy)),
                row=1, col=2)
            fig_3d_x.add_trace(
                go.Scatter3d(x=[amp_control] * len(times[i]), y=times[i], z=results_control_excited_x[i],
                             mode='lines',
                             name='', showlegend=(i == 0), line=dict(color=dark_purple)),
                row=1, col=2)

        # Update layout for 3D X figure
        fig_3d_x.update_layout(
            height=plot_size_3d[0], width=plot_size_3d[1],
            scene=dict(
                xaxis=dict(title='Amp Control', backgroundcolor='white', gridcolor=light_black),
                yaxis=dict(title='Time', backgroundcolor='white', gridcolor=light_black),
                zaxis=dict(title='Result Target X', backgroundcolor='white', gridcolor=light_black)
            ),
            scene2=dict(
                xaxis=dict(title='Amp Control', backgroundcolor='white', gridcolor=light_black),
                yaxis=dict(title='Time', backgroundcolor='white', gridcolor=light_black),
                zaxis=dict(title='Result Control X', backgroundcolor='white', gridcolor=light_black)
            )
        )

        # Create the figure with 3D plots for y data
        fig_3d_y = make_subplots(rows=1, cols=2,
                                 subplot_titles=("Target Qubit - Y axis", "Control Qubits - Y axis"),
                                 specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]])

        # Plot 3D amp_control vs time vs result Y for ground and excited states
        for i, amp_control in enumerate(amp_controls):
            fig_3d_y.add_trace(
                go.Scatter3d(x=[amp_control] * len(times[i]), y=times[i], z=results_ground_y[i], mode='lines',
                             name='Ground Y', showlegend=(i == 0), line=dict(color=dark_navy)),
                row=1, col=1)
            fig_3d_y.add_trace(
                go.Scatter3d(x=[amp_control] * len(times[i]), y=times[i], z=results_excited_y[i], mode='lines',
                             name='Excited Y', showlegend=(i == 0), line=dict(color=dark_purple)),
                row=1, col=1)

        # Plot 3D amp_control vs time vs result control Y for ground and excited states
        for i, amp_control in enumerate(amp_controls):
            fig_3d_y.add_trace(
                go.Scatter3d(x=[amp_control] * len(times[i]), y=times[i], z=results_control_ground_y[i],
                             mode='lines',
                             name='', showlegend=(i == 0), line=dict(color=dark_navy)),
                row=1, col=2)
            fig_3d_y.add_trace(
                go.Scatter3d(x=[amp_control] * len(times[i]), y=times[i], z=results_control_excited_y[i],
                             mode='lines',
                             name='', showlegend=(i == 0), line=dict(color=dark_purple)),
                row=1, col=2)

        # Update layout for 3D Y figure
        fig_3d_y.update_layout(
            height=plot_size_3d[0], width=plot_size_3d[1],
            scene=dict(
                xaxis=dict(title='Amp Control', backgroundcolor='white', gridcolor=light_black),
                yaxis=dict(title='Time', backgroundcolor='white', gridcolor=light_black),
                zaxis=dict(title='Result Target Y', backgroundcolor='white', gridcolor=light_black)
            ),
            scene2=dict(
                xaxis=dict(title='Amp Control', backgroundcolor='white', gridcolor=light_black),
                yaxis=dict(title='Time', backgroundcolor='white', gridcolor=light_black),
                zaxis=dict(title='Result Control Y', backgroundcolor='white', gridcolor=light_black)
            )
        )

        # Extract nominal values and uncertainties
        self.iz_rates = [res['iz_rate'].nominal_value for res in self.results]
        self.iz_uncertainties = [res['iz_rate'].std_dev for res in self.results]
        self.zz_rates = [res['zz_rate'].nominal_value for res in self.results]
        self.zz_uncertainties = [res['zz_rate'].std_dev for res in self.results]
        self.widths = [res['width'] for res in self.results]
        self.amp_controls = [res['amp_control'] for res in self.results]

        # Create the figure with 2D plots
        fig_2d = make_subplots(
            rows=1, cols=3,
            subplot_titles=("ZZ vs Amp Control", "IZ vs Amp Control", "Width vs Amp Control"),
            specs=[[{'type': 'scatter'}, {'type': 'scatter'}, {'type': 'scatter'}]]
        )

        # Plot 2D zz vs amp_control
        fig_2d.add_trace(
            go.Scatter(
                x=self.amp_controls, y=self.zz_rates, mode='lines+markers', name='ZZ vs Amp Control',
                line=dict(color=dark_navy),
                error_y=dict(type='data', array=self.zz_uncertainties, visible=True)
            ),
            row=1, col=1
        )

        # Plot 2D iz vs amp_control
        fig_2d.add_trace(
            go.Scatter(
                x=self.amp_controls, y=self.iz_rates, mode='lines+markers', name='IZ vs Amp Control',
                line=dict(color='gray'),
                error_y=dict(type='data', array=self.iz_uncertainties, visible=True)
            ),
            row=1, col=2
        )

        # Plot 2D width vs amp_control
        fig_2d.add_trace(
            go.Scatter(
                x=self.amp_controls, y=self.widths, mode='lines+markers', name='Width vs Amp Control',
                line=dict(color=dark_purple)
            ),
            row=1, col=3
        )
        # Update layout for 2D figure
        fig_2d.update_layout(
            height=fig_size[0], width=fig_size[1],
            plot_bgcolor='white',
            xaxis=dict(gridcolor=light_black),
            yaxis=dict(gridcolor=light_black)
        )

        # Customize 2D plots
        fig_2d.update_xaxes(title_text="Amp Control", row=1, col=1)
        fig_2d.update_yaxes(title_text="ZZ Rate (MHz)", row=1, col=1)

        fig_2d.update_xaxes(title_text="Amp Control", row=1, col=2)
        fig_2d.update_yaxes(title_text="IZ Rate (MHz)", row=1, col=2)

        fig_2d.update_xaxes(title_text="Amp Control", row=1, col=3)
        fig_2d.update_yaxes(title_text="Width (us)", row=1, col=3)

        return fig_2d.show(), fig_3d_x.show(), fig_3d_y.show()


class ConditionalStarkFinePhaseTuneUp(Experiment):
    @log_and_record
    def run(self, duts, params=None, frequency=None, amp_control=None, phase_diff=0, rise=0.0, trunc=1.0,
            t_start=0, t_stop=20, sweep_points=30,
            phase_diff_start: float = 0, phase_diff_stop: float = 2 * np.pi, phase_diff_step: float = np.pi / 10,
            n_start=0, n_stop=32, update_iz=False, update_zz=True
            ):
        self.duts = duts

        assert update_iz == False

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

        self.current_params = params
        self.params_list = [params]
        self.results = []

        for phase in np.arange(phase_diff_start, phase_diff_stop, phase_diff_step):
            print(f"phase: {phase:.3f} radians")
            self.current_params['phase_diff'] = phase
            iz_rate, zz_rate, width, result, result_target, result_control = self.run_sizzel_xy_hamiltonian_tomography(
                t_start=t_start, t_stop=t_stop, sweep_points=sweep_points
            )

            self.results.append({
                'phase_diff': phase,
                'iz_rate': iz_rate,
                'zz_rate': zz_rate,
                'width': width,
                'result': result,
                'result_target': result_target,
                'result_control': result_control,
                't_start': t_start,
                't_stop': t_stop,
                'sweep_points': sweep_points
            })

            self.current_params['zz_interaction_positive'] = zz_rate.nominal_value > 0

    def run_sizzel_xy_hamiltonian_tomography(self, t_start, t_stop, sweep_points=60):
        t_step = (t_stop - t_start) / sweep_points

        from leeq.experiments.builtin.multi_qubit_gates.sizzel import ConditionalStarkTuneUpRabiXY
        sizzel_xy = ConditionalStarkTuneUpRabiXY(
            qubits=self.duts,
            frequency=self.current_params['frequency'],
            amp_control=self.current_params['amp_control'],
            amp_target=self.current_params['amp_target'],
            rise=self.current_params['rise'],
            start=t_start,
            stop=t_stop,
            step=t_step,
            phase_diff=self.current_params['phase_diff'],
            iz_rate_cancel=0,
            iz_rise_drop=0,
            echo=True
        )

        result = sizzel_xy.analyze_results_with_errs()
        iz_rate = result['iz_rate']
        zz_rate = result['zz_rate']

        new_params = self.current_params.copy()
        new_params['width'] = np.abs(0.125 / zz_rate.nominal_value) / 2
        width = new_params['width']

        print(f'Estimated IZ = {iz_rate} MHz, ZZ = {zz_rate} MHz, width = {new_params["width"]: 0.5f} us')

        self.params_list.append(new_params)
        self.current_params = new_params

        self.result_target = sizzel_xy.result  # Store the result of the current iteration
        self.result_control = sizzel_xy.result_control  # Assuming result_control is accessible from sizzel_xy

        return iz_rate, zz_rate, width, result, self.result_target, self.result_control

    @register_browser_function(available_after=(run,))
    def plot_results(self, plot_size_3d=(1000, 2000), fig_size=(600, 2000)):
        # Colors for the plot
        dark_navy = '#000080'
        dark_purple = '#800080'
        light_black = '#D3D3D3'

        # Prepare data for 3D plots
        phases = [res['phase_diff'] for res in self.results]
        times = [np.linspace(res['t_start'], res['t_stop'], res['sweep_points']) for res in self.results]

        # Extract the necessary data for plotting
        results_ground_x = [np.array(res['result_target'])[:, 0, 0] for res in self.results]
        results_excited_x = [np.array(res['result_target'])[:, 1, 0] for res in self.results]

        results_ground_y = [np.array(res['result_target'])[:, 0, 1] for res in self.results]
        results_excited_y = [np.array(res['result_target'])[:, 1, 1] for res in self.results]

        results_control_ground_x = [np.array(res['result_control'])[:, 0, 0] for res in self.results]
        results_control_excited_x = [np.array(res['result_control'])[:, 1, 0] for res in self.results]

        results_control_ground_y = [np.array(res['result_control'])[:, 0, 1] for res in self.results]
        results_control_excited_y = [np.array(res['result_control'])[:, 1, 1] for res in self.results]

        # Create the figure with 3D plots for x data
        fig_3d_x = make_subplots(rows=1, cols=2,
                                 subplot_titles=("Target Qubit - X axis", "Control Qubits - X axis"),
                                 specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]])

        # Plot 3D phase vs time vs result X for ground and excited states
        for i, phase in enumerate(phases):
            fig_3d_x.add_trace(
                go.Scatter3d(x=[phase] * len(times[i]), y=times[i], z=results_ground_x[i], mode='lines',
                             name='Ground X', showlegend=(i == 0), line=dict(color=dark_navy)),
                row=1, col=1)
            fig_3d_x.add_trace(
                go.Scatter3d(x=[phase] * len(times[i]), y=times[i], z=results_excited_x[i], mode='lines',
                             name='Excited X', showlegend=(i == 0), line=dict(color=dark_purple)),
                row=1, col=1)

        # Plot 3D phase vs time vs result control X for ground and excited states
        for i, phase in enumerate(phases):
            fig_3d_x.add_trace(
                go.Scatter3d(x=[phase] * len(times[i]), y=times[i], z=results_control_ground_x[i], mode='lines',
                             name='', showlegend=(i == 0), line=dict(color=dark_navy)),
                row=1, col=2)
            fig_3d_x.add_trace(
                go.Scatter3d(x=[phase] * len(times[i]), y=times[i], z=results_control_excited_x[i], mode='lines',
                             name='', showlegend=(i == 0), line=dict(color=dark_purple)),
                row=1, col=2)

        # Update layout for 3D X figure
        fig_3d_x.update_layout(
            height=plot_size_3d[0], width=plot_size_3d[1],
            # title_text="Conditional Stark Fine Phase Tune-Up Results - 3D X",
            scene=dict(
                xaxis=dict(title='Phase', backgroundcolor='white', gridcolor=light_black),
                yaxis=dict(title='Time', backgroundcolor='white', gridcolor=light_black),
                zaxis=dict(title='Result Target X', backgroundcolor='white', gridcolor=light_black)
            ),
            scene2=dict(
                xaxis=dict(title='Phase', backgroundcolor='white', gridcolor=light_black),
                yaxis=dict(title='Time', backgroundcolor='white', gridcolor=light_black),
                zaxis=dict(title='Result Control X', backgroundcolor='white', gridcolor=light_black)
            )
        )

        # Create the figure with 3D plots for y data
        fig_3d_y = make_subplots(rows=1, cols=2,
                                 subplot_titles=("Target Qubit - Y axis", "Control Qubits - Y axis"),
                                 specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]])

        # Plot 3D phase vs time vs result Y for ground and excited states
        for i, phase in enumerate(phases):
            fig_3d_y.add_trace(
                go.Scatter3d(x=[phase] * len(times[i]), y=times[i], z=results_ground_y[i], mode='lines',
                             name='Ground Y', showlegend=(i == 0), line=dict(color=dark_navy)),
                row=1, col=1)
            fig_3d_y.add_trace(
                go.Scatter3d(x=[phase] * len(times[i]), y=times[i], z=results_excited_y[i], mode='lines',
                             name='Excited Y', showlegend=(i == 0), line=dict(color=dark_purple)),
                row=1, col=1)

        # Plot 3D phase vs time vs result control Y for ground and excited states
        for i, phase in enumerate(phases):
            fig_3d_y.add_trace(
                go.Scatter3d(x=[phase] * len(times[i]), y=times[i], z=results_control_ground_y[i], mode='lines',
                             name='', showlegend=(i == 0), line=dict(color=dark_navy)),
                row=1, col=2)
            fig_3d_y.add_trace(
                go.Scatter3d(x=[phase] * len(times[i]), y=times[i], z=results_control_excited_y[i], mode='lines',
                             name='', showlegend=(i == 0), line=dict(color=dark_purple)),
                row=1, col=2)

        # Update layout for 3D Y figure
        fig_3d_y.update_layout(
            height=plot_size_3d[0], width=plot_size_3d[1],
            # title_text="Conditional Stark Fine Phase Tune-Up Results - 3D Y",
            scene=dict(
                xaxis=dict(title='Phase', backgroundcolor='white', gridcolor=light_black),
                yaxis=dict(title='Time', backgroundcolor='white', gridcolor=light_black),
                zaxis=dict(title='Result Target Y', backgroundcolor='white', gridcolor=light_black)
            ),
            scene2=dict(
                xaxis=dict(title='Phase', backgroundcolor='white', gridcolor=light_black),
                yaxis=dict(title='Time', backgroundcolor='white', gridcolor=light_black),
                zaxis=dict(title='Result Control Y', backgroundcolor='white', gridcolor=light_black)
            )
        )

        # Extract nominal values and uncertainties
        self.iz_rates = [res['iz_rate'].nominal_value for res in self.results]
        self.iz_uncertainties = [res['iz_rate'].std_dev for res in self.results]
        self.zz_rates = [res['zz_rate'].nominal_value for res in self.results]
        self.zz_uncertainties = [res['zz_rate'].std_dev for res in self.results]
        self.widths = [res['width'] for res in self.results]
        self.phases = [res['phase_diff'] for res in self.results]

        # Create the figure with 2D plots
        fig_2d = make_subplots(
            rows=1, cols=3,
            subplot_titles=("ZZ vs Phase", "IZ vs Phase", "Width vs Phase"),
            specs=[[{'type': 'scatter'}, {'type': 'scatter'}, {'type': 'scatter'}]]
        )

        # Plot 2D zz vs phase
        fig_2d.add_trace(
            go.Scatter(
                x=self.phases, y=self.zz_rates, mode='lines+markers', name='ZZ vs Phase',
                line=dict(color=dark_navy),
                error_y=dict(type='data', array=self.zz_uncertainties, visible=True)
            ),
            row=1, col=1
        )

        # Plot 2D iz vs phase
        fig_2d.add_trace(
            go.Scatter(
                x=self.phases, y=self.iz_rates, mode='lines+markers', name='IZ vs Phase',
                line=dict(color='gray'),
                error_y=dict(type='data', array=self.iz_uncertainties, visible=True)
            ),
            row=1, col=2
        )

        # Plot 2D width vs phase
        fig_2d.add_trace(
            go.Scatter(
                x=self.phases, y=self.widths, mode='lines+markers', name='Width vs Phase',
                line=dict(color=dark_purple)
            ),
            row=1, col=3
        )

        # Update layout for 2D figure
        fig_2d.update_layout(
            height=fig_size[0], width=fig_size[1],
            plot_bgcolor='white',
            xaxis=dict(gridcolor=light_black),
            yaxis=dict(gridcolor=light_black)
        )

        # Customize 2D plots
        fig_2d.update_xaxes(title_text="Phase", row=1, col=1)
        fig_2d.update_yaxes(title_text="ZZ Rate (MHz)", row=1, col=1)

        fig_2d.update_xaxes(title_text="Phase", row=1, col=2)
        fig_2d.update_yaxes(title_text="IZ Rate (MHz)", row=1, col=2)

        fig_2d.update_xaxes(title_text="Phase", row=1, col=3)
        fig_2d.update_yaxes(title_text="Width (us)", row=1, col=3)

        return fig_2d.show(), fig_3d_x.show(), fig_3d_y.show()


class ConditionalStarkFineRiseTuneUp(Experiment):
    @log_and_record
    def run(self, duts, params=None, frequency=None, amp_control=None, phase_diff=0, trunc=1.0,
            rise_start=0.01, rise_stop=0.1, rise_step=0.01,
            t_start=0, t_stop=20, sweep_points=30,
            n_start=0, n_stop=32, update_iz=False, update_zz=True):
        self.duts = duts

        assert update_iz == False

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
                'rise': rise_start,
                'trunc': trunc,
                'width': 0,
                'phase_diff': phase_diff,
                'zz_interaction_positive': True,
                'echo': True
            }

        self.current_params = params
        self.params_list = [params]
        self.results = []

        for rise in np.arange(rise_start, rise_stop, rise_step):
            print(f"rise: {rise}")
            self.current_params['rise'] = rise
            iz_rate, zz_rate, width, result, result_target, result_control = self.run_sizzel_xy_hamiltonian_tomography(
                t_start=t_start, t_stop=t_stop, sweep_points=sweep_points
            )

            self.results.append({
                'rise': rise,
                'iz_rate': iz_rate,
                'zz_rate': zz_rate,
                'width': width,
                'result': result,
                'result_target': result_target,
                'result_control': result_control,
                't_start': t_start,
                't_stop': t_stop,
                'sweep_points': sweep_points
            })

            self.current_params['zz_interaction_positive'] = zz_rate.nominal_value > 0

    def run_sizzel_xy_hamiltonian_tomography(self, t_start, t_stop, sweep_points=60):
        t_step = (t_stop - t_start) / sweep_points

        from leeq.experiments.builtin.multi_qubit_gates.sizzel import ConditionalStarkTuneUpRabiXY
        sizzel_xy = ConditionalStarkTuneUpRabiXY(
            qubits=self.duts,
            frequency=self.current_params['frequency'],
            amp_control=self.current_params['amp_control'],
            amp_target=self.current_params['amp_target'],
            rise=self.current_params['rise'],
            start=t_start,
            stop=t_stop,
            step=t_step,
            phase_diff=self.current_params['phase_diff'],
            iz_rate_cancel=0,
            iz_rise_drop=0,
            echo=True
        )

        result = sizzel_xy.analyze_results_with_errs()
        iz_rate = result['iz_rate']
        zz_rate = result['zz_rate']

        new_params = self.current_params.copy()
        new_params['width'] = np.abs(0.125 / zz_rate.nominal_value) / 2
        width = new_params['width']

        print(f'Estimated IZ = {iz_rate} MHz, ZZ = {zz_rate} MHz, width = {new_params["width"]: 0.5f} us')

        self.params_list.append(new_params)
        self.current_params = new_params

        self.result_target = sizzel_xy.result  # Store the result of the current iteration
        self.result_control = sizzel_xy.result_control  # Assuming result_control is accessible from sizzel_xy

        return iz_rate, zz_rate, width, result, self.result_target, self.result_control

    @register_browser_function(available_after=(run,))
    def plot_results(self, plot_size_3d=(1000, 2000), fig_size=(600, 2000)):
        # Colors for the plot
        dark_navy = '#000080'
        dark_purple = '#800080'
        light_black = '#D3D3D3'

        # Prepare data for 3D plots
        rises = [res['rise'] for res in self.results]
        times = [np.linspace(res['t_start'], res['t_stop'], res['sweep_points']) for res in self.results]

        # Extract the necessary data for plotting
        results_ground_x = [np.array(res['result_target'])[:, 0, 0] for res in self.results]
        results_excited_x = [np.array(res['result_target'])[:, 1, 0] for res in self.results]

        results_ground_y = [np.array(res['result_target'])[:, 0, 1] for res in self.results]
        results_excited_y = [np.array(res['result_target'])[:, 1, 1] for res in self.results]

        results_control_ground_x = [np.array(res['result_control'])[:, 0, 0] for res in self.results]
        results_control_excited_x = [np.array(res['result_control'])[:, 1, 0] for res in self.results]

        results_control_ground_y = [np.array(res['result_control'])[:, 0, 1] for res in self.results]
        results_control_excited_y = [np.array(res['result_control'])[:, 1, 1] for res in self.results]

        # Create the figure with 3D plots for x data
        fig_3d_x = make_subplots(rows=1, cols=2,
                                 subplot_titles=("Target Qubit - X axis", "Control Qubits - X axis"),
                                 specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]])

        # Plot 3D rise vs time vs result X for ground and excited states
        for i, rise in enumerate(rises):
            fig_3d_x.add_trace(
                go.Scatter3d(x=[rise] * len(times[i]), y=times[i], z=results_ground_x[i], mode='lines',
                             name='Ground X', showlegend=(i == 0), line=dict(color=dark_navy)),
                row=1, col=1)
            fig_3d_x.add_trace(
                go.Scatter3d(x=[rise] * len(times[i]), y=times[i], z=results_excited_x[i], mode='lines',
                             name='Excited X', showlegend=(i == 0), line=dict(color=dark_purple)),
                row=1, col=1)

        # Plot 3D rise vs time vs result control X for ground and excited states
        for i, rise in enumerate(rises):
            fig_3d_x.add_trace(
                go.Scatter3d(x=[rise] * len(times[i]), y=times[i], z=results_control_ground_x[i], mode='lines',
                             name='', showlegend=(i == 0), line=dict(color=dark_navy)),
                row=1, col=2)
            fig_3d_x.add_trace(
                go.Scatter3d(x=[rise] * len(times[i]), y=times[i], z=results_control_excited_x[i], mode='lines',
                             name='', showlegend=(i == 0), line=dict(color=dark_purple)),
                row=1, col=2)

        # Update layout for 3D X figure
        fig_3d_x.update_layout(
            height=plot_size_3d[0], width=plot_size_3d[1],
            scene=dict(
                xaxis=dict(title='Rise', backgroundcolor='white', gridcolor=light_black),
                yaxis=dict(title='Time', backgroundcolor='white', gridcolor=light_black),
                zaxis=dict(title='Result Target X', backgroundcolor='white', gridcolor=light_black)
            ),
            scene2=dict(
                xaxis=dict(title='Rise', backgroundcolor='white', gridcolor=light_black),
                yaxis=dict(title='Time', backgroundcolor='white', gridcolor=light_black),
                zaxis=dict(title='Result Control X', backgroundcolor='white', gridcolor=light_black)
            )
        )

        # Create the figure with 3D plots for y data
        fig_3d_y = make_subplots(rows=1, cols=2,
                                 subplot_titles=("Target Qubit - Y axis", "Control Qubits - Y axis"),
                                 specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]])

        # Plot 3D rise vs time vs result Y for ground and excited states
        for i, rise in enumerate(rises):
            fig_3d_y.add_trace(
                go.Scatter3d(x=[rise] * len(times[i]), y=times[i], z=results_ground_y[i], mode='lines',
                             name='Ground Y', showlegend=(i == 0), line=dict(color=dark_navy)),
                row=1, col=1)
            fig_3d_y.add_trace(
                go.Scatter3d(x=[rise] * len(times[i]), y=times[i], z=results_excited_y[i], mode='lines',
                             name='Excited Y', showlegend=(i == 0), line=dict(color=dark_purple)),
                row=1, col=1)

        # Plot 3D rise vs time vs result control Y for ground and excited states
        for i, rise in enumerate(rises):
            fig_3d_y.add_trace(
                go.Scatter3d(x=[rise] * len(times[i]), y=times[i], z=results_control_ground_y[i], mode='lines',
                             name='', showlegend=(i == 0), line=dict(color=dark_navy)),
                row=1, col=2)
            fig_3d_y.add_trace(
                go.Scatter3d(x=[rise] * len(times[i]), y=times[i], z=results_control_excited_y[i], mode='lines',
                             name='', showlegend=(i == 0), line=dict(color=dark_purple)),
                row=1, col=2)

        # Update layout for 3D Y figure
        fig_3d_y.update_layout(
            height=plot_size_3d[0], width=plot_size_3d[1],
            scene=dict(
                xaxis=dict(title='Rise', backgroundcolor='white', gridcolor=light_black),
                yaxis=dict(title='Time', backgroundcolor='white', gridcolor=light_black),
                zaxis=dict(title='Result Target Y', backgroundcolor='white', gridcolor=light_black)
            ),
            scene2=dict(
                xaxis=dict(title='Rise', backgroundcolor='white', gridcolor=light_black),
                yaxis=dict(title='Time', backgroundcolor='white', gridcolor=light_black),
                zaxis=dict(title='Result Control Y', backgroundcolor='white', gridcolor=light_black)
            )
        )

        # Extract nominal values and uncertainties
        self.iz_rates = [res['iz_rate'].nominal_value for res in self.results]
        self.iz_uncertainties = [res['iz_rate'].std_dev for res in self.results]
        self.zz_rates = [res['zz_rate'].nominal_value for res in self.results]
        self.zz_uncertainties = [res['zz_rate'].std_dev for res in self.results]
        self.widths = [res['width'] for res in self.results]
        self.rises = [res['rise'] for res in self.results]

        # Create the figure with 2D plots
        fig_2d = make_subplots(
            rows=1, cols=3,
            subplot_titles=("ZZ vs Rise", "IZ vs Rise", "Width vs Rise"),
            specs=[[{'type': 'scatter'}, {'type': 'scatter'}, {'type': 'scatter'}]]
        )

        # Plot 2D zz vs rise
        fig_2d.add_trace(
            go.Scatter(
                x=self.rises, y=self.zz_rates, mode='lines+markers', name='ZZ vs Rise',
                line=dict(color=dark_navy),
                error_y=dict(type='data', array=self.zz_uncertainties, visible=True)
            ),
            row=1, col=1
        )

        # Plot 2D iz vs rise
        fig_2d.add_trace(
            go.Scatter(
                x=self.rises, y=self.iz_rates, mode='lines+markers', name='IZ vs Rise',
                line=dict(color='gray'),
                error_y=dict(type='data', array=self.iz_uncertainties, visible=True)
            ),
            row=1, col=2
        )

        # Plot 2D width vs rise
        fig_2d.add_trace(
            go.Scatter(
                x=self.rises, y=self.widths, mode='lines+markers', name='Width vs Rise',
                line=dict(color=dark_purple)
            ),
            row=1, col=3
        )

        # Update layout for 2D figure
        fig_2d.update_layout(
            height=fig_size[0], width=fig_size[1],
            plot_bgcolor='white',
            xaxis=dict(gridcolor=light_black),
            yaxis=dict(gridcolor=light_black)
        )

        # Customize 2D plotsfig_2d.update_xaxes(title_text="Rise", row=1, col=1)
        fig_2d.update_yaxes(title_text="ZZ Rate (MHz)", row=1, col=1)

        fig_2d.update_xaxes(title_text="Rise", row=1, col=2)
        fig_2d.update_yaxes(title_text="IZ Rate (MHz)", row=1, col=2)

        fig_2d.update_xaxes(title_text="Rise", row=1, col=3)
        fig_2d.update_yaxes(title_text="Width (us)", row=1, col=3)

        return fig_2d.show(), fig_3d_x.show(), fig_3d_y.show()


class ConditionalStarkFineTruncTuneUp(Experiment):
    @log_and_record
    def run(self, duts, params=None, frequency=None, amp_control=None, phase_diff=0,
            trunc_start=0.5, trunc_stop=2.0, trunc_step=0.1,
            rise=0.01, t_start=0, t_stop=20, sweep_points=30,
            n_start=0, n_stop=32, update_iz=False, update_zz=True):
        self.duts = duts

        assert update_iz == False

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
                'trunc': trunc_start,
                'width': 0,
                'phase_diff': phase_diff,
                'zz_interaction_positive': True,
                'echo': True
            }

        self.current_params = params
        self.params_list = [params]
        self.results = []

        for trunc in np.arange(trunc_start, trunc_stop, trunc_step):
            print(f"trunc: {trunc}")
            self.current_params['trunc'] = trunc
            iz_rate, zz_rate, width, result, result_target, result_control = self.run_sizzel_xy_hamiltonian_tomography(
                t_start=t_start, t_stop=t_stop, sweep_points=sweep_points
            )

            self.results.append({
                'trunc': trunc,
                'iz_rate': iz_rate,
                'zz_rate': zz_rate,
                'width': width,
                'result': result,
                'result_target': result_target,
                'result_control': result_control,
                't_start': t_start,
                't_stop': t_stop,
                'sweep_points': sweep_points
            })

            self.current_params['zz_interaction_positive'] = zz_rate.nominal_value > 0

    def run_sizzel_xy_hamiltonian_tomography(self, t_start, t_stop, sweep_points=60):
        t_step = (t_stop - t_start) / sweep_points

        from leeq.experiments.builtin.multi_qubit_gates.sizzel import ConditionalStarkTuneUpRabiXY
        sizzel_xy = ConditionalStarkTuneUpRabiXY(
            qubits=self.duts,
            frequency=self.current_params['frequency'],
            amp_control=self.current_params['amp_control'],
            amp_target=self.current_params['amp_target'],
            rise=self.current_params['rise'],
            start=t_start,
            stop=t_stop,
            step=t_step,
            phase_diff=self.current_params['phase_diff'],
            iz_rate_cancel=0,
            iz_rise_drop=0,
            echo=True
        )

        result = sizzel_xy.analyze_results_with_errs()
        iz_rate = result['iz_rate']
        zz_rate = result['zz_rate']

        new_params = self.current_params.copy()
        new_params['width'] = np.abs(0.125 / zz_rate.nominal_value) / 2
        width = new_params['width']

        print(f'Estimated IZ = {iz_rate} MHz, ZZ = {zz_rate} MHz, width = {new_params["width"]: 0.5f} us')

        self.params_list.append(new_params)
        self.current_params = new_params

        self.result_target = sizzel_xy.result  # Store the result of the current iteration
        self.result_control = sizzel_xy.result_control  # Assuming result_control is accessible from sizzel_xy

        return iz_rate, zz_rate, width, result, self.result_target, self.result_control

    @register_browser_function(available_after=(run,))
    def plot_results(self, plot_size_3d=(800, 2000), fig_size=(600, 2000)):
        # Colors for the plot
        dark_navy = '#000080'
        dark_purple = '#800080'
        light_black = '#D3D3D3'

        # Prepare data for 3D plots
        truncs = [res['trunc'] for res in self.results]
        times = [np.linspace(res['t_start'], res['t_stop'], res['sweep_points']) for res in self.results]

        # Extract the necessary data for plotting
        results_ground_x = [np.array(res['result_target'])[:, 0, 0] for res in self.results]
        results_excited_x = [np.array(res['result_target'])[:, 1, 0] for res in self.results]

        results_ground_y = [np.array(res['result_target'])[:, 0, 1] for res in self.results]
        results_excited_y = [np.array(res['result_target'])[:, 1, 1] for res in self.results]

        results_control_ground_x = [np.array(res['result_control'])[:, 0, 0] for res in self.results]
        results_control_excited_x = [np.array(res['result_control'])[:, 1, 0] for res in self.results]

        results_control_ground_y = [np.array(res['result_control'])[:, 0, 1] for res in self.results]
        results_control_excited_y = [np.array(res['result_control'])[:, 1, 1] for res in self.results]

        iz_rates = [res['iz_rate'].nominal_value for res in self.results]
        zz_rates = [res['zz_rate'].nominal_value for res in self.results]

        # Create the figure with 3D plots for x data
        fig_3d_x = make_subplots(rows=1, cols=2,
                                 subplot_titles=("Target Qubit - X axis", "Control Qubits - X axis"),
                                 specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]])

        # Plot 3D trunc vs time vs result X for ground and excited states
        for i, trunc in enumerate(truncs):
            fig_3d_x.add_trace(
                go.Scatter3d(x=[trunc] * len(times[i]), y=times[i], z=results_ground_x[i], mode='lines',
                             name='Ground X', showlegend=(i == 0), line=dict(color=dark_navy)),
                row=1, col=1)
            fig_3d_x.add_trace(
                go.Scatter3d(x=[trunc] * len(times[i]), y=times[i], z=results_excited_x[i], mode='lines',
                             name='Excited X', showlegend=(i == 0), line=dict(color=dark_purple)),
                row=1, col=1)

        # Plot 3D trunc vs time vs result control X for ground and excited states
        for i, trunc in enumerate(truncs):
            fig_3d_x.add_trace(
                go.Scatter3d(x=[trunc] * len(times[i]), y=times[i], z=results_control_ground_x[i], mode='lines',
                             name='', showlegend=(i == 0), line=dict(color=dark_navy)),
                row=1, col=2)
            fig_3d_x.add_trace(
                go.Scatter3d(x=[trunc] * len(times[i]), y=times[i], z=results_control_excited_x[i], mode='lines',
                             name='', showlegend=(i == 0), line=dict(color=dark_purple)),
                row=1, col=2)

        # Update layout for 3D X figure
        fig_3d_x.update_layout(
            height=plot_size_3d[0], width=plot_size_3d[1],
            scene=dict(
                xaxis=dict(title='Trunc', backgroundcolor='white', gridcolor=light_black),
                yaxis=dict(title='Time', backgroundcolor='white', gridcolor=light_black),
                zaxis=dict(title='Result Target X', backgroundcolor='white', gridcolor=light_black)
            ),
            scene2=dict(
                xaxis=dict(title='Trunc', backgroundcolor='white', gridcolor=light_black),
                yaxis=dict(title='Time', backgroundcolor='white', gridcolor=light_black),
                zaxis=dict(title='Result Control X', backgroundcolor='white', gridcolor=light_black)
            )
        )

        # Create the figure with 3D plots for y data
        fig_3d_y = make_subplots(rows=1, cols=2,
                                 subplot_titles=("Target Qubit - Y axis", "Control Qubits - Y axis"),
                                 specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]])

        # Plot 3D trunc vs time vs result Y for ground and excited states
        for i, trunc in enumerate(truncs):
            fig_3d_y.add_trace(
                go.Scatter3d(x=[trunc] * len(times[i]), y=times[i], z=results_ground_y[i], mode='lines',
                             name='Ground Y', showlegend=(i == 0), line=dict(color=dark_navy)),
                row=1, col=1)
            fig_3d_y.add_trace(
                go.Scatter3d(x=[trunc] * len(times[i]), y=times[i], z=results_excited_y[i], mode='lines',
                             name='Excited Y', showlegend=(i == 0), line=dict(color=dark_purple)),
                row=1, col=1)

        # Plot 3D trunc vs time vs result control Y for ground and excited states
        for i, trunc in enumerate(truncs):
            fig_3d_y.add_trace(
                go.Scatter3d(x=[trunc] * len(times[i]), y=times[i], z=results_control_ground_y[i], mode='lines',
                             name='', showlegend=(i == 0), line=dict(color=dark_navy)),
                row=1, col=2)
            fig_3d_y.add_trace(
                go.Scatter3d(x=[trunc] * len(times[i]), y=times[i], z=results_control_excited_y[i], mode='lines',
                             name='', showlegend=(i == 0), line=dict(color=dark_purple)),
                row=1, col=2)

        # Update layout for 3D Y figure
        fig_3d_y.update_layout(
            height=plot_size_3d[0], width=plot_size_3d[1],
            scene=dict(
                xaxis=dict(title='Trunc', backgroundcolor='white', gridcolor=light_black),
                yaxis=dict(title='Time', backgroundcolor='white', gridcolor=light_black),
                zaxis=dict(title='Result Target Y', backgroundcolor='white', gridcolor=light_black)
            ),
            scene2=dict(
                xaxis=dict(title='Trunc', backgroundcolor='white', gridcolor=light_black),
                yaxis=dict(title='Time', backgroundcolor='white', gridcolor=light_black),
                zaxis=dict(title='Result Control Y', backgroundcolor='white', gridcolor=light_black)
            )
        )

        # Create the third figure with 2D plots
        fig_2d = make_subplots(rows=1, cols=2,
                               subplot_titles=("ZZ vs Trunc", "IZ vs Trunc"),
                               specs=[[{'type': 'scatter'}, {'type': 'scatter'}]])

        # Plot 2D zz vs trunc
        fig_2d.add_trace(
            go.Scatter(x=truncs, y=zz_rates, mode='lines+markers', name='ZZ vs Trunc', line=dict(color=dark_navy)),
            row=1, col=1)

        # Plot 2D iz vs trunc
        fig_2d.add_trace(
            go.Scatter(x=truncs, y=iz_rates, mode='lines+markers', name='IZ vs Trunc', line=dict(color=dark_purple)),
            row=1, col=2)

        # Update layout for 2D figure
        fig_2d.update_layout(
            height=fig_size[0], width=fig_size[1],
            plot_bgcolor='white',
            xaxis=dict(gridcolor=light_black),
            yaxis=dict(gridcolor=light_black)
        )

        # Customize 2D plots
        fig_2d.update_xaxes(title_text="Trunc", row=1, col=1)
        fig_2d.update_yaxes(title_text="ZZ Rate (MHz)", row=1, col=1)

        fig_2d.update_xaxes(title_text="Trunc", row=1, col=2)
        fig_2d.update_yaxes(title_text="IZ Rate (MHz)", row=1, col=2)

        return fig_3d_x.show(), fig_3d_y.show(), fig_2d.show()

    @register_browser_function(available_after=(run,))
    def plot_results(self, plot_size_3d=(1000, 2000), fig_size=(600, 2000)):
        # Colors for the plot
        dark_navy = '#000080'
        dark_purple = '#800080'
        light_black = '#D3D3D3'

        # Prepare data for 3D plots
        truncs = [res['trunc'] for res in self.results]
        times = [np.linspace(res['t_start'], res['t_stop'], res['sweep_points']) for res in self.results]

        # Extract the necessary data for plotting
        results_ground_x = [np.array(res['result_target'])[:, 0, 0] for res in self.results]
        results_excited_x = [np.array(res['result_target'])[:, 1, 0] for res in self.results]

        results_ground_y = [np.array(res['result_target'])[:, 0, 1] for res in self.results]
        results_excited_y = [np.array(res['result_target'])[:, 1, 1] for res in self.results]

        results_control_ground_x = [np.array(res['result_control'])[:, 0, 0] for res in self.results]
        results_control_excited_x = [np.array(res['result_control'])[:, 1, 0] for res in self.results]

        results_control_ground_y = [np.array(res['result_control'])[:, 0, 1] for res in self.results]
        results_control_excited_y = [np.array(res['result_control'])[:, 1, 1] for res in self.results]

        # Create the figure with 3D plots for x data
        fig_3d_x = make_subplots(rows=1, cols=2,
                                 subplot_titles=("Target Qubit - X axis", "Control Qubits - X axis"),
                                 specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]])

        # Plot 3D trunc vs time vs result X for ground and excited states
        for i, trunc in enumerate(truncs):
            fig_3d_x.add_trace(
                go.Scatter3d(x=[trunc] * len(times[i]), y=times[i], z=results_ground_x[i], mode='lines',
                             name='Ground X', showlegend=(i == 0), line=dict(color=dark_navy)),
                row=1, col=1)
            fig_3d_x.add_trace(
                go.Scatter3d(x=[trunc] * len(times[i]), y=times[i], z=results_excited_x[i], mode='lines',
                             name='Excited X', showlegend=(i == 0), line=dict(color=dark_purple)),
                row=1, col=1)

        # Plot 3D trunc vs time vs result control X for ground and excited states
        for i, trunc in enumerate(truncs):
            fig_3d_x.add_trace(
                go.Scatter3d(x=[trunc] * len(times[i]), y=times[i], z=results_control_ground_x[i], mode='lines',
                             name='', showlegend=(i == 0), line=dict(color=dark_navy)),
                row=1, col=2)
            fig_3d_x.add_trace(
                go.Scatter3d(x=[trunc] * len(times[i]), y=times[i], z=results_control_excited_x[i], mode='lines',
                             name='', showlegend=(i == 0), line=dict(color=dark_purple)),
                row=1, col=2)

        # Update layout for 3D X figure
        fig_3d_x.update_layout(
            height=plot_size_3d[0], width=plot_size_3d[1],
            scene=dict(
                xaxis=dict(title='Trunc', backgroundcolor='white', gridcolor=light_black),
                yaxis=dict(title='Time', backgroundcolor='white', gridcolor=light_black),
                zaxis=dict(title='Result Target X', backgroundcolor='white', gridcolor=light_black)
            ),
            scene2=dict(
                xaxis=dict(title='Trunc', backgroundcolor='white', gridcolor=light_black),
                yaxis=dict(title='Time', backgroundcolor='white', gridcolor=light_black),
                zaxis=dict(title='Result Control X', backgroundcolor='white', gridcolor=light_black)
            )
        )

        # Create the figure with 3D plots for y data
        fig_3d_y = make_subplots(rows=1, cols=2,
                                 subplot_titles=("Target Qubit - Y axis", "Control Qubits - Y axis"),
                                 specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]])

        # Plot 3D trunc vs time vs result Y for ground and excited states
        for i, trunc in enumerate(truncs):
            fig_3d_y.add_trace(
                go.Scatter3d(x=[trunc] * len(times[i]), y=times[i], z=results_ground_y[i], mode='lines',
                             name='Ground Y', showlegend=(i == 0), line=dict(color=dark_navy)),
                row=1, col=1)
            fig_3d_y.add_trace(
                go.Scatter3d(x=[trunc] * len(times[i]), y=times[i], z=results_excited_y[i], mode='lines',
                             name='Excited Y', showlegend=(i == 0), line=dict(color=dark_purple)),
                row=1, col=1)

        # Plot 3D trunc vs time vs result control Y for ground and excited states
        for i, trunc in enumerate(truncs):
            fig_3d_y.add_trace(
                go.Scatter3d(x=[trunc] * len(times[i]), y=times[i], z=results_control_ground_y[i], mode='lines',
                             name='', showlegend=(i == 0), line=dict(color=dark_navy)),
                row=1, col=2)
            fig_3d_y.add_trace(
                go.Scatter3d(x=[trunc] * len(times[i]), y=times[i], z=results_control_excited_y[i], mode='lines',
                             name='', showlegend=(i == 0), line=dict(color=dark_purple)),
                row=1, col=2)

        # Update layout for 3D Y figure
        fig_3d_y.update_layout(
            height=plot_size_3d[0], width=plot_size_3d[1],
            scene=dict(
                xaxis=dict(title='Trunc', backgroundcolor='white', gridcolor=light_black),
                yaxis=dict(title='Time', backgroundcolor='white', gridcolor=light_black),
                zaxis=dict(title='Result Target Y', backgroundcolor='white', gridcolor=light_black)
            ),
            scene2=dict(
                xaxis=dict(title='Trunc', backgroundcolor='white', gridcolor=light_black),
                yaxis=dict(title='Time', backgroundcolor='white', gridcolor=light_black),
                zaxis=dict(title='Result Control Y', backgroundcolor='white', gridcolor=light_black)
            )
        )

        # Extract nominal values and uncertainties
        self.iz_rates = [res['iz_rate'].nominal_value for res in self.results]
        self.iz_uncertainties = [res['iz_rate'].std_dev for res in self.results]
        self.zz_rates = [res['zz_rate'].nominal_value for res in self.results]
        self.zz_uncertainties = [res['zz_rate'].std_dev for res in self.results]
        self.widths = [res['width'] for res in self.results]
        self.truncs = [res['trunc'] for res in self.results]

        # Create the figure with 2D plots
        fig_2d = make_subplots(
            rows=1, cols=3,
            subplot_titles=("ZZ vs Trunc", "IZ vs Trunc", "Width vs Trunc"),
            specs=[[{'type': 'scatter'}, {'type': 'scatter'}, {'type': 'scatter'}]]
        )

        # Plot 2D zz vs trunc
        fig_2d.add_trace(
            go.Scatter(
                x=self.truncs, y=self.zz_rates, mode='lines+markers', name='ZZ vs Trunc',
                line=dict(color=dark_navy),
                error_y=dict(type='data', array=self.zz_uncertainties, visible=True)
            ),
            row=1, col=1
        )

        # Plot 2D iz vs trunc
        fig_2d.add_trace(
            go.Scatter(
                x=self.truncs, y=self.iz_rates, mode='lines+markers', name='IZ vs Trunc',
                line=dict(color='gray'),
                error_y=dict(type='data', array=self.iz_uncertainties, visible=True)
            ),
            row=1, col=2
        )

        # Plot 2D width vs trunc
        fig_2d.add_trace(
            go.Scatter(
                x=self.truncs, y=self.widths, mode='lines+markers', name='Width vs Trunc',
                line=dict(color=dark_purple)
            ),
            row=1, col=3
        )

        # Update layout for 2D figure
        fig_2d.update_layout(
            height=fig_size[0], width=fig_size[1],
            plot_bgcolor='white',
            xaxis=dict(gridcolor=light_black),
            yaxis=dict(gridcolor=light_black)
        )

        # Customize 2D plots
        fig_2d.update_xaxes(title_text="Trunc", row=1, col=1)
        fig_2d.update_yaxes(title_text="ZZ Rate (MHz)", row=1, col=1)

        fig_2d.update_xaxes(title_text="Trunc", row=1, col=2)
        fig_2d.update_yaxes(title_text="IZ Rate (MHz)", row=1, col=2)

        fig_2d.update_xaxes(title_text="Trunc", row=1, col=3)
        fig_2d.update_yaxes(title_text="Width (us)", row=1, col=3)

        return fig_2d.show(), fig_3d_x.show(), fig_3d_y.show()
