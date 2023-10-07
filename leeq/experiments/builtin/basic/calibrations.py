import numpy as np
from scipy import optimize as so
import plotly.graph_objects as go
from labchronicle import log_and_record, register_browser_function
from leeq.utils.compatibility import *
class ResonatorSweepTransmissionWithExtraInitialLPB(experiment):
    @log_and_record
    def run(self, dut_qubit, start=8000, stop=9000, step=5., res_power=-10., num_avs=1000, rep_rate=10., mp_width=None,
            initial_lpb=None, update=True, amp=50e-3):

        res_channel = dut_qubit.get_default_measurement_prim().primary_channel()
        setup().status().set_channel_params(res_channel, power_mode='FIXED', power=res_power)

        # Sweep the frequency
        mprim_index = 0
        mp = dut_qubit.get_measurement_prim_intlist(mprim_index).clone()
        original_freq = mp.freq

        mp.update_pulse_args(width=rep_rate) if mp_width == None else mp.update_pulse_args(width=mp_width)
        if amp is not None:
            mp.update_pulse_args(amp=amp)
        mp.set_transform_function(None)

        delay = prims.Delay(0)

        if initial_lpb is not None:
            lpb = initial_lpb + delay + mp
        else:
            lpb = delay + mp  # seems necessary to run

        swp = sweeper(np.arange, n_kwargs={'start': start, 'stop': stop, 'step': step},
                      params=[sparam.func(mp.update_freq, {}, 'freq')])
        # Perform the experiment
        setup().status().set_param("Shot_Number", num_avs)
        setup().status().set_param("Shot_Period", rep_rate)
        basic(lpb, swp, '')

        result = mp.result()

        print(np.shape(mp.result()))

        self.trace = np.average(mp.result(), axis=1)  # GC temp fix

        print(np.shape(self.trace))
        # print(self.trace)

        self.result = {'Magnitude': np.absolute(self.trace), 'Phase': np.angle(self.trace), 'Real': np.real(self.trace),
                       'Imaginary': np.imag(self.trace)}

        # print(np.shape(self.result['Magnitude']))

        # mp.update_pulse_args(width=original_width)

        self.frequency_guess = np.arange(**{'start': start, 'stop': stop, 'step': step})[
            np.argmin(self.result['Magnitude'])]
        self.frequency_guess = np.arange(**{'start': start, 'stop': stop, 'step': step})[
            np.argmin(self.result['Magnitude'])]

        print("Guess frequency", self.frequency_guess)

        mp.update_freq(original_freq)  # GC temp fix

        if update:
            print("Updating freq to ", self.frequency_guess)
            mp = dut_qubit.get_default_measurement_prim_int()
            mp.update_freq(self.frequency_guess)

    # @register_browser_function(available_after=(run,))
    # def plot_magnitude(self):
    #    return
    #    args = self.retrieve_args(self.run)
    #    f = np.arange(args['start'], args['stop'], args['step'])
    #    fig, ax = plt.subplots()
    #    ax.plot(f, self.result['Magnitude'])
    #    plt.title()
    #    plt.show()

    # '@register_browser_function(available_after=(run,))
    # 'def plot_magnitude_plotly(self):
    #    args = self.retrieve_args(self.run)
    #
    #        fig = go.Figure()
    #
    #        f = np.arange(args['start'], args['stop'], args['step'])
    #        fig.add_trace(go.Scatter(x=f, y=self.result['Magnitude'],
    #                                 mode='lines',
    #                                 name='Magnitude'))
    #
    #        fig.update_layout(title='Resonator spectroscopy',
    #                          xaxis_title='Frequency [MHz]',
    #                          yaxis_title='Magnitude', plot_bgcolor='white')
    #
    #        fig.show()
    #
    #        # @register_browser_function(available_after=(run,))
    #        # def plot_magnitude_fit(self):
    #        args = self.retrieve_args(self.run)
    #        f = np.arange(args['start'], args['stop'], args['step'])
    #
    #        def root_lorentzian(f, f0, Q, amp, baseline):
    #            return abs((amp / (1 + (2j * Q * (f - f0) / f0)))) + baseline
    #
    #        def leastsq(x, f, z):
    #            f0, Q, amp, baseline = x
    #            fit = root_lorentzian(f, f0, Q, amp, baseline)
    #            return np.sum((fit - z) ** 2)
    #
    #        z = self.result['Magnitude']
    #        f0, Q, amp, baseline = f[np.argmax(z)], 5000., max(z) - min(z), min(z)
    #        result = so.minimize(leastsq, np.array([f0, Q, amp, baseline]), args=(f, z),
    #                             tol=1.0e-20)  # method='Nelder-Mead',
    #        f0, Q, amp, baseline = result.x
    #        fig, ax = plt.subplots()
    #        ax.scatter(f, self.result['Magnitude'])
    #        ax.plot(f, root_lorentzian(f, f0, Q, amp, baseline))
    #        ax.set_ylim(min(z), max(z))
    #        # ax.plot(f, root_lorentzian(f, f[np.argmax(self.result['Magnitude'])], 1000, 0.001, 0))
    #        plt.title('f0:%s, Q:%s, amp:%s, base:%s' % (f0, Q, amp, baseline))
    #        plt.show()

    @register_browser_function(available_after=(run,))
    def plot_magnitude_plotly(self):
        args = self.retrieve_args(self.run)
        f = np.arange(args['start'], args['stop'], args['step'])

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=f, y=self.result['Magnitude'],
                                 mode='lines',
                                 name='Magnitude'))

        fig.update_layout(title='Resonator spectroscopy magnitude',
                          xaxis_title='Frequency [MHz]',
                          yaxis_title='Magnitude', plot_bgcolor='white')

        fig.show()

    @register_browser_function(available_after=(run,))
    def plot_phase_plotly(self):
        args = self.retrieve_args(self.run)
        f = np.arange(args['start'], args['stop'], args['step'])
        phase_trace = self.result['Phase']
        phase_trace_mod = self.UnwrapPhase(phase_trace)

        fig = go.Figure()

        f = np.arange(args['start'], args['stop'], args['step'])

        fig.add_trace(go.Scatter(x=f, y=phase_trace_mod,
                                 mode='lines',
                                 name='Phase'))

        fig.add_trace(go.Scatter(x=(f[:-1] + f[1:]) / 2, y=(phase_trace_mod[1:] - phase_trace_mod[:-1]) / args['step'],
                                 mode='lines',
                                 name='Phase derivative'))

        fig.update_layout(title='Resonator spectroscopy phase',
                          xaxis_title='Frequency [MHz]',
                          yaxis_title='Phase', plot_bgcolor='white')

        fig.show()

    def root_lorentzian(self, f, f0, Q, amp, baseline):
        return abs((amp / (1 + (2j * Q * (f - f0) / f0)))) + baseline

    def fit_phase_diff(self):
        args = self.retrieve_args(self.run)
        f = np.arange(args['start'], args['stop'], args['step'])
        phase_trace = self.result['Phase']
        phase_trace_mod = self.UnwrapPhase(phase_trace)

        def leastsq(x, f, z):
            f0, Q, amp, baseline = x
            fit = self.root_lorentzian(f, f0, Q, amp, baseline)
            return np.sum((fit - z) ** 2)

        z = (phase_trace_mod[1:] - phase_trace_mod[:-1]) / args['step']

        direction = 1 if np.abs(np.max(z)) > np.abs(np.min(z)) else -1

        f = (f[:-1] + f[1:]) / 2

        f0, amp, baseline = f[np.argmax(z * direction)], max(z) - min(z), min(z * direction)

        half_cut = z * direction - baseline - amp / 2

        f_diff = (f[:-1] + f[1:]) / 2
        turn_point = np.argwhere(half_cut[:-1] * half_cut[1:] < 0)

        kappa_guess = f_diff[turn_point[1]] - f_diff[turn_point[0]]

        Q_guess = f0 / kappa_guess

        # print("direction", direction, "f0, Q,kappa_guess, amp, baseline", f0, Q_guess, kappa_guess, amp, baseline)

        result = so.minimize(leastsq, np.array([f0, Q_guess, amp, baseline], dtype=object), args=(f, z * direction),
                             tol=1.0e-20)  # method='Nelder-Mead',
        f0, Q, amp, baseline = result.x

        return z, f0, Q, amp, baseline, direction

    @register_browser_function(available_after=(run,))
    def plot_phase_diff_fit_plotly(self):

        z, f0, Q, amp, baseline, direction = self.fit_phase_diff()

        args = self.retrieve_args(self.run)
        f = np.arange(args['start'], args['stop'], args['step'])
        f_interpolate = np.arange(args['start'], args['stop'], args['step'] / 5)

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(x=f_interpolate, y=self.root_lorentzian(f_interpolate, f0, Q, amp, baseline) * direction,
                       mode='lines',
                       name='Lorentzian fit'))

        fig.add_trace(go.Scatter(x=f, y=z,
                                 mode='markers',
                                 name='Phase derivative'))

        fig.update_layout(title='Resonator spectroscopy phase fitting',
                          xaxis_title='Frequency [MHz]',
                          yaxis_title='Phase', plot_bgcolor='white')

        print('Phase diff fit f0:%s, Q:%s, amp:%s, base:%s kappa:%f' % (f0, Q, amp, baseline, f0 / Q))
        fig.show()

    # @register_browser_function(available_after=(run,))
    # def plot_phase_differentiated(self):
    #    args = self.retrieve_args(self.run)
    #    f = np.arange(args['start'], args['stop'], args['step'])
    #    fig, ax = plt.subplots()
    #    phase_trace = self.result['Phase']
    #    phase_trace_mod = self.DifferentiateTrace(self.UnwrapPhase(phase_trace))
    #    ax.plot(f, phase_trace_mod)
    #    plt.show()

    # @register_browser_function(available_after=(run,))
    # def plot_phase_diff_fit(self):
    #    args = self.retrieve_args(self.run)
    #    f = np.arange(args['start'], args['stop'], args['step'])

    #    def root_lorentzian(f, f0, Q, amp, baseline):
    #        return abs((amp / (1 + (2j * Q * (f - f0) / f0)))) + baseline

    #    def leastsq(x, f, z):
    #        f0, Q, amp, baseline = x
    #        fit = root_lorentzian(f, f0, Q, amp, baseline)
    #        return np.sum((fit - z) ** 2)

    #    phase_trace = self.result['Phase']
    #    z = -self.DifferentiateTrace(self.UnwrapPhase(phase_trace))
    #    f0, Q, amp, baseline = f[np.argmax(z)], 5000., max(z) - min(z), min(z)
    #    result = so.minimize(leastsq, np.array([f0, Q, amp, baseline]), args=(f, z),
    #                         tol=1.0e-20)  # method='Nelder-Mead',
    #    f0, Q, amp, baseline = result.x
    #    fig, ax = plt.subplots()
    #    ax.scatter(f, -z)
    #    ax.plot(f, -root_lorentzian(f, f0, Q, amp, baseline))
    #    ax.set_ylim(min(-z), max(-z))
    #    # ax.plot(f, root_lorentzian(f, f[np.argmax(self.result['Magnitude'])], 1000, 0.001, 0))
    #    plt.title('Phase diff fit f0:%s, Q:%s, amp:%s, base:%s kappa:%f' % (f0, Q, amp, baseline, f0 / Q))
    #    print('Phase diff fit f0:%s, Q:%s, amp:%s, base:%s kappa:%f' % (f0, Q, amp, baseline, f0 / Q))
    #    plt.show()

    def UnwrapPhase(self, trace):
        return np.unwrap(trace)

    def DifferentiateTrace(self, trace):
        return np.gradient(trace)
