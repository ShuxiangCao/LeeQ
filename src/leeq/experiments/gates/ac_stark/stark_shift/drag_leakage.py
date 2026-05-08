from .common import *

class StarkRepeatedGateDRAGLeakageCalibration(Experiment):
    @log_and_record
    def run(self, dut, amp, frequency, phase=0, rise=0.01, trunc=1.0, width=0.1, gate_count=40, initial_lpb=None,
            inv_alpha_start=None, inv_alpha_stop=None, sweep_count=20):
        """
        Sweep time and find the initial guess of amplitude

        :return:
        """
        self.dut = dut
        self.frequency = frequency
        self.amp = amp
        self.phase = phase
        self.width = width
        self.rise = rise
        self.trunc = trunc
        self.gate_count = gate_count

        c1 = self.dut.get_default_c1()

        pulse = c1['X'].clone()

        alpha = pulse.alpha

        if inv_alpha_start is None:
            inv_alpha_start = 1 / alpha - 0.006
        if inv_alpha_stop is None:
            inv_alpha_stop = 1 / alpha + 0.006

        def update_alpha(n):
            return pulse.update_parameters(alpha=1 / n)

        # Create a sweeper for the alpha parameter.
        self.sweep_values = np.linspace(inv_alpha_start, inv_alpha_stop, num=sweep_count)
        swp = Sweeper(
            self.sweep_values,
            params=[
                SweepParametersSideEffectFactory.func(
                    update_alpha,
                    argument_name='n',
                    kwargs={})])

        pulse.update_pulse_args(
            amp=self.amp, freq=self.frequency, phase=self.phase, shape='blackman_square', width=self.width,
            rise=self.rise, trunc=self.trunc)

        lpb = pulse

        mprim = self.dut.get_measurement_prim_intlist(0)

        sequence = LogicalPrimitiveBlockSerial([pulse] * (gate_count) + [mprim])

        if initial_lpb is not None:
            lpb = initial_lpb + sequence

        basic(lpb + mprim, swp=swp, basis="<z>")

        self.result = np.squeeze(mprim.result())

    @register_browser_function(available_after=(run,))
    def plot(self):
        """
        Plot the results.
        """
        self._get_run_args_dict()

        inv_alpha = self.sweep_values

        data = self.result.squeeze()

        plt.scatter(inv_alpha, data)
        plt.xlabel('Number of pulses')
        plt.ylabel('<z>')

        return plt.gcf()
