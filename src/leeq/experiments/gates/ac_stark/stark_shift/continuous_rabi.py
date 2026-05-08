from .common import *

class StarkContinuesRabi(Experiment):
    @log_and_record
    def run(self, dut, amp, frequency, phase=0, rise=0.01, trunc=1.0, width_start=0, width_stop=4, width_step=0.01,
            initial_lpb=None):
        """
        Sweep time and find the initial guess of amplitude

        :return:
        """
        self.dut = dut
        self.frequency = frequency
        self.amp = amp
        self.phase = phase
        self.rise = rise
        self.trunc = trunc
        self.width_start = width_start
        self.width_stop = width_stop
        self.width_step = width_step

        c1 = self.dut.get_default_c1()

        pulse = c1['X'].clone()
        pulse.update_pulse_args(
            amp=self.amp, freq=self.frequency, phase=self.phase, shape='blackman_square', width=0,
            rise=self.rise, trunc=self.trunc)

        mprim = self.dut.get_measurement_prim_intlist(0)

        # Set up sweep parameters
        swpparams = [SweepParametersSideEffectFactory.func(
            pulse.update_pulse_args, {}, 'width'
        )]
        swp = Sweeper(
            np.arange,
            n_kwargs={'start': width_start, 'stop': width_stop, 'step': width_step},
            params=swpparams
        )

        if initial_lpb is not None:
            initial_lpb + pulse

        basic(pulse + mprim, swp=swp, basis="<z>")

        self.result = np.squeeze(mprim.result())

    @register_browser_function(available_after=(run,))
    def plot(self):
        """
        Plot the results.
        """
        args = self._get_run_args_dict()

        t = np.arange(args['width_start'], args['width_stop'], args['width_step'])

        data = self.result.squeeze()

        plt.scatter(t, data)
        plt.xlabel('Width [us]')
        plt.ylabel('<z>')

        return plt.gcf()
