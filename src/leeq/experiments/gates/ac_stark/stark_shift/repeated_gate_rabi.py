from .common import *

class StarkRepeatedGateRabi(Experiment):
    @log_and_record
    def run(self, dut, amp, frequency, phase=0, rise=0.01, trunc=1.0, width=0, start_gate_number=0, gate_count=40,
            initial_lpb=None, alpha=1e9):
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
        self.start_gate_number = start_gate_number
        self.gate_count = gate_count

        pulse_count = np.arange(start_gate_number, start_gate_number + gate_count, 1)

        c1 = self.dut.get_default_c1()

        pulse = c1['X'].clone()
        pulse.update_pulse_args(
            amp=self.amp, freq=self.frequency, phase=self.phase, shape='blackman_square', width=self.width,
            rise=self.rise, trunc=self.trunc, alpha=alpha)

        lpb = pulse

        sequence_lpb = []
        mprim = self.dut.get_measurement_prim_intlist(0)

        for n in pulse_count:
            sequence = LogicalPrimitiveBlockSerial([pulse] * (n) + [mprim])
            sequence_lpb.append(sequence)

        lpb = LogicalPrimitiveBlockSweep(sequence_lpb)
        swp = sweeper.from_sweep_lpb(lpb)

        if initial_lpb is not None:
            lpb = initial_lpb + pulse

        basic(lpb, swp=swp, basis="<z>")

        self.result = np.squeeze(mprim.result())

    @register_browser_function(available_after=(run,))
    def plot(self):
        """
        Plot the results.
        """
        args = self._get_run_args_dict()

        t = np.arange(args['start_gate_number'], args['start_gate_number'] + args['gate_count'], 1)

        data = self.result.squeeze()

        plt.scatter(t, data)
        plt.xlabel('Number of pulses')
        plt.ylabel('<z>')

        return plt.gcf()
