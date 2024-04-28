from leeq.utils.compatibility import prims


class GeneralisedTomographyBase(object):

    def initialize_gate_lpbs(self, dut):
        pass

    def get_lpb(self, name):
        if isinstance(name, tuple) or isinstance(name, list):
            lpbs = [self.get_lpb(x) for x in name]
            lpb = prims.SeriesLPB(lpbs)
            return lpb

        return self._gate_lpbs[name]
