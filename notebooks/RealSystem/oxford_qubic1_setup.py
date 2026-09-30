"""LeeQ setup for Oxford QubiC1 through the local RPC forward.

Importing this module does not register a setup and does not run an experiment.
Call :func:`initialize_oxford_qubic1_setup` explicitly before constructing a
LeeQ experiment.
"""

from leeq.chronicle import Chronicle
from leeq.experiments import setup
from leeq.setups.qubic_lbnl_setups import QubiCSingleBoardRemoteRPCSetup


QUBIC1_RPC_URI = "http://127.0.0.1:19095"
QUBIC2_RPC_URI = "http://127.0.0.1:29095"
READOUT_MIXER_LO_MHZ = 15_000.0


class OxfordQubiC1Setup(QubiCSingleBoardRemoteRPCSetup):
    """Oxford fridge setup for QubiC1 (``qubit80_1``)."""

    @staticmethod
    def _readout_frequency_mixing_callback(parameters: dict) -> dict:
        """Convert physical readout MHz to the QubiC IF convention."""
        if "freq" not in parameters:
            return parameters

        modified_parameters = parameters.copy()
        modified_parameters["freq"] = (
            READOUT_MIXER_LO_MHZ - parameters["freq"]
        )
        return modified_parameters

    def __init__(self) -> None:
        super().__init__(
            name="oxford_qubic1_fridge",
            rpc_uri=QUBIC1_RPC_URI,
        )

        # LeeQ uses odd-numbered logical channels for readout. Register the
        # physical-to-IF conversion for every available QubiC readout channel.
        for core_index in range(8):
            readout_channel = 2 * core_index + 1
            self._status.register_compile_lpb_callback(
                channel=readout_channel,
                callback=self._readout_frequency_mixing_callback,
            )


def initialize_oxford_qubic1_setup(
    chronicle_name: str = "",
) -> OxfordQubiC1Setup:
    """Create and register QubiC1 as LeeQ's default experiment setup.

    This constructs the client-side RPC proxy but does not submit an RPC call.
    LeeQ experiment construction is the later execution boundary.
    """
    Chronicle().start_log(name=chronicle_name)
    experiment_setup = OxfordQubiC1Setup()
    setup().register_setup(experiment_setup)
    return experiment_setup


class OxfordQubiC2Setup(OxfordQubiC1Setup):
    """Oxford bench setup for QubiC2 (``qubic81``)."""

    def __init__(self) -> None:
        QubiCSingleBoardRemoteRPCSetup.__init__(
            self,
            name="oxford_qubic2_bench",
            rpc_uri=QUBIC2_RPC_URI,
        )
        for core_index in range(8):
            readout_channel = 2 * core_index + 1
            self._status.register_compile_lpb_callback(
                channel=readout_channel,
                callback=self._readout_frequency_mixing_callback,
            )


def initialize_oxford_qubic2_setup(
    chronicle_name: str = "",
) -> OxfordQubiC2Setup:
    """Create and register QubiC2 as LeeQ's default experiment setup."""
    Chronicle().start_log(name=chronicle_name)
    experiment_setup = OxfordQubiC2Setup()
    setup().register_setup(experiment_setup)
    return experiment_setup
