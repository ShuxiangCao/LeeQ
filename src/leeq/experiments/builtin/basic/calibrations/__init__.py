"""Compatibility imports for :mod:`leeq.experiments.calibrations`."""

from leeq.experiments._compat import reexport_package

reexport_package(
    globals(),
    "leeq.experiments.calibrations",
    [
        "drag",
        "pingpong",
        "qubit_spectroscopy",
        "rabi",
        "ramsey",
        "residual_zz",
        "resonator_spectroscopy",
        "state_discrimination",
        "state_discrimination.assignment",
        "state_discrimination.gaussian_mixture",
        "state_discrimination.windowing_functions",
        "transmon_tuneup",
        "two_tone_spectroscopy",
    ],
)
