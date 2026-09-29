"""Local setup construction for Huracan's aa01c78f two-core configuration."""

import json
import hashlib
from pathlib import Path

import numpy as np

from leeq.setups.qubic_executable_setups import QubiCSingleBoardExecutableRPCSetup


HURACAN_RPC_URI = "http://127.0.0.1:9095"


def create_huracan_setup(channel_config_path, *, rpc_uri=HURACAN_RPC_URI):
    """Read matching local channel metadata and construct an unregistered setup.

    No RPC calls are made. The caller supplies the aa01c78f channel_config.json
    and separately registers the setup and supplies calibrated LeeQ primitives
    when ready to run an experiment.
    """
    from distproc.hwconfig import FPGAConfig

    config = json.loads(Path(channel_config_path).read_text())
    expected = {f"Q{core}.{element}" for core in (0, 1) for element in ("qdrv", "rdrv", "rdlo")}
    channels = {name for name, value in config.items() if isinstance(value, dict)}
    if channels != expected:
        raise ValueError("Huracan aa01c78f requires exactly Q0/Q1 qdrv, rdrv and rdlo metadata")
    for name in expected:
        if config[name]["core_ind"] != int(name[1]):
            raise ValueError(f"Unexpected core assignment for {name}")
    frequency = config.get("fpga_clk_freq", 0)
    if not np.isfinite(frequency) or frequency <= 0:
        raise ValueError("channel_config.json must specify a positive fpga_clk_freq")
    setup = QubiCSingleBoardExecutableRPCSetup(
        name="Huracan-X6Y3", rpc_uri=rpc_uri, channel_configs=config,
        fpga_config=FPGAConfig(fpga_clk_period=1 / frequency),
        leeq_channel_to_qubic_channel={0: "Q0", 1: "Q0", 2: "Q1", 3: "Q1"},
        qubic_core_number=2)
    setup.channel_metadata = config
    setup.channel_config_path = str(Path(channel_config_path).resolve())
    setup.channel_config_sha256 = hashlib.sha256(Path(channel_config_path).read_bytes()).hexdigest()
    return setup
