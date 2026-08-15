"""Load the authoritative STM32 wire-protocol implementation."""

from importlib import util
from pathlib import Path
import sys


def _protocol_path() -> Path:
    source_path = (
        Path(__file__).resolve().parents[3]
        / "firmware"
        / "tools"
        / "emg_protocol.py"
    )
    if source_path.is_file():
        return source_path

    from ament_index_python.packages import get_package_share_directory

    return (
        Path(get_package_share_directory("emg_intent_bridge"))
        / "python"
        / "emg_protocol.py"
    )


def _load_protocol():
    path = _protocol_path()
    module_name = "_emg_intent_bridge_protocol"
    spec = util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load EMG protocol module from {path}")
    module = util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


protocol = _load_protocol()

COMMAND_NAMES = protocol.COMMAND_NAMES
MAGIC = protocol.MAGIC
TYPE_INTENT = protocol.TYPE_INTENT
TYPE_RAW = protocol.TYPE_RAW
TYPE_ACTIVATION_STATE = protocol.TYPE_ACTIVATION_STATE
PacketParser = protocol.PacketParser
crc16 = protocol.crc16
decode_intent = protocol.decode_intent

ActivationState = protocol.ActivationState
ACTIVATION_SOURCE_DEFAULTS = protocol.ACTIVATION_SOURCE_DEFAULTS
ACTIVATION_SOURCE_HOST = protocol.ACTIVATION_SOURCE_HOST
SET_MODE_APPLY = protocol.SET_MODE_APPLY
SET_MODE_DEFAULTS = protocol.SET_MODE_DEFAULTS
SET_RESULT_ACCEPTED = protocol.SET_RESULT_ACCEPTED
SET_RESULT_NONE = protocol.SET_RESULT_NONE
decode_activation_state = protocol.decode_activation_state
encode_set_activation = protocol.encode_set_activation
