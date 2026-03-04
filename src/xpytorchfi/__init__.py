"""
xPytorchFI: A Fault Injection Framework for PyTorch
"""

__version__ = "0.1.0"

from .fault_injection import FIFramework
from .manager import FaultIterator
from .fault_generation import (
    generate_fault_list_sbfm,
    generate_fault_list_sbfm_fails,
    generate_fault_neurons_tailing,
    generate_fault_list_ber,
)
from .bitflips import BitFlipWeights, BitFlipWeightsBER
from .xpytorchfi import XFaultInjection, XSingleBitFlipFI
from .experiment import ExperimentRunner, ExperimentCallback

__all__ = [
    "FIFramework",
    "FaultIterator",
    "generate_fault_list_sbfm",
    "generate_fault_list_sbfm_fails",
    "generate_fault_neurons_tailing",
    "generate_fault_list_ber",
    "BitFlipWeights",
    "BitFlipWeightsBER",
    "XFaultInjection",
    "XSingleBitFlipFI",
    "ExperimentRunner",
    "ExperimentCallback",
]