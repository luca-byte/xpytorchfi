from xpytorchfi import XFaultInjection
from xpytorchfi import BitFlipWeights
from typing import List, Dict
import logging


class FIFramework:
    def __init__(self, pfi_model: XFaultInjection):
        logging.getLogger().setLevel(logging.WARNING)
        self.pfi_model = pfi_model

    def inject_bfw_fault(self, fault: List[Dict]):
        """
        Injects a bit-flip fault into a specific weight of the model.

        Args:
        fault (List[Dict]): List containing a single fault dictionary with keys:
            'layer', 'kernel', 'channel', 'row', 'col', 'bitmask'.
        """
        fault_info = fault[0]
        layer = fault_info["layer"]
        kernel = fault_info["kernel"]
        channel = fault_info["channel"]
        row = fault_info["row"]
        col = fault_info["col"]
        inj_mask = fault_info["bitmask"]

        bfw = BitFlipWeights(bitmasks=[inj_mask], layers=[layer], save_stats=False)

        self.faulty_model = self.pfi_model.declare_weight_fault_injection(
            function=bfw,
            layer_num=[layer],
            k=[kernel],
            dim1=[channel],
            dim2=[row],
            dim3=[col],
        )
        self.faulty_model.eval()
