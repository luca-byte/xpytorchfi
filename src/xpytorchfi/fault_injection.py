from .xpytorchfi import XFaultInjection, XSingleBitFlipFI
from .bitflips import BitFlipWeights, BitFlipWeightsBER
from .neuron_tails import generate_error_list_neurons_tails
from typing import List, Dict, Union
import logging
import torch
import random

logger = logging.getLogger("XPFI")


def _pick_index(size: int, fixed1: int = None) -> int:
    """Return a valid 0-based index, either fixed or randomly sampled."""
    if fixed1 is not None:
        return fixed1 - 1
    return random.randint(0, size - 1)


class FIFramework:
    pfi_model: Union[XFaultInjection, XSingleBitFlipFI]
    faulty_model: torch.nn.Module

    def __init__(
        self,
        device: torch.device,
        model: torch.nn.Module,
        input_shape: List[int],
        batch_size: int = 1,
        layer_types: List[torch.nn.Module] = [torch.nn.Conv2d],
        neuron_fault_injection: bool = False,
        ber: Union[int, bool] = False,
    ):
        """
        Initializes the fault injection framework.

        Args:
        device (torch.device): The device to run the model on.
        model (torch.nn.Module): The model to inject faults into.
        input_shape (List[int]): The shape of the input tensor (C, H, W).
        batch_size (int): The size of the batch.
        layer_types (List[torch.nn.Module]): The types of layers to inject faults into (supports Conv2d, Linear).
        neuron_fault_injection (bool): Whether to inject neuron faults.
        ber (int | bool): The bit error rate intended as an integer counter (number of bit flips). If False, no bit error rate is applied.
        """

        use_cuda = device.type == "cuda"
        if neuron_fault_injection:
            self.pfi_model = XSingleBitFlipFI(
                model,
                batch_size,
                input_shape,
                use_cuda=use_cuda,
                layer_types=layer_types,
                bits=8,
            )
        else:
            self.pfi_model = XFaultInjection(
                model,
                batch_size,
                input_shape,
                use_cuda=use_cuda,
                layer_types=layer_types,
                BER=ber,
            )
        self.pfi_model.print_pytorchfi_layer_summary()

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

    def inject_bf_neuron_fault(self, fault: List[Dict]):
        """
        Inject single-bit flip faults into neuron activations for a selected layer range.

        It builds the list of target neuron coordinates, configures the bit position
        for the injector, declares neuron fault injection on the model, and switches
        the resulting model to evaluation mode.

        The function expects `fault` to be a list containing one dictionary with:
            - layer_start (int): first layer index (inclusive).
            - layer_stop (int): last layer index (inclusive/exclusive based on generator behavior).
            - block_fault_rate (float): probability/rate used to select faulty blocks.
            - neuron_fault_rate (float): probability/rate used to select neurons inside faulty blocks.
            - size_tail_y (int): block height in the activation map.
            - size_tail_x (int): block width in the activation map.
            - bit_faulty_pos (int): bit position to flip.

        Args:
        fault (List[Dict]): List containing a single fault dictionary with keys:
            'layer_start', 'layer_stop', 'block_fault_rate', 'neuron_fault_rate',
            'size_tail_y', 'size_tail_x', 'bit_faulty_pos'.
        """

        fault_info = fault[0]
        layer_start = fault_info["layer_start"]
        layer_stop = fault_info["layer_stop"]
        block_fault_rate = fault_info["block_fault_rate"]
        neuron_fault_rate = fault_info["neuron_fault_rate"]
        size_tail_y = fault_info["size_tail_y"]
        size_tail_x = fault_info["size_tail_x"]
        bit_faulty_pos = fault_info["bit_faulty_pos"]

        # Generate target neuron locations and batch order according to the configured rates.
        (random_layers, random_c, random_h, random_w, batch_order, fault_info) = (
            generate_error_list_neurons_tails(
                self.pfi_model,
                layer_i=layer_start,
                layer_n=layer_stop,
                block_error_rate=block_fault_rate,
                neuron_fault_rate=neuron_fault_rate,
                tail_bloc_y=size_tail_y,
                tail_bloc_x=size_tail_x,
            )
        )

        assert isinstance(self.pfi_model, XSingleBitFlipFI), (
            "Neuron FI requires FIFramework(..., neuron_fault_injection=True)"
        )

        # Configure which bit position will be flipped during neuron injection.
        self.pfi_model.set_conv_max([bit_faulty_pos])

        self.faulty_model = self.pfi_model.declare_neuron_fault_injection(
            layer_num=random_layers,
            batch=batch_order,
            dim1=random_c,
            dim2=random_h,
            dim3=random_w,
            function=self.pfi_model.single_bit_flip_across_batch_tensor,
        )

        self.faulty_model.eval()

    def inject_ber_bfw_fault(self, fault_description, ber, trial, bitmask):
        """
        Injects bit error rate (BER) faults into the weights of the model.

        This method creates a corrupted copy of the original model, injecting faults into weights
        according to a fault description DataFrame. The injection can be customized via a user-defined
        function passed in kwargs. For each layer, the function identifies the weights to corrupt
        and applies the specified fault (e.g., bit flip) using the provided bitmask and BER parameters.

        Args:
            function (callable): Custom function to apply the fault injection.
            fault_description (pd.DataFrame): DataFrame containing fault locations and parameters.
            bitmask (int or array-like): Bitmask specifying which bits to flip.
            ber (float): Bit error rate for the injection.
            trial (int): Trial number for the injection.
        """

        bfw = BitFlipWeightsBER(save_stats=True)

        self.faulty_model = self.pfi_model.declare_ber_weight_fault_injection(
            function=bfw,
            fault_description=fault_description,
            ber=ber,
            trial=trial,
            bitmask=bitmask,
        )

        self.faulty_model.eval()

    def ber_var_bit_flip_weight_inj(self, fault_description, ber, trial):
        """
        Injects bit error rate (BER) faults into the weights of the model with variable bitmask.

        This method creates a corrupted copy of the original model, injecting faults into weights
        according to a fault description DataFrame. The injection can be customized via a user-defined
        function passed in kwargs. For each layer, the function identifies the weights to corrupt
        and applies the specified fault (e.g., bit flip) using the BER parameters. The bitmask can vary for each fault.

        Args:
            function (callable): Custom function to apply the fault injection.
            fault_description (pd.DataFrame): DataFrame containing fault locations and parameters.
            ber (float): Bit error rate for the injection.
            trial (int): Trial number for the injection.
        """
        bfw = BitFlipWeightsBER(save_stats=True)

        self.faulty_model = self.pfi_model.declare_var_bit_ber_weight_fault_injection(
            BitFlip=bfw,
            fault_description=fault_description,
            ber=ber,
            trial=trial,
        )

        self.faulty_model.eval()

    def BER_weight_inj(
        self,
        BER: int,
        layer: int = None,
        kK: int = None,
        kC: int = None,
        kH: int = None,
        kW: int = None,
        inj_mask: int = None,
        timeout: int = 1000,
    ):
        """
        Inject multiple unique bit-flip faults into model weights.

        This method randomly samples weight coordinates and bit positions until
        `BER` unique faults are collected (or a timeout is reached), then applies
        the injection through `declare_weight_fault_injection`.

        Args:
            BER (int): Number of unique bit-flip faults to generate.
            layer (Optional[int]): Fixed 1-based layer index. If None, layer is sampled randomly.
            kK (Optional[int]): Fixed 1-based kernel index. If None, sampled randomly.
            kC (Optional[int]): Fixed 1-based channel index. If None, sampled randomly.
            kH (Optional[int]): Fixed 1-based row index. If None, sampled randomly.
            kW (Optional[int]): Fixed 1-based column index. If None, sampled randomly.
            inj_mask (Optional[int]): Fixed 1-based bit position in [1, 32]. If None, sampled randomly.
            timeout (int): Multiplier used to cap random attempts (`timeout * BER`).

        Notes:
            - Duplicate faults are discarded using a set.
            - If fewer than `BER` unique faults are found before timeout, a warning is logged.
        """

        layers = []
        kernels = []
        channels = []
        rows = []
        cols = []

        selected_faults: set[tuple] = set()
        counter = 0
        TIMEOUT = timeout * BER
        while len(selected_faults) < BER and counter < TIMEOUT:
            layer = _pick_index(self.pfi_model.get_total_layers(), fixed1=layer)
            weight_shape = self.pfi_model.get_weights_size(layer)
            kernel = _pick_index(weight_shape[0], fixed1=kK)
            channel = _pick_index(weight_shape[1], fixed1=kC)
            row = _pick_index(weight_shape[2], fixed1=kH)
            col = _pick_index(weight_shape[3], fixed1=kW)
            bitmask = _pick_index(32, fixed1=inj_mask)

            fault_tuple = (layer, kernel, channel, row, col, bitmask)
            if fault_tuple not in selected_faults:
                layers.append(layer)
                kernels.append(kernel)
                channels.append(channel)
                rows.append(row)
                cols.append(col)
                selected_faults.add(fault_tuple)

            counter += 1

        if len(selected_faults) < BER:
            logger.warning(
                f"Only {len(selected_faults)} unique faults could be generated after {counter} attempts. Consider increasing the TIMEOUT or adjusting the parameters."
            )

        bfw = BitFlipWeightsBER(save_stats=False)

        self.faulty_model = self.pfi_model.declare_weight_fault_injection(
            function=bfw,
            layer_num=layers,
            k=kernels,
            dim1=channels,
            dim2=rows,
            dim3=cols,
        )

        self.faulty_model.eval()
