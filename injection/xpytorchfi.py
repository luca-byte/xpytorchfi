import logging
import struct
import torch
from pytorchfi.core import FaultInjection
from pytorchfi.neuron_error_models import single_bit_flip_func
from typing_extensions import override
import copy
import numpy as np

logger = logging.getLogger("XPFI")


class BitFlipWeights:
    """Class to inject bit-flip faults into weights during model inference."""

    def __init__(
        self, bitmasks: list[str], layers: list[int], save_stats: bool = False
    ):
        """
        Class to inject bit-flip faults into weights during model inference. Multiple faults can be injected,
        in this case, the faults MUST be ordered according to the increasing layer index.
        Args:
            bitmasks (list[str]): List of bitmasks to apply for fault injection. Faults MUST be ordered
            according to the increasing layer index.
            layers (list[int]): List of layer indices corresponding to the bitmasks.
            save_stats (bool, optional): Whether to save fault injection statistics. Defaults to False.
        """
        self.bitmasks = bitmasks
        self.counter = 0
        self.layers = layers
        self.save_stats = save_stats
        self.injected_faults: list[dict] = []

    def __call__(self, data, location):
        """
        Injects a bit-flip fault into the weight at the specified location.
        Args:
            data (torch.Tensor): The weight tensor.
            location (tuple): The location (K, C, H, W) of the weight to be corrupted.
        Returns:
            float: The corrupted weight value.
        """

        # Convert the float to its 32-bit integer representation
        orig_data = data[location].item()
        data_32bit = int(XSingleBitFlipFI.float_to_hex(data[location].item()), 16)

        # Apply the bitmask to flip the specified bit
        injmask = self.bitmasks[self.counter]
        self.counter += 1
        corrupt_32bit = data_32bit ^ int(injmask)
        corrupt_val = self.int_to_float(corrupt_32bit)

        # Log the fault injection details
        self.log_msg = f"F_descriptor: Layer:{self.layers[self.counter]}, (K, C, H, W):{location}, BitMask:{injmask}, Ffree_Weight:{data_32bit}, Faulty_weight:{corrupt_32bit}"
        logger.info(self.log_msg)

        # Save fault injection statistics if required
        if self.save_stats:
            fsim_dict = {
                "Layer": self.layers[self.counter],
                "kernel": location[0],
                "channel": location[1],
                "row": location[2],
                "col": location[3],
                "BitMask": injmask,
                "Ffree_Weight": data_32bit,
                "Faulty_weight": corrupt_32bit,
                "Abs_error": (orig_data - corrupt_val),
            }
            self.injected_faults.append(fsim_dict)

        return corrupt_val


class XFaultInjection(FaultInjection):
    input_size = []

    @override
    def _save_output_size(self, module, input_val, output):
        shape = list(output.size())
        dim = len(shape)
        self.input_size.append(input_val[0].size())
        self.layers_type.append(type(module))
        self.layers_dim.append(dim)
        self.output_size.append(shape)

    # Bug fix for "all" layers from PR #96 of pytorchfi
    @override
    def print_pytorchfi_layer_summary(self):
        summary_str = (
            "============================ PYTORCHFI INIT SUMMARY =============================="
            + "\n\n"
        )

        summary_str += "Layer types allowing injections:\n"
        summary_str += (
            "----------------------------------------------------------------------------------"
            + "\n"
        )
        for l_type in self._inj_layer_types:
            summary_str += "{:>5}".format("- ")
            substring = str(l_type).split(".")[-1].split("'")[0]
            summary_str += substring + "\n"
        summary_str += "\n"

        summary_str += "Model Info:\n"
        summary_str += (
            "----------------------------------------------------------------------------------"
            + "\n"
        )

        summary_str += "   - Shape of input into the model: ("
        for dim in self._input_shape:
            summary_str += str(dim) + " "
        summary_str += ")\n"

        summary_str += "   - Batch Size: " + str(self.batch_size) + "\n"
        summary_str += "   - CUDA Enabled: " + str(self.use_cuda) + "\n\n"

        summary_str += "Layer Info:\n"
        summary_str += (
            "----------------------------------------------------------------------------------"
            + "\n"
        )
        line_new = "{:>5}  {:>20}  {:>10} {:>20} {:>20}".format(
            "Layer #", "Layer type", "Dimensions", "Weight Shape", "Output Shape"
        )
        summary_str += line_new + "\n"
        summary_str += (
            "----------------------------------------------------------------------------------"
            + "\n"
        )
        for layer, _dim in enumerate(self.output_size):
            weight_shape = (
                list(self.weights_size[layer]) if self.weights_size[layer] else None
            )
            line_new = "{:>5}  {:>20}  {:>10} {:>20} {:>20}".format(
                layer,
                str(self.layers_type[layer]).split(".")[-1].split("'")[0],
                str(self.layers_dim[layer]),
                str(weight_shape),
                str(self.output_size[layer]),
            )
            summary_str += line_new + "\n"

        summary_str += (
            "=================================================================================="
            + "\n"
        )

        logging.info(summary_str)
        return summary_str

    def _declare_berw(self, function, fault_description, ber, trial, bitmask=None):
        """Injects bit error rate (BER) faults into the weights of the model."""
        self._reset_fault_injection_state()

        # Create a deep copy of the original model to inject faults
        self.corrupted_model = copy.deepcopy(self.original_model)
        current_weight_layer = 0
        induced_errors = []

        def corrupt_weights(layer, corrupt_idxs, fault_description_per_layer, bitmasks):
            """
            Applies the custom fault injection function to the specified weight indices of a layer.
            """
            for idxs, bitmask in zip(corrupt_idxs, bitmasks):
                with torch.no_grad():
                    corrupt_val, induced_error = function(
                        layer.weight,
                        tuple(idxs),
                        bitmask,
                        ber,
                        len(fault_description_per_layer),
                        trial,
                        induced_errors,
                    )
                    # Inject the corrupted value into the weight tensor
                    layer.weight[tuple(idxs)] = corrupt_val
                    induced_errors.append(induced_error)

        # Iterate over all layers of the model
        for layer in self.corrupted_model.modules():
            # Filter the fault description for the current layer
            fault_description_per_layer = fault_description.query(
                f"layer=={current_weight_layer}"
            )
            # Select indices to corrupt based on layer type
            if isinstance(layer, torch.nn.modules.conv.Conv2d):
                corrupt_idxs = np.array(
                    fault_description_per_layer[["kernel", "channel", "row", "col"]],
                    dtype=int,
                )
            elif isinstance(layer, torch.nn.modules.linear.Linear):
                corrupt_idxs = np.array(
                    fault_description_per_layer[["kernel", "channel"]], dtype=int
                )
            else:
                continue

            # Distinguish between fixed and variable bitmask
            if bitmask is None:
                bitmasks = np.array(fault_description_per_layer[["bitmask"]], dtype=int)
            else:
                bitmasks = [bitmask] * len(corrupt_idxs)

            # Inject faults into the selected weights
            corrupt_weights(layer, corrupt_idxs, fault_description_per_layer, bitmasks)
            current_weight_layer += 1

        return self.corrupted_model

    def declare_ber_weight_fault_injection(self, **kwargs):
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

        Returns:
            torch.nn.Module: The corrupted model with injected faults.

        Raises:
            ValueError: If no kwargs are provided or required parameters are missing.
        """
        if not kwargs:
            raise ValueError("Please specify an injection or injection function")

        fault_description = kwargs.get("fault_description")
        bitmask = kwargs.get("bitmask")
        ber = kwargs.get("ber")
        trial = kwargs.get("trial")
        function = kwargs.get("function")

        self.corrupted_model = self._declare_berw(
            function, fault_description, ber, trial, bitmask
        )

    def declare_var_bit_ber_weight_fault_injection(self, **kwargs):
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
        Returns:
            torch.nn.Module: The corrupted model with injected faults.
        Raises:
            ValueError: If no kwargs are provided or required parameters are missing.
        """
        if not kwargs:
            raise ValueError("Please specify an injection or injection function")

        fault_description = kwargs.get("fault_description")
        ber = kwargs.get("ber")
        trial = kwargs.get("trial")
        custom_function = kwargs.get("function")

        return self._declare_berw(
            custom_function, fault_description, ber, trial, bitmask=None
        )

    def get_all_weights_sizes(self):
        """Returns the sizes of all weights in the model."""
        return self.weights_size


class XSingleBitFlipFI(single_bit_flip_func):
    @staticmethod
    def _float_to_hex(f):
        """Convert a float to its hexadecimal representation."""
        h = hex(struct.unpack("<I", struct.pack("<f", f))[0])
        return h[2 : len(h)]

    @staticmethod
    def _hex_to_float(h):
        """Convert a hexadecimal representation to a float."""
        return float(struct.unpack(">f", struct.pack(">I", int(h, 16)))[0])

    @staticmethod
    def _int_to_float(h):
        """Convert an integer representation to a float."""
        return float(struct.unpack(">f", struct.pack(">I", h))[0])

    @staticmethod
    def _max_num_bits(data):
        """Return the maximum number of bits for the given data's dtype."""
        return data.dtype.itemsize * 8

    def _bit_flip_value(self, orig_values, bit_pos):
        """Flip a specific bit in the float value."""
        # Convert tensor to float values
        orig_data = orig_values.float()

        # Generate injection mask for bit flip
        injmask = 2**bit_pos

        data_32bit = orig_data.view(torch.int32)
        corrupt_32bit = torch.bitwise_xor(data_32bit, injmask.type(torch.int32))
        corrupt_val = corrupt_32bit.view(torch.float)
        return corrupt_val

    # TODO: Are these two functions functionally identical?
    def single_bit_flip_across_batch(self, module, input_val, output):
        corrupt_conv_set = self.corrupt_layer
        bit_flip_pos = self.get_conv_max(0)
        logger.info(f"Current layer: {self.current_layer}")
        # logger.info(f"Range_max: {range_max}")

        if type(corrupt_conv_set) is list:
            inj_list = list(
                filter(
                    lambda x: corrupt_conv_set[x] == self.current_layer,
                    range(len(corrupt_conv_set)),
                )
            )
            for i in inj_list:
                # print(self.output_size[self.current_layer])
                if i < output.shape[0]:
                    self.assert_injection_bounds(index=i)
                    prev_value = output[self.corrupt_batch[i]][self.corrupt_dim[0][i]][
                        self.corrupt_dim[1][i]
                    ][self.corrupt_dim[2][i]]

                    # rand_bit = random.randint(0, self._max_num_bits(prev_value) - 1)
                    # rand_bit = random.randint(0, bit_flip_pos)
                    rand_bit = bit_flip_pos
                    logger.info(f"Random Bit: {rand_bit}")
                    new_value = self._bit_flip_value(prev_value, rand_bit)

                    output[self.corrupt_batch[i]][self.corrupt_dim[0][i]][
                        self.corrupt_dim[1][i]
                    ][self.corrupt_dim[2][i]] = new_value

        else:
            if self.current_layer == corrupt_conv_set:
                prev_value = output[self.corrupt_batch][self.corrupt_dim[0]][
                    self.corrupt_dim[1]
                ][self.corrupt_dim[2]]

                # rand_bit = random.randint(0, self._max_num_bits(prev_value) - 1)
                # rand_bit = random.randint(0, bit_flip_pos)
                rand_bit = bit_flip_pos
                logger.info(f"Random Bit: {rand_bit}")
                new_value = self._bit_flip_value(prev_value, rand_bit)

                output[self.corrupt_batch][self.corrupt_dim[0]][self.corrupt_dim[1]][
                    self.corrupt_dim[2]
                ] = new_value

        self.update_layer()
        if self.current_layer >= len(self.output_size):
            self.reset_current_layer()

    def single_bit_flip_across_batch_tensor(self, module, input_val, output):
        corrupt_conv_set = self.corrupt_layer
        bit_flip_pos = self.get_conv_max(0)
        logger.info(f"Current layer: {self.current_layer}")
        # logger.info(f"Range_max: {range_max}")

        if type(corrupt_conv_set) is list:
            inj_list = list(
                filter(
                    lambda x: corrupt_conv_set[x] == self.current_layer,
                    range(len(corrupt_conv_set)),
                )
            )
            if len(inj_list) > 0:
                dim = len(list(output.size()))
                indices_dim0 = torch.tensor(self.corrupt_batch)  # batch
                indices_dim1 = torch.tensor(
                    self.corrupt_dim[0][inj_list[0] : inj_list[0] + len(inj_list)]
                )  # channel
                if dim > 2:
                    indices_dim2 = torch.tensor(
                        self.corrupt_dim[1][inj_list[0] : inj_list[0] + len(inj_list)]
                    )  # row
                    indices_dim3 = torch.tensor(
                        self.corrupt_dim[2][inj_list[0] : inj_list[0] + len(inj_list)]
                    )  # colum

                for i in range(output.shape[0]):
                    # self.assert_injection_bounds(index=i)
                    if dim > 2:
                        prev_value = output[i, indices_dim1, indices_dim2, indices_dim3]
                    else:
                        prev_value = output[i, indices_dim1]

                    rand_bit = torch.tensor([bit_flip_pos], device=output.device.type)

                    logger.info(f"Random Bit: {bit_flip_pos}")

                    new_value = self._bit_flip_value(prev_value, rand_bit)
                    if dim > 2:
                        output[i, indices_dim1, indices_dim2, indices_dim3] = new_value
                    else:
                        output[i, indices_dim1] = new_value

        else:
            if self.current_layer == corrupt_conv_set:
                dim = len(list(output.size()))
                indices_dim0 = torch.tensor(self.corrupt_batch)  # batch
                indices_dim1 = torch.tensor(self.corrupt_dim[0])  # channel
                if dim > 2:
                    indices_dim2 = torch.tensor(self.corrupt_dim[1])  # row
                    indices_dim3 = torch.tensor(self.corrupt_dim[2])  # colum

                if dim > 2:
                    prev_value = output[i, indices_dim1, indices_dim2, indices_dim3]
                else:
                    prev_value = output[i, indices_dim1]
                rand_bit = torch.tensor([bit_flip_pos], device=output.device.type)

                logger.info(f"Random Bit: {bit_flip_pos}")

                new_value = self._bit_flip_value(prev_value, rand_bit)
                if dim > 2:
                    output[i, indices_dim1, indices_dim2, indices_dim3] = new_value
                else:
                    output[i, indices_dim1] = new_value

        self.update_layer()
        if self.current_layer >= len(self.output_size):
            self.reset_current_layer()
