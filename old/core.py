"""pytorchfi.core contains the core functionality for fault injections"""

import copy
import logging
import warnings
from typing import List

import torch
import torch.nn as nn
from torchdistill.common.constant import def_logger
import numpy as np
logger = def_logger.getChild(__name__)
 

# logger=logging.getLogger("pytorchfi") 
# logger.setLevel(logging.DEBUG) 

class FaultInjection:
    def __init__(
        self,
        model,
        batch_size: int,
        input_shape: List[int] = None,
        layer_types=None,
        **kwargs,
    ):
        if not input_shape:
            input_shape = [3, 224, 224]
        if not layer_types:
            layer_types = [nn.Conv2d]
        logging.basicConfig(format="%(asctime)-15s %(clientip)s %(user)-8s %(message)s")

        self.original_model = model
        self.output_size = []
        self.layers_type = []
        self.layers_dim = []
        self.weights_size = []
        self.input_size = []
        self.batch_size = batch_size

        self._input_shape = input_shape
        self._inj_layer_types = layer_types

        self.corrupted_model = None
        self.current_layer = 0
        self.handles = []
        self.corrupt_batch = []
        self.corrupt_layer = []
        self.corrupt_dim = [[], [], []]  # C, H, W
        self.corrupt_value = []

        self.use_cuda = kwargs.get("use_cuda", next(model.parameters()).is_cuda)

        if not isinstance(input_shape, list):
            raise AssertionError("Error: Input shape must be provided as a list.")
        if not (isinstance(batch_size, int) and batch_size >= 1):
            raise AssertionError("Error: Batch size must be an integer greater than 1.")
        if len(layer_types) < 0:
            raise AssertionError("Error: At least one layer type must be selected.")

        handles, _shapes, self.weights_size = self._traverse_model_set_hooks(
            self.original_model, self._inj_layer_types
        )
 
        dummy_shape = (1, *self._input_shape)  # profiling only needs one batch element
        model_dtype = next(model.parameters()).dtype
        device = "cuda" if self.use_cuda else None
        _dummy_tensor = torch.randn(dummy_shape, dtype=model_dtype, device=device)

        with torch.no_grad():
            self.original_model(_dummy_tensor)

        for index, _handle in enumerate(handles):
            handles[index].remove()

        logger.info("Input shape:")
        logger.info(dummy_shape[1:])

        logger.info("Model layer sizes:")
        logger.info(
            "\n".join(
                [
                    "".join(["{:4}".format(item) for item in row])
                    for row in self.output_size
                ]
            )
        )

    def reset_fault_injection(self):
        self._reset_fault_injection_state()
        self.corrupted_model = None
        logger.info("Fault injector reset.")

    def _reset_fault_injection_state(self):
        (
            self.current_layer,
            self.corrupt_batch,
            self.corrupt_layer,
            self.corrupt_dim,
            self.corrupt_value,
        ) = (0, [], [], [[], [], []], [])

        for index, _handle in enumerate(self.handles):
            self.handles[index].remove()

    def _traverse_model_set_hooks(self, model, layer_types):
        handles = []
        output_shape = []
        weights_shape = []
        for layer in model.children():
            # leaf node
            if list(layer.children()) == []:
                if "all" in layer_types:
                    handles.append(layer.register_forward_hook(self._save_output_size))
                else:
                    for i in layer_types:
                        if isinstance(layer, i):
                            # neurons
                            handles.append(
                                layer.register_forward_hook(self._save_output_size)
                            )
                            output_shape.append(layer)

                            # weights
                            weights_shape.append(layer.weight.shape)
            # unpack node
            else:
                subhandles, subbase, subweight = self._traverse_model_set_hooks(
                    layer, layer_types
                )
                for i in subhandles:
                    handles.append(i)
                for i in subbase:
                    output_shape.append(i)
                for i in subweight:
                    weights_shape.append(i)

        return (handles, output_shape, weights_shape)

    def _traverse_model_set_hooks_neurons(self, model, layer_types, customInj, injFunc):
        handles = []
        for layer in model.children():
            # leaf node
            if list(layer.children()) == []:
                if "all" in layer_types:
                    hook = injFunc if customInj else self._set_value
                    handles.append(layer.register_forward_hook(hook))
                else:
                    for i in layer_types:
                        if isinstance(layer, i):
                            hook = injFunc if customInj else self._set_value
                            handles.append(layer.register_forward_hook(hook))
            # unpack node
            else:
                subHandles = self._traverse_model_set_hooks_neurons(
                    layer, layer_types, customInj, injFunc
                )
                for i in subHandles:
                    handles.append(i)

        return handles

    def declare_weight_fault_injection(self, **kwargs):
        self._reset_fault_injection_state()
        custom_injection = False
        custom_function = False        
        bitflip_injection = False

        if kwargs:
            if "function" in kwargs:
                custom_injection, custom_function = True, kwargs.get("function")
                corrupt_layer = kwargs.get("layer_num", [])
                corrupt_k = kwargs.get("k", [])
                corrupt_c = kwargs.get("dim1", [])
                corrupt_kH = kwargs.get("dim2", [])
                corrupt_kW = kwargs.get("dim3", [])

            elif "BitFlip" in kwargs:
                bitflip_injection, custom_injection, custom_function = True, True, kwargs.get("BitFlip")
                corrupt_layer = kwargs.get("layer_num", [])
                corrupt_k = kwargs.get("k", [])
                corrupt_c = kwargs.get("dim1", [])
                corrupt_kH = kwargs.get("dim2", [])
                corrupt_kW = kwargs.get("dim3", [])
                corrupt_bitmask=kwargs.get("bitmask", [])

            else:
                corrupt_layer = kwargs.get(
                    "layer_num",
                )
                corrupt_k = kwargs.get("k", [])
                corrupt_c = kwargs.get("dim1", [])
                corrupt_kH = kwargs.get("dim2", [])
                corrupt_kW = kwargs.get("dim3", [])
                corrupt_value = kwargs.get("value", [])
        else:
            raise ValueError("Please specify an injection or injection function")

        # TODO: bound check here

        self.corrupted_model = copy.deepcopy(self.original_model)

        current_weight_layer = 0
        for layer in self.corrupted_model.modules():
            if isinstance(layer, tuple(self._inj_layer_types)):
                inj_list = list(
                    filter(
                        lambda x: corrupt_layer[x] == current_weight_layer,
                        range(len(corrupt_layer)),
                    )
                )

                for inj in inj_list:
                    corrupt_idx = tuple(
                        [
                            corrupt_k[inj],
                            corrupt_c[inj],
                            corrupt_kH[inj],
                            corrupt_kW[inj],
                        ]
                    )
                    orig_value = layer.weight[corrupt_idx].item()

                    with torch.no_grad():
                        if custom_injection:
                            if bitflip_injection:
                                corrupt_value = custom_function(layer.weight, corrupt_idx, corrupt_bitmask[inj])
                            else:
                                corrupt_value = custom_function(layer.weight, corrupt_idx)
                            layer.weight[corrupt_idx] = corrupt_value
                        else:
                            layer.weight[corrupt_idx] = corrupt_value[inj]

                    # logger.info("Weight Injection")
                    # logger.info(f"Layer index: {corrupt_layer[inj]}")
                    # logger.info(f"Module: {layer}")
                    # logger.info(f"Original value: {orig_value}")
                    # logger.info(f"Injected value: {layer.weight[corrupt_idx]}")
                current_weight_layer += 1
        return self.corrupted_model
    
    def declare_ber_weight_fault_injection(self, **kwargs):
        self._reset_fault_injection_state()
        custom_injection = False
        custom_function = False        
        bitflip_injection = False

        if kwargs:
            if "function" in kwargs:
                custom_injection, custom_function = True, kwargs.get("function")
                corrupt_layer = None
            elif "BitFlip" in kwargs:
                bitflip_injection, custom_injection, custom_function = True, True, kwargs.get("BitFlip")
                corrupt_layer = None
            else:
                corrupt_layer = kwargs.get(
                    "layer_num",
                )
            fault_description = kwargs.get("fault_description")
            bitmask= kwargs.get('bitmask')
            ber = kwargs.get('ber')
            trial = kwargs.get('trial')
        else:
            raise ValueError("Please specify an injection or injection function")

        self.corrupted_model = copy.deepcopy(self.original_model)
        current_weight_layer = 0
        induced_errors = list()
        for layer in self.corrupted_model.modules():
            if isinstance(layer, torch.nn.modules.conv.Conv2d):
                fault_description_per_layer = fault_description.query(f'layer=={current_weight_layer}')
                corrupt_idxs = np.array(fault_description_per_layer[['kernel','channel','row','col']], dtype=int)
                for idxs in corrupt_idxs:
                    with torch.no_grad():
                        if custom_injection:
                            if bitflip_injection:
                                corrupt_val, induced_error = custom_function(layer.weight, tuple(idxs), bitmask, ber, len(fault_description_per_layer), trial, induced_errors)
                            else:
                                corrupt_val, induced_error = custom_function(layer.weight, tuple(idxs), bitmask, ber, len(fault_description_per_layer), trial, induced_errors)
                            layer.weight[tuple(idxs)] = corrupt_val
                            induced_errors.append(induced_error)
                current_weight_layer += 1
                
            elif isinstance(layer, torch.nn.modules.linear.Linear):
                fault_description_per_layer = fault_description.query(f'layer=={current_weight_layer}')
                corrupt_idxs = np.array(fault_description_per_layer[['kernel','channel']], dtype=int)
                for idxs in corrupt_idxs:
                    with torch.no_grad():
                        if custom_injection:
                            if bitflip_injection:
                                corrupt_val, induced_error = custom_function(layer.weight, tuple(idxs), bitmask, ber, len(fault_description_per_layer), trial, induced_errors)
                            else:
                                corrupt_val, induced_error = custom_function(layer.weight, tuple(idxs), bitmask, ber, len(fault_description_per_layer), trial, induced_errors)
                            layer.weight[tuple(idxs)] = corrupt_val
                            induced_errors.append(induced_error)
                current_weight_layer += 1

        return self.corrupted_model
    
    def declare_var_bit_ber_weight_fault_injection(self, **kwargs):
        self._reset_fault_injection_state()
        custom_injection = False
        custom_function = False        
        bitflip_injection = False

        if kwargs:
            if "function" in kwargs:
                custom_injection, custom_function = True, kwargs.get("function")
                corrupt_layer = None
            elif "BitFlip" in kwargs:
                bitflip_injection, custom_injection, custom_function = True, True, kwargs.get("BitFlip")
                corrupt_layer = None
            else:
                corrupt_layer = kwargs.get(
                    "layer_num",
                )
            fault_description = kwargs.get("fault_description")
            ber = kwargs.get('ber')
            trial = kwargs.get('trial')
        else:
            raise ValueError("Please specify an injection or injection function")

        self.corrupted_model = copy.deepcopy(self.original_model)
        current_weight_layer = 0
        induced_errors = list()
        for layer in self.corrupted_model.modules():

            if isinstance(layer, torch.nn.modules.conv.Conv2d):
                fault_description_per_layer = fault_description.query(f'layer=={current_weight_layer}')
                corrupt_idxs = np.array(fault_description_per_layer[['kernel','channel','row','col']], dtype=int)
                bitmasks = np.array(fault_description_per_layer[['bitmask']], dtype=int)

                for idxs, bitmask in zip(corrupt_idxs, bitmasks):
                    with torch.no_grad():
                        if custom_injection:
                            if bitflip_injection:
                                corrupt_val, induced_error = custom_function(layer.weight, tuple(idxs), bitmask, ber, trial, induced_errors)
                            else:
                                corrupt_val, induced_error = custom_function(layer.weight, tuple(idxs), bitmask, ber, trial, induced_errors)
                            layer.weight[tuple(idxs)] = corrupt_val
                            induced_errors.append(induced_error)
                current_weight_layer += 1
                
            elif isinstance(layer, torch.nn.modules.linear.Linear):
                fault_description_per_layer = fault_description.query(f'layer=={current_weight_layer}')
                corrupt_idxs = np.array(fault_description_per_layer[['kernel','channel']], dtype=int)
                bitmasks = np.array(fault_description_per_layer[['bitmask']], dtype=int)

                for idxs, bitmask in zip(corrupt_idxs, bitmasks):
                    with torch.no_grad():
                        if custom_injection:
                            if bitflip_injection:
                                corrupt_val, induced_error = custom_function(layer.weight, tuple(idxs), bitmask, ber, trial, induced_errors)
                            else:
                                corrupt_val, induced_error = custom_function(layer.weight, tuple(idxs), bitmask, ber, trial, induced_errors)
                            layer.weight[tuple(idxs)] = corrupt_val
                            induced_errors.append(induced_error)
                current_weight_layer += 1

        return self.corrupted_model
 

    def declare_neuron_fault_injection(self, **kwargs):
        self._reset_fault_injection_state()
        custom_injection = False
        injection_function = False

        if kwargs:
            if "function" in kwargs:
                logger.info("Declaring Custom Function")
                custom_injection, injection_function = True, kwargs.get("function")
            else:
                logger.info("Declaring Specified Fault Injector")
                self.corrupt_value = kwargs.get("value", [])

            self.corrupt_layer = kwargs.get("layer_num", [])
            self.corrupt_batch = kwargs.get("batch", [])
            self.corrupt_dim[0] = kwargs.get("dim1", [])
            self.corrupt_dim[1] = kwargs.get("dim2", [])
            self.corrupt_dim[2] = kwargs.get("dim3", [])

            logger.info(f"Convolution: {self.corrupt_layer}")
            logger.info("Batch, x, y, z:")
            logger.info(
                f"{self.corrupt_batch}, {self.corrupt_dim[0]}, {self.corrupt_dim[1]}, {self.corrupt_dim[2]}"
            )
        else:
            raise ValueError("Please specify an injection or injection function")

        self.check_bounds(
            self.corrupt_batch,
            self.corrupt_layer,
            self.corrupt_dim,
        )

        self.corrupted_model = copy.deepcopy(self.original_model)
        handles_neurons = self._traverse_model_set_hooks_neurons(
            self.corrupted_model,
            self._inj_layer_types,
            custom_injection,
            injection_function,
        )

        for i in handles_neurons:
            self.handles.append(i)

        return self.corrupted_model

    def check_bounds(self, batch, layer, dim):
        if (
            len(batch) != len(layer)
            or len(batch) != len(dim[0])
            or len(batch) != len(dim[1])
            or len(batch) != len(dim[2])
        ):
            raise AssertionError("Injection location missing values.")

        logger.info("Checking bounds before runtime")
        for i in range(len(batch)):
            self.assert_injection_bounds(i)

    def assert_injection_bounds(self, index: int):
        if index < 0:
            raise AssertionError(f"Invalid injection index: {index}")
        if self.corrupt_batch[index] >= self.batch_size:
            raise AssertionError(
                f"{self.corrupt_batch[index]} < {self.batch_size()}: Invalid batch element!"
            )
        if self.corrupt_layer[index] >= len(self.output_size):
            raise AssertionError(
                f"{self.corrupt_layer[index]} < {len(self.output_size)}: Invalid layer!"
            )

        corrupt_layer_num = self.corrupt_layer[index]
        layer_type = self.layers_type[corrupt_layer_num]
        layer_dim = self.layers_dim[corrupt_layer_num]
        layer_shape = self.output_size[corrupt_layer_num]

        for d in range(1, 4):
            if layer_dim > d and self.corrupt_dim[d - 1][index] >= layer_shape[d]:
                raise AssertionError(
                    f"{self.corrupt_dim[d - 1][index]} < {layer_shape[d]}: Out of bounds error in Dimension {d}!"
                )

        if layer_dim <= 2 and (
            self.corrupt_dim[1][index] is not None
            or self.corrupt_dim[2][index] is not None
        ):
            warnings.warn(
                f"Values in Dim2 and Dim3 ignored, since layer is {layer_type}"
            )

        if layer_dim <= 3 and self.corrupt_dim[2][index] is not None:
            warnings.warn(f"Values Dim3 ignored, since layer is {layer_type}")

        logger.info(f"Finished checking bounds on inj '{index}'")

    def _set_value(self, module, input_val, output):
        logger.info(
            f"Processing hook of Layer {self.current_layer}: {self.layers_type[self.current_layer]}"
        )
        inj_list = list(
            filter(
                lambda x: self.corrupt_layer[x] == self.current_layer,
                range(len(self.corrupt_layer)),
            )
        )

        layer_dim = self.layers_dim[self.current_layer]

        logger.info(f"Layer {self.current_layer} injection list size: {len(inj_list)}")
        if layer_dim == 2:
            for i in inj_list:
                self.assert_injection_bounds(index=i)
                logger.info(
                    f"Original value at [{self.corrupt_batch[i]}][{self.corrupt_dim[0][i]}]: {output[self.corrupt_batch[i]][self.corrupt_dim[0][i]]}"
                )
                logger.info(f"Changing value to {self.corrupt_value[i]}")
                output[self.corrupt_batch[i]][
                    self.corrupt_dim[0][i]
                ] = self.corrupt_value[i]
        elif layer_dim == 3:
            for i in inj_list:
                self.assert_injection_bounds(index=i)
                logger.info(
                    f"Original value at [{self.corrupt_batch[i]}][{self.corrupt_dim[0][i]}][{self.corrupt_dim[1][i]}]: {output[self.corrupt_batch[i]][self.corrupt_dim[0][i]][self.corrupt_dim[1][i]]}"
                )
                logger.info(f"Changing value to {self.corrupt_value[i]}")
                output[self.corrupt_batch[i]][
                    self.corrupt_dim[0][i], self.corrupt_dim[1][i]
                ] = self.corrupt_value[i]
        elif layer_dim == 4:
            for i in inj_list:
                self.assert_injection_bounds(index=i)
                logger.info(
                    f"Original value at [{self.corrupt_batch[i]}][{self.corrupt_dim[0][i]}][{self.corrupt_dim[1][i]}][{self.corrupt_dim[2][i]}]: {output[self.corrupt_batch[i]][self.corrupt_dim[0][i]][self.corrupt_dim[1][i]][self.corrupt_dim[2][i]]}"
                )
                logger.info(f"Changing value to {self.corrupt_value[i]}")
                output[self.corrupt_batch[i]][self.corrupt_dim[0][i]][
                    self.corrupt_dim[1][i]
                ][self.corrupt_dim[2][i]] = self.corrupt_value[i]

        self.update_layer()

    def _save_output_size(self, module, input_val, output):
        shape = list(output.size())
        dim = len(shape)
        self.input_size.append(input_val[0].size())
        self.layers_type.append(type(module))
        self.layers_dim.append(dim)
        self.output_size.append(shape)
        # breakpoint()

    def update_layer(self, value=1):
        self.current_layer += value

    def reset_current_layer(self):
        self.current_layer = 0

    def get_weights_size(self, layer_num):
        return self.weights_size[layer_num]

    def get_all_weights_sizes(self):
        return self.weights_size

    def get_weights_dim(self, layer_num):
        return len(self.weights_size[layer_num])

    def get_layer_type(self, layer_num):
        return self.layers_type[layer_num]

    def get_layer_dim(self, layer_num):
        return self.layers_dim[layer_num]

    def get_layer_shape(self, layer_num):
        return self.output_size[layer_num]

    def get_total_layers(self):
        return len(self.output_size)

    def get_tensor_dim(self, layer, dim):
        if dim > len(self.layers_dim):
            raise AssertionError(f"Dimension {dim} is out of bounds for layer {layer}")
        return self.output_size[layer][dim]

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
        line_new = "{:>5}  {:>15}  {:>10} {:>20} {:>20}".format(
            "Layer #", "Layer type", "Dimensions", "Weight Shape", "Output Shape"
        )
        summary_str += line_new + "\n"
        summary_str += (
            "----------------------------------------------------------------------------------"
            + "\n"
        )
        for layer, _dim in enumerate(self.output_size):
            try: 
                strt_try = "(0,0,0,0)" if "all" in self._inj_layer_types else str(list(self.weights_size[layer]))
            except:
                breakpoint()
            line_new = "{:>5}  {:>15}  {:>10} {:>20} {:>20}".format(
                layer,
                str(self.layers_type[layer]).split(".")[-1].split("'")[0],
                str(self.layers_dim[layer]),
                str(list(self.weights_size[layer])),
                str(self.output_size[layer]),
            )
            summary_str += line_new + "\n"

        summary_str += (
            "=================================================================================="
            + "\n"
        )

        logger.info(summary_str)
        return summary_str