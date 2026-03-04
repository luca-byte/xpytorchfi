import os
import random
import pandas as pd
from pytorchfi.core import FaultInjection


def random_weight_position(weight_shape: list, kernel: int = None, channel: int = None) -> tuple:
    """
    Generates a random position in the weight tensor.
    Args:
        weight_shape: Shape of the weight tensor.
        kernel: Specific kernel (optional, random if None).
        channel: Specific channel (optional, random if None).
    Returns:
        Tuple of (kernel, channel, height, width) indices.
    """
    k = kernel if kernel is not None else random.randint(0, weight_shape[0] - 1)
    c = channel if channel is not None else random.randint(0, weight_shape[1] - 1)
    if len(weight_shape) == 4:
        h = random.randint(0, weight_shape[2] - 1)
        w = random.randint(0, weight_shape[3] - 1)
    else:
        h = None
        w = None
    return k, c, h, w


def get_num_weights(weight_shape: list, kernel: int = None, channel: int = None) -> int:
    """
    Computes the total number of weights considered for fault injection.
    Args:
        weight_shape: Shape of the weight tensor.
        kernel: Specific kernel (optional, all if None).
        channel: Specific channel (optional, all if None).
    Returns:
        Total number of weights.
    """
    n_kernels = 1 if kernel is not None else weight_shape[0]
    n_channels = 1 if channel is not None else weight_shape[1]
    if len(weight_shape) == 4:
        return n_kernels * n_channels * weight_shape[2] * weight_shape[3]
    else:
        return n_kernels * n_channels


def generate_fault_list_sbfm(
    path: str,
    pfi_model: FaultInjection,
    f_list_file: str,
    layer: int = None,
    kernel: int = None,
    channel: int = None,
    num_faults: int = None,
    unique_faults: bool = True,
    msb_injection: int = 31,
    lsb_injection: int = 20,
    confidence_level: float = 1.64485362695147,
    error_margin: float = 0.01,
    p_error: float = 0.5
) -> pd.DataFrame:
    """
    Generates a list of faults (bit flips) to inject into the weights of a neural network according to the SBFM model.
    Saves the list to a CSV file and returns it as a DataFrame.

    Args:
        path: Folder where the CSV file will be saved.
        pfi_model: Instance of FaultInjection.
        f_list_file: Name of the CSV file.
        layer: Layer to inject faults into (optional, random if None).
        kernel: Specific kernel (optional, random if None).
        channel: Specific channel (optional, random if None).
        num_faults: Number of faults to generate (optional, computed if None).
        msb_injection: Most significant bit that can be flipped.
        lsb_injection: Least significant bit that can be flipped.
        confidence_level: Statistical confidence level.
        error_margin: Statistical error margin.
        p_error: Error probability.

    Returns:
        DataFrame with the list of faults.
    """

    # If the file already exists, load and return it
    full_path = os.path.join(path, f_list_file)
    if os.path.exists(full_path):
        return pd.read_csv(full_path, index_col=0)

    # Select the layer and obtain the weight shape
    layr = layer if layer is not None else random.randint(0, pfi_model.get_total_layers() - 1)
    weight_shape = list(pfi_model.get_weights_size(layr))

    # Compute total number of weights considered for fault injection
    n_weights = get_num_weights(weight_shape, kernel, channel)

    # Compute number of faults
    if num_faults is None:
        num_faults = int(n_weights / (1 + (error_margin ** 2) * (n_weights - 1) / ((confidence_level ** 2) * p_error * (1 - p_error))))  # noqa: E501

    fault_list = []
    i = 0
    while i < num_faults:

        # Generate a random fault
        k, c, h, w = random_weight_position(weight_shape, kernel, channel)
        mask = 2 ** random.randint(lsb_injection, msb_injection)
        fault = [layr, k, c, h, w, mask]

        if fault not in fault_list or not unique_faults:
            fault_list.append(fault)
            i += 1

    f_list = pd.DataFrame(fault_list, columns=['layer', 'kernel', 'channel', 'row', 'col', 'bitmask'])
    f_list.to_csv(full_path, sep=',')
    return f_list


def generate_fault_list_sbfm_fails(
    path: str,
    pfi_model: FaultInjection,
    f_list_file: str,
    layer: int = None,
    kernel: int = None,
    channel: int = None,
    msb_injection: int = 31,
    lsb_injection: int = 19,
    confidence_level: float = 2.576,
    error_margin: float = 0.01,
    bit_error_probs: list = None
) -> pd.DataFrame:
    """
    Generates a list of faults (bit flips) to inject into the weights of a neural network,
    with a specific probability for each bit (SBFM model with per-bit probability).
    Saves the list to a CSV file and returns it as a DataFrame.

    Args:
        path: Folder where the CSV file will be saved.
        pfi_model: Instance of FaultInjection.
        f_list_file: Name of the CSV file.
        layer: Layer to inject faults into (optional, random if None).
        kernel: Specific kernel (optional, random if None).
        channel: Specific channel (optional, random if None).
        msb_injection: Most significant bit that can be flipped.
        lsb_injection: Least significant bit that can be flipped.
        confidence_level: Statistical confidence level.
        error_margin: Statistical error margin.
        bit_error_probs: List of error probabilities for each bit (length 32).

    Returns:
        DataFrame with the list of faults.
    """
    full_path = os.path.join(path, f_list_file)
    if os.path.exists(full_path):
        return pd.read_csv(full_path, index_col=0)

    # Per bit error probabilities (default values if None)
    if bit_error_probs is None:
        bit_error_probs = [
            0.000001, 0.000001, 0.000001, 0.000001, 0.000001, 0.000001, 0.000001, 0.000001,
            0.000001, 0.000001, 0.000001, 0.000001, 0.000001, 0.000001, 0.000001, 0.000001,
            0.000001, 0.000001, 0.00001, 0.0005, 0.0005, 0.0005, 0.005, 0.005,
            0.05, 0.05, 0.05, 0.01, 0.01, 0.01, 0.5, 0.05
        ]

    # Select the layer and obtain the weight shape
    layr = layer if layer is not None else random.randint(0, pfi_model.get_total_layers() - 1)
    weight_shape = list(pfi_model.get_weights_size(layr))

    # Compute total number of weights considered for fault injection
    num_weights = get_num_weights(weight_shape, kernel, channel)

    fault_list = []

    for bitdx in range(lsb_injection, msb_injection + 1):
        P = bit_error_probs[bitdx]
        N = num_weights
        n = int(N / (1 + (error_margin ** 2) * (N - 1) / ((confidence_level ** 2) * P * (1 - P))))
        i = 0
        while i < n:

            # Generate a random fault
            k, c, h, w = random_weight_position(weight_shape, kernel, channel)
            mask = 2 ** bitdx
            fault = [layr, k, c, h, w, mask]


            if fault not in fault_list:
                fault_list.append(fault)
                i += 1

    f_list = pd.DataFrame(fault_list, columns=['layer', 'kernel', 'channel', 'row', 'col', 'bitmask'])
    f_list.to_csv(full_path, sep=',')
    return f_list


def generate_fault_neurons_tailing(
    path: str,
    pfi_model: FaultInjection,
    f_list_file: str,
    trials: int,
    size_tail_y: int,
    size_tail_x: int,
    layers: list,
    block_fault_rate_delta: float = 0.01,
    block_fault_rate_steps: int = 1,
    neuron_fault_rate_delta: float = 0.01,
    neuron_fault_rate_steps: int = 1
) -> pd.DataFrame:
    """
    Generates a list of faults localized in blocks of neurons (tail) on one or more layers.
    Each row represents a fault configuration to inject.

    Args:
        path: Folder where the CSV file will be saved.
        pfi_model: Instance of FaultInjection.
        f_list_file: Name of the CSV file.
        trials: Number of trials for each parameter combination.
        size_tail_y: Size of the neuron block (y axis).
        size_tail_x: Size of the neuron block (x axis).
        layers: List [start, stop] of layers to inject faults into.
        block_fault_rate_delta: Increment of the block fault rate.
        block_fault_rate_steps: Number of steps for the block fault rate.
        neuron_fault_rate_delta: Increment of the neuron fault rate.
        neuron_fault_rate_steps: Number of steps for the neuron fault rate.

    Returns:
        DataFrame with the list of faults.
    """
    full_path = os.path.join(path, f_list_file)
    if os.path.exists(full_path):
        return pd.read_csv(full_path, index_col=0)

    fault_dict = {
        'layer_start': layers[0],
        'layer_stop': layers[1] if len(layers) > 1 else layers[0],
        'size_tail_y': size_tail_y,
        'size_tail_x': size_tail_x
    }

    fault_list = []
    for bfr in range(1, block_fault_rate_steps + 1):
        fault_dict['block_fault_rate'] = bfr * block_fault_rate_delta
        for nfr in range(1, neuron_fault_rate_steps + 1):
            n_rate = nfr * neuron_fault_rate_delta
            fault_dict['neuron_fault_rate'] = n_rate
            for bit_pos_fault in range(19, 32):
                for _ in range(trials):
                    fault_dict['bit_faulty_pos'] = bit_pos_fault
                    fault_list.append(fault_dict.copy())

    f_list = pd.DataFrame(fault_list)
    f_list.to_csv(full_path, sep=',')
    return f_list


def generate_fault_list_ber(
    path: str,
    pfi_model: FaultInjection,
    f_list_file: str,
    BER: int,
    trials: int,
    layer: int = None,
    kernel: int = None,
    channel: int = None,
    row: int = None,
    col: int = None
) -> pd.DataFrame:
    """
    Generate a list of faults according to a given Bit Error Rate (BER),
    varying the number of errors from 0 to BER for each trial.
    The list is saved as a CSV and returned as a DataFrame.

    Args:
        path: Folder where the CSV file will be saved.
        pfi_model: Instance of FaultInjection.
        f_list_file: Name of the CSV file.
        BER: Maximum Bit Error Rate (number of bit flips).
        trials: Number of trials for each BER value.
        layer: Layer to inject faults into (optional, random if None).
        kernel: Specific kernel (optional, random if None).
        channel: Specific channel (optional, random if None).
        row: Specific row (optional, random if None).
        col: Specific column (optional, random if None).
    Returns:
        DataFrame with the list of faults.
    """
    full_path = os.path.join(path, f_list_file)
    if os.path.exists(full_path):
        return pd.read_csv(full_path, index_col=0)

    fault_dict = {
        'layer': layer,
        'kernel': kernel,
        'channel': channel,
        'row': row,
        'col': col
    }

    fault_list = []
    for _ in range(trials):
        for brate in range(0, BER + 1):
            fault_dict['ber'] = brate
            fault_list.append(fault_dict.copy())

    f_list = pd.DataFrame(fault_list)
    f_list.to_csv(full_path, sep=',')
    return f_list
