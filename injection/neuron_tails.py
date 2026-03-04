from typing import Any, Dict, List, Optional, Tuple
import torch


def _randint(low: int, high: int):
    return int(torch.randint(low, high, (1,)).item())


def loc_neuron(
    layer: int = -1,
    dim: int = 1,
    shape: List[int] = [],
    BlockID_y: int = 1,
    BlockID_x: int = 1,
    Neuron_x: int = -1,
    Neuron_y: int = -1,
    tail_bloc_y: int = 1,
    tail_bloc_x: int = 1,
) -> Tuple[int, int, Optional[int], Optional[int]]:
    """
    Build a single neuron location tuple `(layer, dim1, dim2, dim3)`.

    For 4D activations (e.g., Conv2d outputs), this maps a block id plus an
    in-block neuron coordinate to channel/height/width indices.
    For 2D activations (e.g., Linear outputs), only `dim1` is used and spatial
    dimensions are returned as `None`.

    Args:
        layer: Target layer index.
        dim: Layer tensor dimensionality returned by the FI model.
        shape: Layer shape as returned by the FI model.
        BlockID_y: Block index on Y axis.
        BlockID_x: Block index on X axis.
        Neuron_x: In-block X coordinate. If -1, sampled randomly.
        Neuron_y: In-block Y coordinate. If -1, sampled randomly.
        tail_bloc_y: Block height.
        tail_bloc_x: Block width.

    Returns:
        Tuple `(layer, dim1, dim2, dim3)` identifying the neuron position.
    """

    # Sample in-block coordinates only when not explicitly provided.
    dy = _randint(0, tail_bloc_y) if Neuron_y == -1 else Neuron_y
    dx = _randint(0, tail_bloc_x) if Neuron_x == -1 else Neuron_x

    if dim > 3:
        # Flatten block coordinates to the equivalent tensor coordinates.
        dim1_rand = (BlockID_y * tail_bloc_y + dy) // tail_bloc_y
        dim2_rand = (BlockID_x * tail_bloc_x + dx) // shape[2]
        dim3_rand = (BlockID_x * tail_bloc_x + dx) % shape[2]

        # Clamp to valid bounds for safety.
        dim1_rand = min(dim1_rand, shape[1] - 1)
        dim2_rand = min(dim2_rand, shape[2] - 1)
        dim3_rand = min(dim3_rand, shape[3] - 1)
    else:
        # Dense-like tensors only use one logical axis.
        dim1_rand = (BlockID_y * tail_bloc_y + dy) // tail_bloc_y
        dim2_rand = None
        dim3_rand = None

    return (layer, dim1_rand, dim2_rand, dim3_rand)


def _loc_neurons(
    dim,
    shape,
    block_y: torch.Tensor,
    block_x: torch.Tensor,
    neuron_y: torch.Tensor,
    neuron_x: torch.Tensor,
    curr_tail_y: int,
    curr_tail_x: int,
):
    """
    Vectorized version of `loc_neuron` to map multiple block/neuron coordinates to tensor indices.

    Args:
        dim: Layer tensor dimensionality.
        shape: Layer shape as returned by the FI model.
        block_y: Tensor of block Y indices.
        block_x: Tensor of block X indices.
        neuron_y: Tensor of in-block Y coordinates.
        neuron_x: Tensor of in-block X coordinates.
        curr_tail_y: Block height.
        curr_tail_x: Block width.

    Returns:
            Tuple of tensors `(dim1, dim2, dim3)` representing the mapped coordinates.
    """
    num_faulty_blocks = block_y.numel()
    num_faulty_neurons = neuron_y.numel()

    # Cartesian product blocks x neurons (vectorized).
    bx = block_x.repeat_interleave(num_faulty_neurons)
    by = block_y.repeat_interleave(num_faulty_neurons)
    nx = neuron_x.repeat(num_faulty_blocks)
    ny = neuron_y.repeat(num_faulty_blocks)

    linear_y = by * curr_tail_y + ny
    linear_x = bx * curr_tail_x + nx

    dim1_t = linear_y // curr_tail_y

    if dim > 3:
        dim2_t = linear_x // shape[2]
        dim3_t = linear_x % shape[2]

        # Clamp for safety.
        dim1_t = torch.clamp(dim1_t, max=shape[1] - 1)
        dim2_t = torch.clamp(dim2_t, max=shape[2] - 1)
        dim3_t = torch.clamp(dim3_t, max=shape[3] - 1)
    else:
        dim2_t = [None] * dim1_t.numel()
        dim3_t = [None] * dim1_t.numel()

    return dim1_t.tolist(), dim2_t, dim3_t


def generate_error_list_neurons_tails(
    pfi_model: Any,
    layer_i: int = -1,
    layer_n: int = -1,
    block_error_rate: float = 1,
    neuron_fault_rate: float = 0.001,
    tail_bloc_y: int = 32,
    tail_bloc_x: int = 32,
):
    """
    Generate neuron fault locations by selecting faulty blocks and faulty neurons per block.

    The function scans layers in `[layer_i, layer_n]`, computes a block tiling over each
    activation map, then samples:
      - a subset of blocks using `block_error_rate`
      - a subset of neurons in each selected block using `neuron_fault_rate`

    Args:
        pfi_model: Fault injection model exposing layer metadata helpers.
        layer_i: First layer index. If -1, randomly select one layer.
        layer_n: Last layer index. If -1, equals `layer_i`.
        block_error_rate: Fraction of blocks to mark as faulty.
        neuron_fault_rate: Fraction of neurons per block to mark as faulty.
        tail_bloc_y: Block height.
        tail_bloc_x: Block width.
        generator: Optional torch random generator for reproducibility.

    Returns:
        Tuple:
            - all_layers: List of layer indices for each faulty neuron.
            - all_dim1: List of dim1 indices for each faulty neuron.
            - all_dim2: List of dim2 indices (or None) for each faulty neuron.
            - all_dim3: List of dim3 indices (or None) for each faulty neuron.
            - batch_order: List of batch indices for each faulty neuron (currently all zeros).
            - fault_info: per-layer metadata dictionary
    """
    # Pick a random starting layer if not specified.
    if layer_i == -1:
        layer_i = _randint(0, pfi_model.get_total_layers())
    if layer_n == -1:
        layer_n = layer_i

    all_layers: List[int] = []
    all_dim1: List[int] = []
    all_dim2: List[Optional[int]] = []
    all_dim3: List[Optional[int]] = []
    batch_order: List[int] = []
    fault_info: Dict[int, Dict[str, int]] = {}

    for layer in range(layer_i, layer_n + 1):
        dim = pfi_model.get_layer_dim(layer)
        shape = pfi_model.get_layer_shape(layer)

        # GEMM-like logical view of the activation tensor.
        gemm_y = shape[1]
        gemm_x = 1 if dim == 2 else shape[2] * shape[3]

        # Work on local copies to avoid mutating function input across layers.
        curr_tail_y = int(tail_bloc_y)
        curr_tail_x = int(tail_bloc_x)

        # Adapt block size if bigger than available activation footprint.
        if gemm_x * gemm_y < curr_tail_y * curr_tail_x:
            curr_tail_x = gemm_x
            curr_tail_y = gemm_y
        else:
            if gemm_y < curr_tail_y and gemm_x >= curr_tail_x:
                curr_tail_x = curr_tail_y * curr_tail_x // gemm_y
                curr_tail_y = gemm_y
            elif gemm_x < curr_tail_x and gemm_y >= curr_tail_y:
                curr_tail_y = curr_tail_y * curr_tail_x // gemm_x
                curr_tail_x = gemm_x

        if dim == 2:
            max_tail_y = 0 if gemm_y == curr_tail_y * curr_tail_x else gemm_y // (curr_tail_y * curr_tail_x)
            max_tail_x = 0 if gemm_x == curr_tail_y * curr_tail_x else gemm_x // (curr_tail_y * curr_tail_x)
        else:
            max_tail_y = 0 if gemm_y == curr_tail_y else gemm_y // curr_tail_y
            max_tail_x = 0 if gemm_x == curr_tail_x else gemm_x // curr_tail_x

        total_blocks = (max_tail_y + 1) * (max_tail_x + 1)
        total_neurons_per_block = curr_tail_y * curr_tail_x

        num_faulty_neurons = int(neuron_fault_rate * total_neurons_per_block)
        num_faulty_blocks = int(total_blocks * block_error_rate)
        num_faulty_blocks = max(1, num_faulty_blocks)  # keep previous behavior intent

        # Clamp to valid ranges.
        num_faulty_blocks = min(num_faulty_blocks, total_blocks)
        num_faulty_neurons = min(num_faulty_neurons, total_neurons_per_block)

        fault_info[layer] = {
            "layer": layer,
            "tot_blocks": total_blocks,
            "faulty_blocks": num_faulty_blocks,
            "faulty_neuron": num_faulty_neurons,
        }

        if num_faulty_neurons == 0 or num_faulty_blocks == 0:
            continue

        # Sample unique faulty blocks and unique faulty neurons inside each block (without replacement).
        block_idx = torch.randperm(total_blocks)[:num_faulty_blocks]
        neuron_idx = torch.randperm(total_neurons_per_block)[:num_faulty_neurons]

        block_x = block_idx % (max_tail_x + 1)
        block_y = block_idx // (max_tail_x + 1)
        neuron_x = neuron_idx % curr_tail_x
        neuron_y = neuron_idx // curr_tail_x

        dim_1, dim_2, dim_3 = _loc_neurons(
            dim=dim,
            shape=shape,
            block_y=block_y,
            block_x=block_x,
            neuron_y=neuron_y,
            neuron_x=neuron_x,
            curr_tail_y=curr_tail_y,
            curr_tail_x=curr_tail_x,
        )
        count = len(dim_1)
        all_layers.extend([layer] * count)
        all_dim1.extend(dim_1)
        all_dim2.extend(dim_2)
        all_dim3.extend(dim_3)

        batch_order.extend([0] * count)  # Currently all zeros

    return all_layers, all_dim1, all_dim2, all_dim3, batch_order, fault_info
