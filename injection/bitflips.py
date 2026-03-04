import logging
from xpytorchfi import XSingleBitFlipFI

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


class BitFlipWeightsBER:
    """
    Class to inject bit-flip faults into weights during model inference based on a specified bit error rate (BER).
    """
    def __init__(self, save_stats: bool = False):
        """Initializes the BitFlipWeightsBER class.
        Args:
            save_stats (bool, optional): Whether to save fault injection statistics. Defaults to False.
        """
        self.save_stats = save_stats
        self.injected_fault: dict = {}

    def _avg(self, err: list[float]) -> float:
        """Calculate the average of a list of errors."""
        return sum(err) / len(err) if err else 0.0

    def __call__(self, data, location, injmask, ber, trial, error_list=None):
        # Inject the BFW
        orig_data = data[location].item()
        data_32bit = int(self.float_to_hex(data[location].item()), 16)
        corrupt_32bit = data_32bit ^ int(injmask)
        corrupt_val = self.int_to_float(corrupt_32bit)

        # Log the fault injection details
        self.log_msg = f"F_descriptor: Layer:{self._layer}, (K, C, H, W):{location}, BitMask:{injmask}, Ffree_Weight:{data_32bit}, Faulty_weight:{corrupt_32bit}"

        # Save fault injection statistics if required
        if self.save_stats:
            fsim_dict = {
                "ber": ber,
                "trail": trial,
                "induced_error": self._avg(error_list),
            }

            self.injected_faults = fsim_dict

        # Calculate the induced error
        induced_error = abs(corrupt_val - orig_data)
        return corrupt_val, induced_error
