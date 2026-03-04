# xPytorchFI: An Extended Fault Injection Library for PyTorch

`xPytorchFI` is a Python library designed to facilitate flexible and reproducible fault injection experiments on PyTorch neural networks. It extends the principles of `PytorchFI` with a high-level API that simplifies the orchestration of complex experiments, allowing researchers and engineers to focus on their specific evaluation logic rather than on the boilerplate of fault management.

The core of the library is the `ExperimentRunner`, which manages the entire fault injection lifecycle, including:
-   Declarative experiment configuration.
-   Fault list generation and management.
-   Resumable experiments through automatic checkpointing.
-   A callback-based system for injecting custom evaluation logic.
-   Automatic aggregation of results.

## Installation

```bash
pip install -e .
```

## Quickstart: Running a Fault Injection Experiment

Here’s how to set up and run a complete fault injection experiment.

### 1. Implement a Custom Callback

Define your experiment's logic by inheriting from `ExperimentCallback`. You'll implement methods to handle the golden (fault-free) run and to compute metrics after each fault is injected.

```python
import torch
from xpytorchfi import ExperimentCallback

class MyExperimentCallback(ExperimentCallback):
    def on_golden_run_end(self, model, output):
        """Save the golden run's output for later comparison."""
        self.golden_output = output
        print("Golden run finished. Output stored.")

    def on_fault_injection_end(self, faulty_model, output, fault):
        """
        Compute metrics by comparing the faulty output with the golden output.
        This method is called after each fault injection.
        """
        # Example: Calculate Mean Squared Error between golden and faulty output
        mse = torch.nn.functional.mse_loss(self.golden_output, output)

        # Return a dictionary of metrics to be saved
        return {
            "mse": mse.item(),
            "fault_layer": fault["layer"],
            "fault_bitmask": fault["bitmask"],
        }

    def on_experiment_end(self, results_df):
        """Called when the experiment is complete and results are aggregated."""
        print("Experiment finished!")
        print("Aggregated Results:")
        print(results_df.head())
```

### 2. Define an Inference Function

Create a function that takes a model and your data and performs a forward pass, returning the output you want to evaluate.

```python
import torch

def my_inference_fn(model, data):
    """A simple inference function that runs the model on the input data."""
    model.eval()
    with torch.no_grad():
        return model(data)
```

### 3. Configure and Run the Experiment

Set up a configuration dictionary that defines every aspect of your experiment. Then, instantiate the `ExperimentRunner` and call `run()`.

```python
import torch
import torch.nn as nn
from xpytorchfi import ExperimentRunner

# --- Setup ---
# 1. Define a simple model and some dummy data
model = nn.Sequential(nn.Conv2d(3, 8, 3), nn.ReLU(), nn.Flatten(), nn.Linear(8 * 30 * 30, 10))
dummy_data = torch.randn(1, 3, 32, 32)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
dummy_data = dummy_data.to(device)

# 2. Define the experiment configuration
# This can also be loaded from a YAML file
experiment_config = {
    "output_dir": "./fi_results/my_first_experiment",
    "policy": "sfbm", # Selects the fault generation policy

    "injection": {
        "input_shape": [3, 32, 32],
        "batch_size": 1,
        "layer_types": [nn.Conv2d],
    },

    "faults": {
        "layer": 0,
        "num_faults": 50, # Generate 50 random bit-flips
    },
}

# 3. Instantiate the callback
my_callback = MyExperimentCallback()

# 4. Create and run the experiment
runner = ExperimentRunner(
    model=model,
    data=dummy_data,
    config=experiment_config,
    callback=my_callback,
    inference_fn=my_inference_fn,
)
runner.run()
```

## Configuration Options

The `ExperimentRunner` is controlled by a single configuration dictionary (or YAML file) with the following structure:

| Key          | Type   | Description                                                                                                                            |
| ------------ | ------ | -------------------------------------------------------------------------------------------------------------------------------------- |
| `output_dir` | `str`  | **Required.** Path to the directory where the fault list, checkpoints, and final results will be stored.                               |
| `policy`     | `str`  | **Required.** The fault generation policy to use. Supported values: `"sfbm"`, `"ber"`, `"neurons"`.             |
| `injection`  | `dict` | **Required.** Configuration for the underlying fault injection engine. See sub-table below.                                            |
| `faults`     | `dict` | **Required.** Configuration for generating the fault list, specific to the chosen `policy`. See sub-table below.                       |

### `injection` 

| Key           | Type           | Description                                                                                             |
| ------------- | -------------- | ------------------------------------------------------------------------------------------------------- |
| `input_shape` | `List[int]`    | **Required.** The shape of a single input sample, e.g., `[C, H, W]`.                                     |
| `batch_size`  | `int`          | The batch size used for injection setup. Defaults to `1`.                                               |
| `layer_types` | `List[Module]` | A list of PyTorch layer types to consider for fault injection (e.g., `[nn.Conv2d, nn.Linear]`).          |
| `...`         |                | Other parameters accepted by `xpytorchfi.fault_injection.FIFramework`.                                  |

### `faults` (formerly `fault_generation_config`)

| Key        | Type    | Description                                                                                                                                                                                          |
| ---------- | ------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `...`      |         | Parameters specific to the chosen `policy`. For example, for `policy: "sfbm"`, you can specify `layer`, `num_faults`, `msb_injection`, `lsb_injection`, and other statistical parameters. |
