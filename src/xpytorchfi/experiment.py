"""
High-level API for orchestrating fault injection experiments.
"""
import os
import logging
from typing import Any, Dict, Callable, List, Union
import pandas as pd
import torch
import torch.nn as nn
import yaml
from tqdm.auto import tqdm

from .fault_generation import generate_fault_list_sbfm, generate_fault_list_ber, generate_fault_neurons_tailing, 
from .fault_injection import FIFramework
from .manager import FaultIterator

logger = logging.getLogger("XPFI")

POLICY_FUNCTIONS = {
    "sfbm": generate_fault_list_sbfm,
    "ber": generate_fault_list_ber,
    "neurons": generate_fault_neurons_tailing,
}



class ExperimentCallback:
    """
    User-defined hooks for experiment lifecycle events.

    Implement the methods of this class to define the custom logic for an
    experiment, such as model evaluation, metric computation, and result logging.
    """

    def on_golden_run_start(self, model: nn.Module) -> None:
        """Called before the golden run (fault-free execution)."""
        pass

    def on_golden_run_end(self, model: nn.Module, output: Any) -> None:
        """
        Called after the golden run.

        Args:
            model: The fault-free model.
            output: The output of the `inference_fn` from the golden run.
        """
        self.golden_output = output

    def on_fault_injection_start(self, faulty_model: nn.Module, fault: Dict) -> None:
        """Called after a fault has been injected, before the model is run."""
        pass

    def on_fault_injection_end(
        self, faulty_model: nn.Module, output: Any, fault: Dict
    ) -> Dict:
        """
        Called after the faulty model has been run.

        This method should compute and return a dictionary of metrics for the
        current fault injection step.

        Args:
            faulty_model: The model with the injected fault.
            output: The output of the `inference_fn` from the faulty run.
            fault: The fault dictionary for the current step.

        Returns:
            A dictionary of metrics to be saved for this step.
        """
        return {}

    def on_experiment_end(self, results_df: pd.DataFrame) -> None:
        """Called after all faults have been processed and results are aggregated."""
        pass


class ExperimentRunner:
    """
    Orchestrates a fault injection experiment from configuration to result aggregation.
    """

    def __init__(
        self,
        model: nn.Module,
        data: Any,
        config: Union[Dict, str],
        callback: ExperimentCallback,
        inference_fn: Callable[[nn.Module, Any], Any],
        
    ):
        """
        Initializes the experiment runner.

        Args:
            model: The PyTorch model to be used in the experiment.
            config: A dictionary containing the experiment configuration.
            callback: An instance of ExperimentCallback with user-defined logic.
            inference_fn: A function that takes a model and data and returns some output.
        """
        self.model = model
        self.data = data
        self.config = config
        self.callback = callback
        self.inference_fn = inference_fn

        # Load configuration from file if a path is provided
        if isinstance(config, str):
            with open(config, "r") as f:
                self.config = yaml.safe_load(f)

        # Set up device and working directory
        self.workdir = self.config["output_dir"]
        os.makedirs(self.workdir, exist_ok=True)

        with open(os.path.join(self.workdir, "config.yaml"), "w") as f:
            yaml.dump(self.config, f)

        self.policy = self.config.get("policy")

        logger.info("Initializing Fault Injection Framework...")
        self.fi_framework = FIFramework(
            device=self.device,
            model=self.model,
            **self.config["injection"],
        )

        logger.info("Generating or loading fault list...")
        self._generate_faults()

        self.fault_iterator = FaultIterator(
            workdir=self.workdir,
            fault_file="fault_list.csv",
            ckpt_file="ckpt.json",
        )
        self.fault_iterator.load_checkpoint()

    def _generate_faults(self) -> None:
        """Generates the fault list based on the configuration."""
        fault_gen_config: Dict[str, Any] = self.config["faults"]
        
        gen_func = POLICY_FUNCTIONS.get(self.policy)

        gen_func(
            path=self.workdir,
            pfi_model=self.fi_framework.pfi_model,
            f_list_file="fault_list.csv",
            **fault_gen_config,
        )

    def run(self):
        """
        Executes the full fault injection experiment.
        """
        # 1. Golden Run
        logger.info("Starting Golden Run...")
        self.callback.on_golden_run_start(self.model)
        golden_output = self.inference_fn(self.model, self.data)
        metrics = self.callback.on_golden_run_end(self.model, golden_output)
        torch.save(metrics, os.path.join(self.workdir, "result_G.pt"))
        logger.info("Golden Run complete.")

        # 2. Fault Injection Loop
        total_faults = len(self.fault_iterator)
        logger.info(f"Starting fault injection loop for {total_faults} faults.")

        for fault_record, idx in tqdm(self.fault_iterator.iter_faults(), total=total_faults):
            logger.info(f"Processing fault {idx + 1}/{total_faults}...")

            # 2.1. Inject fault
            self.fi_framework.inject_fault(self.policy, fault_record)
            faulty_model = self.fi_framework.faulty_model
            self.callback.on_fault_injection_start(faulty_model, fault_record[0])

            # 2.2. Run inference on faulty model
            faulty_output = self.inference_fn(faulty_model, self.data)

            # 2.3. Compute and save metrics via callback
            metrics = self.callback.on_fault_injection_end(
                faulty_model, faulty_output, fault_record[0]
            )
            
            # Add fault index to metrics
            metrics["fault_index"] = idx
            
            # Save metrics for this step
            step_result_path = os.path.join(self.workdir, f"result_{idx}.pt")
            torch.save(metrics, step_result_path)

        # 3. Aggregate results
        logger.info("Fault injection loop complete. Aggregating results...")
        final_results_df = self.fault_iterator.collate_results()
        
        output_path = os.path.join(self.workdir, "results.csv")
        final_results_df.to_csv(output_path, index=False)
        
        self.callback.on_experiment_end(final_results_df)
        logger.info(f"Experiment finished. Aggregated results saved to {output_path}")

