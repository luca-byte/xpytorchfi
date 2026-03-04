import os
import json
from dataclasses import dataclass, asdict
from typing import Iterator, List, Dict, Tuple, Optional

import pandas as pd


@dataclass
class CheckpointState:
    """Stores the current fault index (next fault to execute)."""
    fault_idx: int = 0


class FIManager:
    """
    Minimal manager for:
    - checkpoint persistence
    - fault list loading
    - resumable fault iteration
    """

    def __init__(
        self,
        workdir: str,
        ckpt_file: str = "ckpt.json",
        fault_file: str = "fault_list.csv",
    ):
        """
        Initialize paths and internal state.

        Args:
            workdir: Working directory used to store checkpoint and fault list.
            ckpt_file: Checkpoint filename.
            fault_file: Fault list filename (CSV).
        """
        self.workdir = workdir
        self.ckpt_path = os.path.join(workdir, ckpt_file)
        self.fault_list_path = os.path.join(workdir, fault_file)

        self.state = CheckpointState()

        # Ensure the working directory exists.
        os.makedirs(workdir, exist_ok=True)

    def load_checkpoint(self) -> CheckpointState:
        """
        Load checkpoint state from disk.

        If the checkpoint file does not exist, create it with default state.
        """
        if not os.path.exists(self.ckpt_path):
            self.save_checkpoint()
            return self.state

        with open(self.ckpt_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.state = CheckpointState(**data)
        return self.state

    def save_checkpoint(self) -> None:
        """
        Persist checkpoint state atomically.

        A temporary file is written first, then replaced to avoid partial writes.
        """
        tmp = self.ckpt_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(asdict(self.state), f, indent=2)
        os.replace(tmp, self.ckpt_path)

    def reset_checkpoint(self) -> None:
        """Reset checkpoint to the beginning (fault_idx = 0)."""
        self.state = CheckpointState(fault_idx=0)
        self.save_checkpoint()

    def advance(self, idx: Optional[int] = None) -> None:
        """
        Advance the checkpoint.

        Args:
            idx: If provided, set next index to idx + 1.
                 If None, increment current index by 1.
        """
        if idx is not None:
            self.state.fault_idx = idx + 1
        else:
            self.state.fault_idx += 1
        self.save_checkpoint()

    def iter_faults(self, from_ckpt: bool = True, auto_advance: bool = True):
        """
        Iterate over faults from the CSV fault list.

        Args:
            from_ckpt: Start from checkpoint index if True, else from 0.
            auto_advance: Automatically update checkpoint after each yielded fault.

        Yields:
            A tuple (fault_record_list, idx), where:
            - fault_record_list is a list containing one fault record dict
            - idx is the fault row index in the CSV
        """
        if not os.path.exists(self.fault_list_path):
            raise FileNotFoundError(f"Fault list file not found: {self.fault_list_path}")

        # Load fault table each time to reflect external updates.
        self.fault_df = pd.read_csv(self.fault_list_path)
        start_idx = self.state.fault_idx if from_ckpt else 0

        for idx in range(start_idx, len(self.fault_df)):
            # Keep compatibility with existing injection API (list[dict] with one element).
            yield (self.fault_df.iloc[[idx]].to_dict("records"), idx)

            # Advance only after successful consumer step.
            if auto_advance:
                self.advance(idx)