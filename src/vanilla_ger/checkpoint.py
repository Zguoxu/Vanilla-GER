"""Explicit checkpoint loading for the pinned Transformers 4.46 single-device path.

Keep Trainer's checkpoint format, but avoid its implicit torch.load defaults.
Only the NumPy types needed by the saved MT19937 RNG state are allowlisted,
inside one load call. Unsupported objects fail instead of using pickle fallback.
"""
import logging
from pathlib import Path
import random

import numpy as np
import torch

logger = logging.getLogger(__name__)


class CheckpointResumeMixin:
    """Resume ordinary AdamW training on one CPU, MPS device, or CUDA GPU."""

    def _require_single_device_resume(self):
        if (self.args.world_size != 1 or self.is_deepspeed_enabled
                or self.is_fsdp_enabled or self.args.device.type not in {"cpu", "mps", "cuda"}):
            raise ValueError("Checkpoint resume supports the project's single-device CPU/MPS/CUDA path only")

    def _load_optimizer_and_scheduler(self, checkpoint):
        if checkpoint is None:
            return
        self._require_single_device_resume()
        optimizer_path = Path(checkpoint) / "optimizer.pt"
        scheduler_path = Path(checkpoint) / "scheduler.pt"
        if optimizer_path.is_file() and scheduler_path.is_file():
            # Match Trainer's single-GPU CPU staging to avoid extra GPU memory.
            self.optimizer.load_state_dict(torch.load(optimizer_path, map_location="cpu", weights_only=True))
            self.lr_scheduler.load_state_dict(torch.load(scheduler_path, map_location="cpu", weights_only=True))

    def _load_rng_state(self, checkpoint):
        if checkpoint is None:
            return
        self._require_single_device_resume()
        path = Path(checkpoint) / "rng_state.pth"
        if not path.is_file():
            logger.warning("RNG checkpoint missing at %s; exact continuation is not guaranteed", path)
            return
        # Trainer stores NumPy's MT19937 state as an ndarray with uint32 dtype.
        numpy_rng_types = [np.core.multiarray._reconstruct, np.ndarray,
                           np.dtype, type(np.dtype(np.uint32))]
        with torch.serialization.safe_globals(numpy_rng_types):
            state = torch.load(path, map_location="cpu", weights_only=True)
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.random.set_rng_state(state["cpu"])
        if torch.cuda.is_available():
            if "cuda" in state:
                torch.cuda.random.set_rng_state(state["cuda"])
            else:
                logger.warning("CUDA RNG state missing; exact GPU continuation is not guaranteed")
