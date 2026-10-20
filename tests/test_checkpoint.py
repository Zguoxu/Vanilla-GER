"""Check continuation behavior and rejection of unsupported checkpoint objects."""
from datetime import datetime
import pickle
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vanilla_ger.checkpoint import CheckpointResumeMixin


class ResumeHarness(CheckpointResumeMixin):
    def __init__(self, parameter):
        self.args = SimpleNamespace(world_size=1, device=torch.device("cpu"))
        self.is_deepspeed_enabled = self.is_fsdp_enabled = False
        self.optimizer = torch.optim.AdamW([parameter], lr=0.1)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.9)


def random_update(trainer, parameter):
    trainer.optimizer.zero_grad()
    parameter.grad = torch.ones_like(parameter) * (random.random() + np.random.random() + torch.rand(()))
    trainer.optimizer.step()
    trainer.lr_scheduler.step()


def test_resume_matches_next_update_and_restores_rng(tmp_path):
    original_rng = (random.getstate(), np.random.get_state(), torch.get_rng_state())
    try:
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        parameter = torch.nn.Parameter(torch.tensor([1.0]))
        original = ResumeHarness(parameter)
        random_update(original, parameter)
        saved_parameter = parameter.detach().clone()
        torch.save(original.optimizer.state_dict(), tmp_path / "optimizer.pt")
        torch.save(original.lr_scheduler.state_dict(), tmp_path / "scheduler.pt")
        torch.save({"python": random.getstate(), "numpy": np.random.get_state(),
                    "cpu": torch.get_rng_state()}, tmp_path / "rng_state.pth")
        random_update(original, parameter)
        expected_draws = (random.random(), np.random.random(), torch.rand(()))

        resumed_parameter = torch.nn.Parameter(saved_parameter)
        resumed = ResumeHarness(resumed_parameter)
        resumed._load_optimizer_and_scheduler(tmp_path)
        resumed._load_rng_state(tmp_path)
        random_update(resumed, resumed_parameter)
        actual_draws = (random.random(), np.random.random(), torch.rand(()))
        assert torch.equal(resumed_parameter, parameter)
        assert actual_draws == expected_draws
        assert resumed.lr_scheduler.state_dict() == original.lr_scheduler.state_dict()
        assert resumed.optimizer.param_groups[0]["lr"] == original.optimizer.param_groups[0]["lr"]
    finally:
        random.setstate(original_rng[0])
        np.random.set_state(original_rng[1])
        torch.set_rng_state(original_rng[2])


@pytest.mark.parametrize("filename", ["optimizer.pt", "rng_state.pth"])
def test_resume_rejects_unsupported_objects_without_global_allowlist_changes(tmp_path, filename):
    trainer = ResumeHarness(torch.nn.Parameter(torch.tensor([1.0])))
    torch.save(trainer.lr_scheduler.state_dict(), tmp_path / "scheduler.pt")
    torch.save({"unsupported": datetime(2026, 1, 1)}, tmp_path / filename)
    allowed_before = list(torch.serialization.get_safe_globals())
    load = trainer._load_rng_state if filename == "rng_state.pth" else trainer._load_optimizer_and_scheduler
    with pytest.raises(pickle.UnpicklingError, match="Unsupported global"):
        load(tmp_path)
    assert torch.serialization.get_safe_globals() == allowed_before
