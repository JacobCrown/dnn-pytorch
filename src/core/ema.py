"""Exponential moving average helper for PyTorch modules."""

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import torch
import torch.nn as nn


class ModelEMA:
    """Track an exponential moving average of floating-point model parameters."""

    def __init__(
        self,
        model: nn.Module,
        *,
        decay: float,
        update_after_step: int = 0,
        update_every: int = 1,
    ) -> None:
        if not 0.0 <= decay < 1.0:
            raise ValueError(f"EMA decay must be in [0, 1), got {decay}")
        if update_after_step < 0:
            raise ValueError(
                f"EMA update_after_step must be non-negative, got {update_after_step}"
            )
        if update_every < 1:
            raise ValueError(f"EMA update_every must be >= 1, got {update_every}")

        self.decay = float(decay)
        self.update_after_step = int(update_after_step)
        self.update_every = int(update_every)
        self.optimization_step = 0
        self.num_updates = 0
        self.shadow_params = self._clone_model_params(model)

    @staticmethod
    def _clone_model_params(model: nn.Module) -> dict[str, torch.Tensor]:
        return {
            name: param.detach().clone()
            for name, param in model.named_parameters()
            if param.is_floating_point()
        }

    def update(self, model: nn.Module) -> None:
        """Update EMA parameters after an optimizer step."""
        self.optimization_step += 1
        if self.optimization_step <= self.update_after_step:
            self._copy_from(model)
            return
        if (self.optimization_step - self.update_after_step) % self.update_every != 0:
            return

        with torch.no_grad():
            model_params = dict(model.named_parameters())
            for name, shadow in self.shadow_params.items():
                param = model_params[name].detach()
                shadow.lerp_(param, 1.0 - self.decay)
        self.num_updates += 1

    def _copy_from(self, model: nn.Module) -> None:
        with torch.no_grad():
            model_params = dict(model.named_parameters())
            for name, shadow in self.shadow_params.items():
                shadow.copy_(model_params[name].detach())

    def model_state_dict(self, model: nn.Module) -> dict[str, torch.Tensor]:
        """Return a full module state dict with EMA parameters and current buffers."""
        state_dict = {
            key: value.detach().clone() for key, value in model.state_dict().items()
        }
        for name, shadow in self.shadow_params.items():
            state_dict[name] = shadow.detach().clone()
        return state_dict

    @contextmanager
    def average_parameters(self, model: nn.Module) -> Iterator[None]:
        """Temporarily copy EMA parameters into ``model`` and restore afterwards."""
        online_params = self._clone_model_params(model)
        self.copy_to(model)
        try:
            yield
        finally:
            self._load_params_into_model(model, online_params)

    def copy_to(self, model: nn.Module) -> None:
        """Copy EMA parameters into ``model``."""
        self._load_params_into_model(model, self.shadow_params)

    @staticmethod
    def _load_params_into_model(
        model: nn.Module, params: dict[str, torch.Tensor]
    ) -> None:
        with torch.no_grad():
            model_params = dict(model.named_parameters())
            for name, tensor in params.items():
                model_params[name].copy_(tensor)

    def state_dict(self) -> dict[str, Any]:
        """Return serializable EMA training state."""
        return {
            "decay": self.decay,
            "update_after_step": self.update_after_step,
            "update_every": self.update_every,
            "optimization_step": self.optimization_step,
            "num_updates": self.num_updates,
            "shadow_params": {
                key: value.detach().clone() for key, value in self.shadow_params.items()
            },
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore EMA state from a checkpoint."""
        self.decay = float(state_dict["decay"])
        self.update_after_step = int(state_dict["update_after_step"])
        self.update_every = int(state_dict["update_every"])
        self.optimization_step = int(state_dict["optimization_step"])
        self.num_updates = int(state_dict["num_updates"])
        loaded_params = state_dict["shadow_params"]
        if loaded_params.keys() != self.shadow_params.keys():
            missing = self.shadow_params.keys() - loaded_params.keys()
            unexpected = loaded_params.keys() - self.shadow_params.keys()
            raise ValueError(
                "EMA shadow parameter keys do not match model parameters. "
                f"Missing: {sorted(missing)}; unexpected: {sorted(unexpected)}"
            )
        self.shadow_params = {
            key: value.detach().clone() for key, value in loaded_params.items()
        }
