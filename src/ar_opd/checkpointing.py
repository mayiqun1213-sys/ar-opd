"""Safe, resumable training checkpoints for AR-OPD."""

from __future__ import annotations

import os
import random
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import torch
from torch import nn

from ar_opd.distillation import LocalSFTDataset


_FORMAT_VERSION = 1


@dataclass(frozen=True)
class LoadedTrainingCheckpoint:
    """Non-parameter state returned after restoring a training checkpoint."""

    completed_updates: int
    config: dict[str, Any]
    metrics: list[dict[str, Any]]
    local_sft_evaluations: list[dict[str, Any]]
    local_sft_replay: LocalSFTDataset


def _safe_value(value: Any, *, field: str) -> Any:
    """Convert metadata to types accepted by ``torch.load(weights_only=True)``."""

    if is_dataclass(value) and not isinstance(value, type):
        return _safe_value(asdict(value), field=field)
    if isinstance(value, Enum):
        return _safe_value(value.value, field=field)
    if isinstance(value, Path | torch.device):
        return str(value)
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, Mapping):
        converted: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{field} mapping keys must be strings")
            converted[key] = _safe_value(item, field=f"{field}.{key}")
        return converted
    if isinstance(value, tuple):
        return tuple(
            _safe_value(item, field=f"{field}[{index}]")
            for index, item in enumerate(value)
        )
    if isinstance(value, list):
        return [
            _safe_value(item, field=f"{field}[{index}]")
            for index, item in enumerate(value)
        ]
    raise TypeError(
        f"{field} contains unsupported checkpoint value {type(value).__name__}"
    )


def _metadata_rows(
    rows: Sequence[Mapping[str, Any]], *, field: str
) -> list[dict[str, Any]]:
    converted = _safe_value(list(rows), field=field)
    if not isinstance(converted, list) or any(
        not isinstance(row, dict) for row in converted
    ):
        raise TypeError(f"{field} must be a sequence of mappings")
    return converted


def _capture_rng_state(generator: torch.Generator) -> dict[str, Any]:
    cuda_states = (
        [state.clone() for state in torch.cuda.get_rng_state_all()]
        if generator.device.type == "cuda" and torch.cuda.is_available()
        else []
    )
    return {
        "generator": generator.get_state().clone(),
        "python": random.getstate(),
        "torch": torch.get_rng_state().clone(),
        "cuda": cuda_states,
    }


def save_training_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    ppo_optimizer: torch.optim.Optimizer,
    completed_updates: int,
    config: Mapping[str, Any] | Any,
    metrics: Sequence[Mapping[str, Any]],
    local_sft_evaluations: Sequence[Mapping[str, Any]],
    local_sft_replay: LocalSFTDataset,
    generator: torch.Generator,
) -> Path:
    """Save all state needed to continue training deterministically."""

    if isinstance(completed_updates, bool) or not isinstance(completed_updates, int):
        raise TypeError("completed_updates must be an integer")
    if completed_updates < 0:
        raise ValueError("completed_updates must be a non-negative integer")

    if is_dataclass(config) and not isinstance(config, type):
        config = asdict(config)
    if not isinstance(config, Mapping):
        raise TypeError("config must be a mapping or dataclass instance")

    # Import lazily so this module remains importable while replay migrations
    # are being developed independently.
    from ar_opd.distillation_replay import local_sft_replay_state_dict

    payload = {
        "format_version": _FORMAT_VERSION,
        "model_state_dict": model.state_dict(),
        "ppo_optimizer_state_dict": ppo_optimizer.state_dict(),
        "completed_updates": completed_updates,
        "config": _safe_value(config, field="config"),
        "metrics": _metadata_rows(metrics, field="metrics"),
        "local_sft_evaluations": _metadata_rows(
            local_sft_evaluations, field="local_sft_evaluations"
        ),
        "local_sft_replay": _safe_value(
            local_sft_replay_state_dict(local_sft_replay),
            field="local_sft_replay",
        ),
        "rng_state": _capture_rng_state(generator),
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    os.close(descriptor)
    try:
        with temporary_path.open("wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        temporary_path.replace(destination)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    return destination


def _mapping(value: Any, *, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"checkpoint {field} must be a dictionary")
    return value


def _rows(value: Any, *, field: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or any(not isinstance(row, dict) for row in value):
        raise ValueError(f"checkpoint {field} must be a list of dictionaries")
    return value


def _validate_rng_state(
    value: Any, *, require_restorable_cuda: bool
) -> dict[str, Any]:
    state = _mapping(value, field="rng_state")
    if set(state) != {"generator", "python", "torch", "cuda"}:
        raise ValueError("checkpoint rng_state has unexpected fields")
    if not isinstance(state["generator"], torch.Tensor):
        raise ValueError("checkpoint generator RNG state must be a tensor")
    if not isinstance(state["python"], tuple):
        raise ValueError("checkpoint Python RNG state must be a tuple")
    if not isinstance(state["torch"], torch.Tensor):
        raise ValueError("checkpoint Torch RNG state must be a tensor")
    if not isinstance(state["cuda"], list) or any(
        not isinstance(row, torch.Tensor) for row in state["cuda"]
    ):
        raise ValueError("checkpoint CUDA RNG state must be a list of tensors")
    probe = random.Random()
    try:
        probe.setstate(state["python"])
    except (TypeError, ValueError) as error:
        raise ValueError("checkpoint Python RNG state is invalid") from error
    if state["cuda"] and require_restorable_cuda:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "checkpoint contains CUDA RNG state but CUDA is unavailable; "
                "load with restore_rng=False for a non-resumable inspection"
            )
        if len(state["cuda"]) != torch.cuda.device_count():
            raise RuntimeError("checkpoint CUDA RNG state does not match device count")
    return state


def load_training_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    ppo_optimizer: torch.optim.Optimizer,
    generator: torch.Generator,
    map_location: str | torch.device | None = None,
    restore_rng: bool = True,
) -> LoadedTrainingCheckpoint:
    """Load a safe checkpoint and restore model, optimizer, and RNG state."""

    payload = torch.load(path, map_location=map_location, weights_only=True)
    payload = _mapping(payload, field="root")
    if payload.get("format_version") != _FORMAT_VERSION:
        raise ValueError("unsupported training checkpoint format version")

    completed_updates = payload.get("completed_updates")
    if (
        isinstance(completed_updates, bool)
        or not isinstance(completed_updates, int)
        or completed_updates < 0
    ):
        raise ValueError("checkpoint completed_updates must be a non-negative integer")
    config = _mapping(payload.get("config"), field="config")
    metrics = _rows(payload.get("metrics"), field="metrics")
    evaluations = _rows(
        payload.get("local_sft_evaluations"), field="local_sft_evaluations"
    )
    model_state = _mapping(payload.get("model_state_dict"), field="model_state_dict")
    optimizer_state = _mapping(
        payload.get("ppo_optimizer_state_dict"), field="ppo_optimizer_state_dict"
    )
    rng_state = _validate_rng_state(
        payload.get("rng_state"), require_restorable_cuda=restore_rng
    )

    from ar_opd.distillation_replay import load_local_sft_replay_state_dict

    replay = load_local_sft_replay_state_dict(payload.get("local_sft_replay"))

    model.load_state_dict(model_state)
    ppo_optimizer.load_state_dict(optimizer_state)
    if restore_rng:
        generator.set_state(rng_state["generator"].cpu())
        random.setstate(rng_state["python"])
        torch.set_rng_state(rng_state["torch"].cpu())
        if rng_state["cuda"]:
            torch.cuda.set_rng_state_all(
                [state.cpu() for state in rng_state["cuda"]]
            )

    return LoadedTrainingCheckpoint(
        completed_updates=completed_updates,
        config=config,
        metrics=metrics,
        local_sft_evaluations=evaluations,
        local_sft_replay=replay,
    )
