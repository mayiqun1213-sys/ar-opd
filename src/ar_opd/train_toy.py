"""Command-line entry point for the end-to-end toy AR-OPD training loop."""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import MISSING, asdict, dataclass, fields
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable

import torch

from ar_opd.checkpointing import (
    load_training_checkpoint,
    save_training_checkpoint,
)
from ar_opd.core import EpisodeRollout
from ar_opd.distillation import (
    LocalSFTConfig,
    LocalSFTDataset,
    LocalSFTExample,
    extract_local_sft_examples,
    local_sft_update,
)
from ar_opd.distillation_replay import append_local_sft_replay
from ar_opd.models import ActorCritic
from ar_opd.opd import (
    OPDAnnotationLedger,
    OPDConfig,
    OPDDataset,
    extract_student_only_opd,
    opd_update,
)
from ar_opd.ppo import PPOConfig, build_batch, ppo_update
from ar_opd.rollout import RolloutConfig, collect_episodes
from ar_opd.toy_env import JammedChainConfig
from ar_opd.toy_runtime import ToyOracleDistributionAnnotator, ToyRuntimeAdapter


_METRIC_SCHEMA_VERSION = 2
_OPD_CONFIG_FIELDS = frozenset(
    {
        "opd_episodes_per_update",
        "opd_epochs",
        "opd_learning_rate",
        "opd_target_temperature",
        "opd_teacher_preferred_probability",
        "opd_annotation_query_cost",
        "opd_max_grad_norm",
    }
)
_BASE_OPD_METRIC_FIELDS = frozenset(
    {
        "opd_examples",
        "opd_optimizer_steps",
        "opd_loss",
        "opd_kl_before",
        "opd_kl_after",
        "opd_student_entropy_before",
        "opd_student_entropy_after",
        "opd_teacher_entropy",
        "opd_disagreement_before",
        "opd_disagreement_after",
        "opd_proposal_disagreement",
        "opd_enabled",
        "opd_collection_id",
        "opd_annotation_query_count",
        "opd_annotation_scored_actions",
        "opd_annotation_query_cost",
        "opd_rollout_episodes",
        "opd_rollout_actor_rows",
        "opd_rollout_success_rate",
        "opd_rollout_mean_task_return",
        "opd_rollout_mean_steps",
        "opd_rollout_teacher_probe_count",
        "opd_rollout_teacher_query_count",
        "opd_rollout_teacher_executed_steps",
        "opd_rollout_teacher_cost",
        "total_teacher_resource_cost",
    }
)
_ENABLED_OPD_METRIC_FIELDS = frozenset(
    {
        "student_only_success_before_opd",
        "student_only_success_after_opd",
        "student_only_return_before_opd",
        "student_only_return_after_opd",
    }
)


@dataclass(frozen=True)
class ToyTrainConfig:
    seed: int = 7
    device: str = "cpu"
    updates: int = 3
    episodes_per_update: int = 8
    evaluation_episodes: int = 8
    hidden_size: int = 32
    learning_rate: float = 0.003
    goal_position: int = 5
    trap_positions: tuple[int, ...] = (2, 4)
    max_steps: int = 16
    probe_probability: float = 1.0
    recovery_horizon: int = 2
    teacher_query_cost: float = 0.01
    teacher_execution_cost: float = 0.02
    gamma: float = 0.97
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.01
    max_grad_norm: float = 1.0
    ppo_epochs: int = 4
    local_sft_epochs: int = 0
    local_sft_learning_rate: float = 0.01
    corrective_sft_coefficient: float = 1.0
    fallback_sft_coefficient: float = 1.0
    local_sft_max_grad_norm: float = 1.0
    local_sft_replay_capacity_per_kind: int = 256
    opd_episodes_per_update: int = 0
    opd_epochs: int = 0
    opd_learning_rate: float = 0.1
    opd_target_temperature: float = 1.0
    opd_teacher_preferred_probability: float = 0.95
    opd_annotation_query_cost: float = 0.01
    opd_max_grad_norm: float = 1.0
    output_dir: str = "outputs/toy_smoke"

    def __post_init__(self) -> None:
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed must be an integer")
        for name in ("opd_episodes_per_update", "opd_epochs"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
        if self.updates < 1:
            raise ValueError("updates must be positive")
        if self.episodes_per_update < 1 or self.evaluation_episodes < 1:
            raise ValueError("training and evaluation episode counts must be positive")
        if self.hidden_size < 1:
            raise ValueError("hidden_size must be positive")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if self.local_sft_replay_capacity_per_kind < 1:
            raise ValueError("local SFT replay capacity must be positive")
        if self.opd_episodes_per_update < 0:
            raise ValueError("OPD episode count must be non-negative")
        if self.opd_epochs > 0 and self.opd_episodes_per_update < 1:
            raise ValueError("enabled OPD requires at least one Student-only episode")
        self.opd_config()
        self.opd_annotator()

    @classmethod
    def from_json(cls, path: str | Path) -> ToyTrainConfig:
        with Path(path).open(encoding="utf-8") as stream:
            values = json.load(stream)
        if "trap_positions" in values:
            values["trap_positions"] = tuple(values["trap_positions"])
        return cls(**values)

    def environment_config(self) -> JammedChainConfig:
        return JammedChainConfig(
            goal_position=self.goal_position,
            trap_positions=self.trap_positions,
            max_steps=self.max_steps,
        )

    def rollout_config(self, probe_probability: float | None = None) -> RolloutConfig:
        return RolloutConfig(
            gamma=self.gamma,
            probe_probability=(
                self.probe_probability if probe_probability is None else probe_probability
            ),
            recovery_horizon=self.recovery_horizon,
            teacher_query_cost=self.teacher_query_cost,
            teacher_execution_cost=self.teacher_execution_cost,
        )

    def ppo_config(self) -> PPOConfig:
        return PPOConfig(
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            clip_ratio=self.clip_ratio,
            value_coefficient=self.value_coefficient,
            entropy_coefficient=self.entropy_coefficient,
            max_grad_norm=self.max_grad_norm,
            epochs=self.ppo_epochs,
        )

    def local_sft_config(self) -> LocalSFTConfig:
        return LocalSFTConfig(
            epochs=self.local_sft_epochs,
            learning_rate=self.local_sft_learning_rate,
            corrective_coefficient=self.corrective_sft_coefficient,
            fallback_coefficient=self.fallback_sft_coefficient,
            max_grad_norm=self.local_sft_max_grad_norm,
        )

    def opd_config(self) -> OPDConfig:
        return OPDConfig(
            epochs=self.opd_epochs,
            learning_rate=self.opd_learning_rate,
            target_temperature=self.opd_target_temperature,
            max_grad_norm=self.opd_max_grad_norm,
        )

    def opd_annotator(self) -> ToyOracleDistributionAnnotator:
        return ToyOracleDistributionAnnotator(
            preferred_probability=self.opd_teacher_preferred_probability,
            query_cost=self.opd_annotation_query_cost,
        )


def _aggregate_episodes(episodes: list[EpisodeRollout]) -> dict[str, float]:
    if not episodes:
        raise ValueError("at least one episode is required")
    return {
        "episodes": float(len(episodes)),
        "success_rate": fmean(float(episode.success) for episode in episodes),
        "mean_task_return": fmean(episode.task_return for episode in episodes),
        "mean_net_return": fmean(episode.net_return for episode in episodes),
        "mean_steps": fmean(len(episode.transitions) for episode in episodes),
        "actor_rows": float(sum(episode.actor_rows for episode in episodes)),
        "teacher_probe_count": float(
            sum(episode.teacher_costs.probe_count for episode in episodes)
        ),
        "teacher_query_count": float(
            sum(episode.teacher_costs.query_count for episode in episodes)
        ),
        "teacher_generated_steps": float(
            sum(episode.teacher_costs.generated_teacher_steps for episode in episodes)
        ),
        "teacher_executed_steps": float(
            sum(episode.teacher_costs.executed_teacher_steps for episode in episodes)
        ),
        "teacher_query_cost": sum(
            episode.teacher_costs.query_cost for episode in episodes
        ),
        "teacher_execution_cost": sum(
            episode.teacher_costs.execution_cost for episode in episodes
        ),
    }


def _collect_episodes(
    model: ActorCritic,
    config: ToyTrainConfig,
    *,
    count: int,
    seed: int,
    probe_probability: float,
    deterministic_student: bool,
    generator: torch.Generator,
) -> list[EpisodeRollout]:
    return collect_episodes(
        ToyRuntimeAdapter(config.environment_config()),
        model,
        config.rollout_config(probe_probability),
        count=count,
        seed=seed,
        deterministic_student=deterministic_student,
        generator=generator,
    )


def evaluate(
    model: ActorCritic,
    config: ToyTrainConfig,
    *,
    probe_probability: float,
    seed: int,
    generator: torch.Generator,
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    episodes = _collect_episodes(
        model,
        config,
        count=config.evaluation_episodes,
        seed=seed,
        probe_probability=probe_probability,
        deterministic_student=True,
        generator=generator,
    )
    model.train(was_training)
    return _aggregate_episodes(episodes)


def _validate_student_only_metrics(metrics: dict[str, float]) -> None:
    teacher_keys = (
        "teacher_probe_count",
        "teacher_query_count",
        "teacher_generated_steps",
        "teacher_executed_steps",
        "teacher_query_cost",
        "teacher_execution_cost",
    )
    if any(metrics[key] != 0.0 for key in teacher_keys):
        raise AssertionError("student-only evaluation used Teacher resources")


def _clear_optimizer_state(
    optimizer: torch.optim.Optimizer,
    parameters: Iterable[torch.nn.Parameter],
) -> int:
    """Drop stale PPO moments after another optimizer changes actor weights."""

    cleared = 0
    for parameter in parameters:
        if parameter in optimizer.state:
            del optimizer.state[parameter]
            cleared += 1
    return cleared


def _segment_count(examples: tuple[LocalSFTExample, ...]) -> int:
    return len(
        {
            (
                example.collection_id,
                example.episode_index,
                example.decision_id,
                example.kind,
            )
            for example in examples
        }
    )


def _validate_resume_configuration(
    saved_config: dict[str, Any],
    config: ToyTrainConfig,
    completed_updates: int,
) -> bool:
    """Validate immutable configuration and report a strict M2a migration."""

    if completed_updates > config.updates:
        raise ValueError(
            "resume checkpoint has more completed updates than the requested run"
        )
    current = asdict(config)
    current_keys = set(current)
    saved_keys = set(saved_config)
    legacy_keys = current_keys - _OPD_CONFIG_FIELDS
    field_defaults = {field.name: field.default for field in fields(config)}

    if saved_keys == legacy_keys:
        nondefault_opd_fields = sorted(
            name
            for name in _OPD_CONFIG_FIELDS
            if field_defaults[name] is MISSING
            or current[name] != field_defaults[name]
        )
        if nondefault_opd_fields:
            raise ValueError(
                "resume config differs in immutable fields: "
                + ", ".join(nondefault_opd_fields)
            )
        saved_with_defaults = {
            **saved_config,
            **{name: field_defaults[name] for name in _OPD_CONFIG_FIELDS},
        }
        is_legacy_m2a = True
    elif saved_keys == current_keys:
        saved_with_defaults = dict(saved_config)
        is_legacy_m2a = False
    else:
        mismatches = sorted(current_keys ^ saved_keys)
        raise ValueError(
            "resume config differs in immutable fields: " + ", ".join(mismatches)
        )

    ignored = {"updates", "output_dir"}
    saved_comparable = {
        key: value for key, value in saved_with_defaults.items() if key not in ignored
    }
    current_comparable = {
        key: value for key, value in current.items() if key not in ignored
    }
    if saved_comparable != current_comparable:
        mismatches = sorted(
            key
            for key in saved_comparable.keys() | current_comparable.keys()
            if saved_comparable.get(key) != current_comparable.get(key)
        )
        raise ValueError(
            "resume config differs in immutable fields: " + ", ".join(mismatches)
        )
    return is_legacy_m2a


def _disabled_opd_metric_values() -> dict[str, float]:
    return {
        "opd_examples": 0.0,
        "opd_optimizer_steps": 0.0,
        "opd_loss": 0.0,
        "opd_kl_before": 0.0,
        "opd_kl_after": 0.0,
        "opd_student_entropy_before": 0.0,
        "opd_student_entropy_after": 0.0,
        "opd_teacher_entropy": 0.0,
        "opd_disagreement_before": 0.0,
        "opd_disagreement_after": 0.0,
        "opd_proposal_disagreement": 0.0,
        "opd_enabled": 0.0,
        "opd_collection_id": -1.0,
        "opd_annotation_query_count": 0.0,
        "opd_annotation_scored_actions": 0.0,
        "opd_annotation_query_cost": 0.0,
        "opd_rollout_episodes": 0.0,
        "opd_rollout_actor_rows": 0.0,
        "opd_rollout_success_rate": 0.0,
        "opd_rollout_mean_task_return": 0.0,
        "opd_rollout_mean_steps": 0.0,
        "opd_rollout_teacher_probe_count": 0.0,
        "opd_rollout_teacher_query_count": 0.0,
        "opd_rollout_teacher_executed_steps": 0.0,
        "opd_rollout_teacher_cost": 0.0,
    }


def _migrate_m2a_metric_rows(
    metrics: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Backfill the disabled OPD schema only for unambiguous M2a rows."""

    migrated: list[dict[str, Any]] = []
    for row in metrics:
        has_m2b_trace = (
            "metric_schema_version" in row
            or "total_teacher_resource_cost" in row
            or any(key.startswith("opd_") or key.endswith("_opd") for key in row)
        )
        if has_m2b_trace:
            raise ValueError("legacy M2a metrics contain M2b OPD fields")
        migrated.append(
            {
                **row,
                "metric_schema_version": float(_METRIC_SCHEMA_VERSION),
                **_disabled_opd_metric_values(),
                "total_teacher_resource_cost": (
                    float(row["teacher_query_cost"])
                    + float(row["teacher_execution_cost"])
                ),
            }
        )
    return migrated


def _finite_metric(row: dict[str, Any], key: str) -> float:
    if key not in row:
        raise ValueError(f"checkpoint metric is missing {key}")
    value = row[key]
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(value)
    ):
        raise ValueError(f"checkpoint metric {key} must be a finite real number")
    return float(value)


def _validate_restored_history(
    completed_updates: int,
    metrics: list[dict[str, Any]],
    local_sft_evaluations: list[dict[str, Any]],
    local_sft_replay: LocalSFTDataset,
    *,
    config: ToyTrainConfig,
    expected_action_size: int,
) -> None:
    if len(metrics) != completed_updates:
        raise ValueError("checkpoint metric history does not match completed_updates")
    for index, row in enumerate(metrics, start=1):
        if (
            isinstance(row.get("update"), bool)
            or row.get("update") != float(index)
        ):
            raise ValueError("checkpoint metric update ids must be contiguous")
        if row.get("metric_schema_version") != float(_METRIC_SCHEMA_VERSION):
            raise ValueError("checkpoint metric schema version is unsupported")
        missing_opd_fields = sorted(_BASE_OPD_METRIC_FIELDS - row.keys())
        if missing_opd_fields:
            raise ValueError(
                "checkpoint metrics lack OPD fields: "
                + ", ".join(missing_opd_fields)
            )
        for key in sorted(_BASE_OPD_METRIC_FIELDS):
            _finite_metric(row, key)
        cost_fields = (
            "teacher_query_cost",
            "teacher_execution_cost",
            "opd_annotation_query_cost",
            "opd_rollout_teacher_cost",
            "total_teacher_resource_cost",
        )
        for key in cost_fields:
            if _finite_metric(row, key) < 0.0:
                raise ValueError("checkpoint Teacher resource costs must be non-negative")

        opd_enabled = float(config.opd_epochs > 0)
        if row["opd_enabled"] != opd_enabled:
            raise ValueError("checkpoint OPD metrics disagree with configuration")
        expected_collection_id = float(index - 1) if opd_enabled else -1.0
        if row["opd_collection_id"] != expected_collection_id:
            raise ValueError("checkpoint OPD collection ids are inconsistent")
        if opd_enabled:
            missing_enabled_fields = sorted(
                _ENABLED_OPD_METRIC_FIELDS - row.keys()
            )
            if missing_enabled_fields:
                raise ValueError(
                    "checkpoint metrics lack enabled OPD fields: "
                    + ", ".join(missing_enabled_fields)
                )
            for key in sorted(_ENABLED_OPD_METRIC_FIELDS):
                _finite_metric(row, key)
            if row["opd_optimizer_steps"] != float(config.opd_epochs):
                raise ValueError("checkpoint OPD steps disagree with configuration")
            expected_examples = row["opd_examples"]
            count_fields = (
                "opd_examples",
                "opd_annotation_query_count",
                "opd_annotation_scored_actions",
                "opd_rollout_episodes",
                "opd_rollout_actor_rows",
            )
            if any(not float(row[key]).is_integer() for key in count_fields):
                raise ValueError("checkpoint OPD counts must be integral")
            if (
                expected_examples <= 0.0
                or row["opd_annotation_query_count"] != expected_examples
                or row["opd_rollout_actor_rows"] != expected_examples
                or row["opd_rollout_episodes"]
                != float(config.opd_episodes_per_update)
                or row["opd_annotation_scored_actions"]
                != expected_action_size * expected_examples
            ):
                raise ValueError("checkpoint OPD collection metrics are inconsistent")
            expected_annotation_cost = (
                config.opd_annotation_query_cost * expected_examples
            )
            if not math.isclose(
                row["opd_annotation_query_cost"],
                expected_annotation_cost,
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise ValueError("checkpoint OPD annotation cost is inconsistent")
            teacher_fields = (
                "opd_rollout_teacher_probe_count",
                "opd_rollout_teacher_query_count",
                "opd_rollout_teacher_executed_steps",
                "opd_rollout_teacher_cost",
            )
            if any(row[key] != 0.0 for key in teacher_fields):
                raise ValueError("checkpoint OPD rollout used Teacher resources")
            if not math.isclose(
                row["opd_rollout_mean_steps"]
                * row["opd_rollout_episodes"],
                row["opd_rollout_actor_rows"],
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise ValueError("checkpoint OPD rollout lengths are inconsistent")
            rate_fields = (
                "opd_rollout_success_rate",
                "opd_disagreement_before",
                "opd_disagreement_after",
                "opd_proposal_disagreement",
                "student_only_success_before_opd",
                "student_only_success_after_opd",
            )
            if any(not 0.0 <= row[key] <= 1.0 for key in rate_fields):
                raise ValueError("checkpoint OPD rates must lie in [0, 1]")
        else:
            if _ENABLED_OPD_METRIC_FIELDS & row.keys():
                raise ValueError(
                    "checkpoint has enabled-stage metrics for disabled OPD"
                )
            disabled_values = _disabled_opd_metric_values()
            if any(row[key] != value for key, value in disabled_values.items()):
                raise ValueError("checkpoint has nonzero metrics for disabled OPD")

        expected_total_cost = (
            row["teacher_query_cost"]
            + row["teacher_execution_cost"]
            + row["opd_annotation_query_cost"]
        )
        if not math.isclose(
            row["total_teacher_resource_cost"],
            expected_total_cost,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError("checkpoint Teacher resource costs are inconsistent")
    expected_evaluations = (
        completed_updates if config.local_sft_epochs > 0 else 0
    )
    if len(local_sft_evaluations) != expected_evaluations:
        raise ValueError("checkpoint local-SFT evaluations do not match training history")
    for index, row in enumerate(local_sft_evaluations, start=1):
        if row.get("update") != index:
            raise ValueError("checkpoint local-SFT evaluation ids must be contiguous")
    replay_examples = (
        *local_sft_replay.corrective,
        *local_sft_replay.fallback,
    )
    if any(
        example.collection_id >= completed_updates for example in replay_examples
    ):
        raise ValueError("checkpoint replay provenance exceeds completed updates")
    if metrics:
        expected_replay_sizes = {
            "replay_corrective_sft_examples": float(
                len(local_sft_replay.corrective)
            ),
            "replay_fallback_sft_examples": float(len(local_sft_replay.fallback)),
        }
        if any(
            metrics[-1].get(key) != value
            for key, value in expected_replay_sizes.items()
        ):
            raise ValueError("checkpoint replay metrics disagree with replay state")


def _write_metric(stream: Any, metrics: dict[str, Any]) -> None:
    stream.write(json.dumps(metrics, sort_keys=True) + "\n")


def run_training(
    config: ToyTrainConfig,
    *,
    output_dir: str | Path | None = None,
    resume_from: str | Path | None = None,
) -> dict[str, Any]:
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    runtime_adapter = ToyRuntimeAdapter(config.environment_config())
    model = ActorCritic(
        runtime_adapter.spec.observation_size,
        runtime_adapter.spec.action_size,
        config.hidden_size,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    generator = torch.Generator(device=device.type).manual_seed(config.seed)
    destination = Path(output_dir or config.output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    metrics_path = destination / "metrics.jsonl"
    checkpoint_path = destination / "checkpoint.pt"

    start_update = 0
    update_metrics: list[dict[str, Any]] = []
    local_sft_evaluations: list[dict[str, Any]] = []
    local_sft_replay = LocalSFTDataset()
    if resume_from is not None:
        loaded = load_training_checkpoint(
            resume_from,
            model=model,
            ppo_optimizer=optimizer,
            generator=generator,
            map_location=device,
        )
        start_update = loaded.completed_updates
        is_legacy_m2a = _validate_resume_configuration(
            loaded.config,
            config,
            start_update,
        )
        update_metrics = [dict(row) for row in loaded.metrics]
        if is_legacy_m2a:
            update_metrics = _migrate_m2a_metric_rows(update_metrics)
        local_sft_evaluations = [
            dict(row) for row in loaded.local_sft_evaluations
        ]
        local_sft_replay = loaded.local_sft_replay
        _validate_restored_history(
            start_update,
            update_metrics,
            local_sft_evaluations,
            local_sft_replay,
            config=config,
            expected_action_size=runtime_adapter.spec.action_size,
        )

    with metrics_path.open("w", encoding="utf-8") as metrics_stream:
        for restored_metrics in update_metrics:
            _write_metric(metrics_stream, restored_metrics)
        metrics_stream.flush()

        for update_index in range(start_update, config.updates):
            episodes = _collect_episodes(
                model,
                config,
                count=config.episodes_per_update,
                seed=config.seed + 10_000 * update_index,
                probe_probability=config.probe_probability,
                deterministic_student=False,
                generator=generator,
            )
            batch = build_batch(episodes, model, config.ppo_config())
            ppo_metrics = ppo_update(model, optimizer, batch, config.ppo_config())

            replay_before_corrective = len(local_sft_replay.corrective)
            replay_before_fallback = len(local_sft_replay.fallback)
            replay_before_corrective_segments = _segment_count(
                local_sft_replay.corrective
            )
            replay_before_fallback_segments = _segment_count(
                local_sft_replay.fallback
            )
            dataset = LocalSFTDataset()
            fresh_local_sft = LocalSFTDataset()
            before_local_sft = None
            after_local_sft = None
            if config.local_sft_epochs > 0:
                fresh_local_sft = extract_local_sft_examples(
                    episodes,
                    collection_id=update_index,
                )
                local_sft_replay = append_local_sft_replay(
                    local_sft_replay,
                    fresh_local_sft,
                    capacity_per_kind=config.local_sft_replay_capacity_per_kind,
                )
                dataset = local_sft_replay
                evaluation_seed = config.seed + 3_000_000 + update_index
                before_local_sft = evaluate(
                    model,
                    config,
                    probe_probability=0.0,
                    seed=evaluation_seed,
                    generator=generator,
                )
                _validate_student_only_metrics(before_local_sft)

            local_sft_metrics = local_sft_update(
                model,
                dataset,
                config.local_sft_config(),
            )
            cleared_actor_states = 0
            if local_sft_metrics["local_sft_optimizer_steps"] > 0.0:
                cleared_actor_states = _clear_optimizer_state(
                    optimizer,
                    model.actor_head.parameters(),
                )

            if before_local_sft is not None:
                after_local_sft = evaluate(
                    model,
                    config,
                    probe_probability=0.0,
                    seed=evaluation_seed,
                    generator=generator,
                )
                _validate_student_only_metrics(after_local_sft)
                local_sft_evaluations.append(
                    {
                        "update": update_index + 1,
                        "student_only_before": before_local_sft,
                        "student_only_after": after_local_sft,
                    }
                )

            # OPD uses a separate, freshly sampled Student-only state
            # distribution after every preceding model update. The ephemeral
            # dataset is consumed immediately and is never replayed or saved.
            opd_dataset = OPDDataset(collection_id=update_index)
            opd_ledger = OPDAnnotationLedger()
            opd_rollout_metrics = {
                "episodes": 0.0,
                "actor_rows": 0.0,
                "success_rate": 0.0,
                "mean_task_return": 0.0,
                "mean_steps": 0.0,
                "teacher_probe_count": 0.0,
                "teacher_query_count": 0.0,
                "teacher_executed_steps": 0.0,
                "teacher_query_cost": 0.0,
                "teacher_execution_cost": 0.0,
            }
            before_opd = None
            after_opd = None
            if config.opd_epochs > 0:
                opd_episodes = _collect_episodes(
                    model,
                    config,
                    count=config.opd_episodes_per_update,
                    seed=config.seed + 4_000_000 + update_index,
                    probe_probability=0.0,
                    deterministic_student=False,
                    generator=generator,
                )
                opd_rollout_metrics = _aggregate_episodes(opd_episodes)
                _validate_student_only_metrics(opd_rollout_metrics)
                extraction = extract_student_only_opd(
                    opd_episodes,
                    config.opd_annotator(),
                    expected_action_size=runtime_adapter.spec.action_size,
                    collection_id=update_index,
                )
                opd_dataset = extraction.dataset
                opd_ledger = extraction.ledger
                if opd_ledger.query_count != int(
                    opd_rollout_metrics["actor_rows"]
                ):
                    raise AssertionError(
                        "each fresh OPD state must consume exactly one annotation query"
                    )
                del extraction, opd_episodes
                opd_evaluation_seed = config.seed + 5_000_000 + update_index
                before_opd = evaluate(
                    model,
                    config,
                    probe_probability=0.0,
                    seed=opd_evaluation_seed,
                    generator=generator,
                )
                _validate_student_only_metrics(before_opd)

            opd_metrics = opd_update(
                model,
                opd_dataset,
                config.opd_config(),
                expected_collection_id=update_index,
            )
            del opd_dataset
            if opd_metrics["opd_optimizer_steps"] > 0.0:
                cleared_actor_states += _clear_optimizer_state(
                    optimizer,
                    model.actor_head.parameters(),
                )
            if before_opd is not None:
                after_opd = evaluate(
                    model,
                    config,
                    probe_probability=0.0,
                    seed=opd_evaluation_seed,
                    generator=generator,
                )
                _validate_student_only_metrics(after_opd)

            replay_corrective = len(local_sft_replay.corrective)
            replay_fallback = len(local_sft_replay.fallback)
            replay_corrective_segments = _segment_count(
                local_sft_replay.corrective
            )
            replay_fallback_segments = _segment_count(local_sft_replay.fallback)
            fresh_corrective_segments = _segment_count(fresh_local_sft.corrective)
            fresh_fallback_segments = _segment_count(fresh_local_sft.fallback)
            metrics: dict[str, Any] = {
                "metric_schema_version": float(_METRIC_SCHEMA_VERSION),
                "update": float(update_index + 1),
                **_aggregate_episodes(episodes),
                **ppo_metrics,
                **local_sft_metrics,
                **opd_metrics,
                "opd_enabled": float(config.opd_epochs > 0),
                "opd_collection_id": (
                    float(update_index) if config.opd_epochs > 0 else -1.0
                ),
                "opd_annotation_query_count": float(opd_ledger.query_count),
                "opd_annotation_scored_actions": float(opd_ledger.scored_actions),
                "opd_annotation_query_cost": opd_ledger.query_cost,
                "total_teacher_resource_cost": (
                    sum(
                        episode.teacher_costs.query_cost for episode in episodes
                    )
                    + sum(
                        episode.teacher_costs.execution_cost for episode in episodes
                    )
                    + opd_ledger.query_cost
                ),
                "opd_rollout_episodes": opd_rollout_metrics["episodes"],
                "opd_rollout_actor_rows": opd_rollout_metrics["actor_rows"],
                "opd_rollout_success_rate": opd_rollout_metrics["success_rate"],
                "opd_rollout_mean_task_return": opd_rollout_metrics[
                    "mean_task_return"
                ],
                "opd_rollout_mean_steps": opd_rollout_metrics["mean_steps"],
                "opd_rollout_teacher_probe_count": opd_rollout_metrics[
                    "teacher_probe_count"
                ],
                "opd_rollout_teacher_query_count": opd_rollout_metrics[
                    "teacher_query_count"
                ],
                "opd_rollout_teacher_executed_steps": opd_rollout_metrics[
                    "teacher_executed_steps"
                ],
                "opd_rollout_teacher_cost": (
                    opd_rollout_metrics["teacher_query_cost"]
                    + opd_rollout_metrics["teacher_execution_cost"]
                ),
                "new_corrective_sft_examples": float(len(fresh_local_sft.corrective)),
                "new_fallback_sft_examples": float(len(fresh_local_sft.fallback)),
                "replay_corrective_sft_examples": float(replay_corrective),
                "replay_fallback_sft_examples": float(replay_fallback),
                "replay_corrective_segments": float(replay_corrective_segments),
                "replay_fallback_segments": float(replay_fallback_segments),
                "replay_corrective_evicted_examples": float(
                    max(
                        0,
                        replay_before_corrective
                        + len(fresh_local_sft.corrective)
                        - replay_corrective,
                    )
                ),
                "replay_fallback_evicted_examples": float(
                    max(
                        0,
                        replay_before_fallback
                        + len(fresh_local_sft.fallback)
                        - replay_fallback,
                    )
                ),
                "replay_corrective_evicted_segments": float(
                    max(
                        0,
                        replay_before_corrective_segments
                        + fresh_corrective_segments
                        - replay_corrective_segments,
                    )
                ),
                "replay_fallback_evicted_segments": float(
                    max(
                        0,
                        replay_before_fallback_segments
                        + fresh_fallback_segments
                        - replay_fallback_segments,
                    )
                ),
                "replay_corrective_soft_cap_ratio": (
                    replay_corrective / config.local_sft_replay_capacity_per_kind
                ),
                "replay_fallback_soft_cap_ratio": (
                    replay_fallback / config.local_sft_replay_capacity_per_kind
                ),
                "ppo_actor_optimizer_states_cleared": float(cleared_actor_states),
            }
            if before_local_sft is not None and after_local_sft is not None:
                metrics.update(
                    {
                        "student_only_success_before_local_sft": before_local_sft[
                            "success_rate"
                        ],
                        "student_only_success_after_local_sft": after_local_sft[
                            "success_rate"
                        ],
                        "student_only_return_before_local_sft": before_local_sft[
                            "mean_task_return"
                        ],
                        "student_only_return_after_local_sft": after_local_sft[
                            "mean_task_return"
                        ],
                    }
                )
            if before_opd is not None and after_opd is not None:
                metrics.update(
                    {
                        "student_only_success_before_opd": before_opd[
                            "success_rate"
                        ],
                        "student_only_success_after_opd": after_opd[
                            "success_rate"
                        ],
                        "student_only_return_before_opd": before_opd[
                            "mean_task_return"
                        ],
                        "student_only_return_after_opd": after_opd[
                            "mean_task_return"
                        ],
                    }
                )
            update_metrics.append(metrics)
            _write_metric(metrics_stream, metrics)
            metrics_stream.flush()
            save_training_checkpoint(
                checkpoint_path,
                model=model,
                ppo_optimizer=optimizer,
                completed_updates=update_index + 1,
                config=config,
                metrics=update_metrics,
                local_sft_evaluations=local_sft_evaluations,
                local_sft_replay=local_sft_replay,
                generator=generator,
            )

    # Materialize a checkpoint in a new destination only when resume had
    # already completed every requested update. Fresh work checkpoints inside
    # the loop, avoiding a duplicate final write.
    if start_update == config.updates:
        save_training_checkpoint(
            checkpoint_path,
            model=model,
            ppo_optimizer=optimizer,
            completed_updates=config.updates,
            config=config,
            metrics=update_metrics,
            local_sft_evaluations=local_sft_evaluations,
            local_sft_replay=local_sft_replay,
            generator=generator,
        )

    hybrid_eval = evaluate(
        model,
        config,
        probe_probability=config.probe_probability,
        seed=config.seed + 1_000_000,
        generator=generator,
    )
    student_only_eval = evaluate(
        model,
        config,
        probe_probability=0.0,
        seed=config.seed + 2_000_000,
        generator=generator,
    )
    _validate_student_only_metrics(student_only_eval)
    summary: dict[str, Any] = {
        "updates": update_metrics,
        "local_sft_evaluations": local_sft_evaluations,
        "hybrid_eval": hybrid_eval,
        "student_only_eval": student_only_eval,
        "checkpoint": str(checkpoint_path),
        "metrics": str(metrics_path),
        "resumed_from": str(resume_from) if resume_from is not None else None,
        "start_update": start_update,
    }
    summary_path = destination / "summary.json"
    with summary_path.open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="path to a toy training JSON config")
    parser.add_argument("--output-dir", help="override the generated artifact directory")
    parser.add_argument("--resume", help="resume from a training checkpoint")
    arguments = parser.parse_args()
    config = ToyTrainConfig.from_json(arguments.config)
    summary = run_training(
        config,
        output_dir=arguments.output_dir,
        resume_from=arguments.resume,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
