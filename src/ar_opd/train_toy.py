"""Command-line entry point for the first end-to-end AR-OPD smoke loop."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean
from typing import Any

import torch

from ar_opd.core import EpisodeRollout
from ar_opd.models import ActorCritic
from ar_opd.ppo import PPOConfig, build_batch, ppo_update
from ar_opd.rollout import RolloutCollector, RolloutConfig
from ar_opd.teacher import OracleTeacher
from ar_opd.toy_env import JammedChainConfig, JammedChainEnv


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
    output_dir: str = "outputs/toy_smoke"

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
        "teacher_query_cost": sum(episode.teacher_costs.query_cost for episode in episodes),
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
    episodes: list[EpisodeRollout] = []
    for episode_index in range(count):
        env = JammedChainEnv(config.environment_config())
        teacher = OracleTeacher(env.config)
        collector = RolloutCollector(
            config.rollout_config(probe_probability),
            seed=seed + episode_index,
            torch_generator=generator,
        )
        episodes.append(
            collector.collect_episode(
                env,
                model,
                teacher,
                deterministic_student=deterministic_student,
            )
        )
    return episodes


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


def run_training(
    config: ToyTrainConfig,
    *,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    model = ActorCritic(
        JammedChainEnv.observation_size,
        JammedChainEnv.action_size,
        config.hidden_size,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    generator = torch.Generator(device=device.type).manual_seed(config.seed)
    destination = Path(output_dir or config.output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    metrics_path = destination / "metrics.jsonl"
    update_metrics: list[dict[str, float]] = []

    with metrics_path.open("w", encoding="utf-8") as metrics_stream:
        for update_index in range(config.updates):
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
            losses = ppo_update(model, optimizer, batch, config.ppo_config())
            metrics = {
                "update": float(update_index + 1),
                **_aggregate_episodes(episodes),
                **losses,
            }
            update_metrics.append(metrics)
            metrics_stream.write(json.dumps(metrics, sort_keys=True) + "\n")
            metrics_stream.flush()

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
    checkpoint_path = destination / "checkpoint.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": asdict(config),
            "completed_updates": config.updates,
        },
        checkpoint_path,
    )
    summary: dict[str, Any] = {
        "updates": update_metrics,
        "hybrid_eval": hybrid_eval,
        "student_only_eval": student_only_eval,
        "checkpoint": str(checkpoint_path),
        "metrics": str(metrics_path),
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
    arguments = parser.parse_args()
    config = ToyTrainConfig.from_json(arguments.config)
    summary = run_training(config, output_dir=arguments.output_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
