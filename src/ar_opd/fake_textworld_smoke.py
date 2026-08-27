"""Run the dependency-free fake TextWorld rollout and one PPO update.

Usage::

    PYTHONPATH=src python -m ar_opd.fake_textworld_smoke
"""

from __future__ import annotations

import json
import math

import torch

from ar_opd.core import ActionSource, OptionKind
from ar_opd.fake_textworld import (
    FAKE_TEXTWORLD_BACKEND_IDENTITY,
    FakeTextWorldBackendFactory,
    JAMMED_QUEST_ACTION_VOCABULARY,
    JAMMED_QUEST_GAME,
    JammedQuestTeacherFactory,
)
from ar_opd.models import ActorCritic
from ar_opd.ppo import PPOConfig, build_batch, ppo_update
from ar_opd.rollout import RolloutConfig, collect_episodes
from ar_opd.textworld_runtime import TextWorldRuntimeAdapter, TextWorldRuntimeConfig


_SEED = 20260826
_OBSERVATION_SIZE = 16
_HIDDEN_SIZE = 8
_MAX_STEPS = 8


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _make_deterministic_student(action_size: int) -> ActorCritic:
    """Create a fixed Student that always proposes the global ``go east`` ID."""

    torch.manual_seed(_SEED)
    model = ActorCritic(
        observation_size=_OBSERVATION_SIZE,
        action_size=action_size,
        hidden_size=_HIDDEN_SIZE,
    )
    go_east = JAMMED_QUEST_ACTION_VOCABULARY.index("go east")
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.actor_head.bias[go_east] = 4.0
    return model


def _finite_ppo_summary(metrics: dict[str, float]) -> dict[str, float | int]:
    summary: dict[str, float | int] = {}
    for name, value in metrics.items():
        _require(math.isfinite(value), f"PPO metric {name!r} is not finite")
        if name in {"actor_rows", "critic_rows"}:
            summary[name] = int(value)
            continue
        rounded = round(value, 8)
        summary[name] = 0.0 if rounded == 0.0 else rounded
    return summary


def run_smoke() -> dict[str, object]:
    """Run and assert the deterministic Hybrid rollout/PPO smoke path."""

    backend_factory = FakeTextWorldBackendFactory()
    teacher_factory = JammedQuestTeacherFactory()
    runtime_config = TextWorldRuntimeConfig(
        game_name=JAMMED_QUEST_GAME.name,
        backend_identity=FAKE_TEXTWORLD_BACKEND_IDENTITY,
        action_vocabulary=JAMMED_QUEST_ACTION_VOCABULARY,
        observation_size=_OBSERVATION_SIZE,
        project_max_steps=_MAX_STEPS,
    )
    adapter = TextWorldRuntimeAdapter(
        runtime_config,
        backend_factory=backend_factory,
        teacher_factory=teacher_factory,
    )
    student = _make_deterministic_student(adapter.spec.action_size)
    rollout_config = RolloutConfig(
        gamma=0.97,
        probe_probability=1.0,
        recovery_horizon=2,
        teacher_query_cost=0.01,
        teacher_execution_cost=0.02,
    )
    episode = collect_episodes(
        adapter,
        student,
        rollout_config,
        count=1,
        seed=_SEED,
        deterministic_student=True,
        generator=None,
    )[0]

    expected_decisions = [
        OptionKind.STUDENT,
        OptionKind.TEACHER_RECOVERY,
    ]
    expected_actions = [
        JAMMED_QUEST_ACTION_VOCABULARY.index("go east"),
        JAMMED_QUEST_ACTION_VOCABULARY.index("repair cart"),
        JAMMED_QUEST_ACTION_VOCABULARY.index("go east"),
    ]
    expected_sources = [
        ActionSource.STUDENT,
        ActionSource.TEACHER,
        ActionSource.TEACHER,
    ]
    _require(episode.success, "Hybrid fake TextWorld rollout did not succeed")
    _require(
        [decision.selected_option for decision in episode.decisions]
        == expected_decisions,
        "Hybrid fake TextWorld rollout did not select S then F",
    )
    _require(
        [transition.action for transition in episode.transitions]
        == expected_actions,
        "Hybrid fake TextWorld rollout executed an unexpected action trace",
    )
    _require(
        [transition.source for transition in episode.transitions]
        == expected_sources,
        "Hybrid fake TextWorld rollout has incorrect action ownership",
    )

    _require(backend_factory.call_count == 2, "runtime must create two backends")
    online_backend, scratch_backend = backend_factory.instances
    _require(
        len(online_backend.reset_calls) == 1,
        "online backend must reset exactly once",
    )
    _require(
        len(scratch_backend.reset_calls) == 6,
        "scratch backend must reset once for each S/T/F preview",
    )
    _require(
        online_backend.close_count == 1 and scratch_backend.close_count == 1,
        "online and scratch backends must each close exactly once",
    )
    _require(teacher_factory.call_count == 1, "runtime must create one Teacher")
    _require(
        teacher_factory.instances[0].call_count == 2,
        "Teacher must be queried at both decision boundaries",
    )

    ppo_config = PPOConfig(gamma=rollout_config.gamma, epochs=1)
    batch = build_batch([episode], student, ppo_config)
    proposed_actions = [
        decision.student_proposal.action for decision in episode.decisions
    ]
    _require(
        batch.actions.detach().cpu().tolist() == proposed_actions,
        "PPO batch actions must be the sampled Student proposals",
    )
    _require(
        batch.actions.detach().cpu().tolist() == [expected_actions[0]] * 2,
        "PPO batch accidentally used the executed repair action",
    )
    _require(
        batch.actor_mask.detach().cpu().tolist() == [True, True],
        "both gate decisions must remain PPO actor rows",
    )
    optimizer = torch.optim.Adam(student.parameters(), lr=1e-3)
    ppo_metrics = _finite_ppo_summary(
        ppo_update(student, optimizer, batch, ppo_config)
    )
    _require(
        all(bool(torch.isfinite(parameter).all()) for parameter in student.parameters()),
        "PPO update produced non-finite model parameters",
    )

    return {
        "batch": {
            "actions": batch.actions.detach().cpu().tolist(),
            "actor_rows": int(batch.actor_mask.sum().item()),
            "critic_rows": int(batch.actor_mask.numel()),
        },
        "episode": {
            "decisions": [kind.value for kind in expected_decisions],
            "net_return": round(episode.net_return, 8),
            "success": episode.success,
            "task_return": round(episode.task_return, 8),
            "transition_actions": expected_actions,
            "transition_sources": [source.value for source in expected_sources],
        },
        "lifecycle": {
            "backend_close_counts": [
                online_backend.close_count,
                scratch_backend.close_count,
            ],
            "backend_instances": backend_factory.call_count,
            "online_resets": len(online_backend.reset_calls),
            "scratch_resets": len(scratch_backend.reset_calls),
            "teacher_instances": teacher_factory.call_count,
            "teacher_queries": teacher_factory.instances[0].call_count,
        },
        "ppo": ppo_metrics,
        "seed": _SEED,
    }


def main() -> None:
    summary = run_smoke()
    print(json.dumps(summary, allow_nan=False, separators=(",", ":"), sort_keys=True))


if __name__ == "__main__":
    main()
