"""Executed-only local SFT extraction and actor-head updates."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Sequence

import torch
from torch import nn

from ar_opd.core import ActionSource, EpisodeRollout, OptionKind
from ar_opd.models import ActorCritic


class LocalSFTKind(str, Enum):
    CORRECTIVE = "corrective"
    FALLBACK = "fallback"


@dataclass(frozen=True)
class LocalSFTExample:
    observation: tuple[float, ...]
    target_action: int
    kind: LocalSFTKind
    episode_index: int
    decision_id: int
    primitive_offset: int
    segment_length: int
    weight: float
    collection_id: int = 0

    def __post_init__(self) -> None:
        if (
            self.collection_id < 0
            or self.episode_index < 0
            or self.decision_id < 0
            or self.primitive_offset < 0
        ):
            raise ValueError("local SFT provenance indices must be non-negative")
        if self.segment_length < 1 or self.primitive_offset >= self.segment_length:
            raise ValueError("primitive offset must lie inside its teacher segment")
        if self.weight <= 0.0 or not math.isfinite(self.weight):
            raise ValueError("local SFT example weight must be finite and positive")


@dataclass(frozen=True)
class LocalSFTDataset:
    corrective: tuple[LocalSFTExample, ...] = ()
    fallback: tuple[LocalSFTExample, ...] = ()

    @property
    def total_examples(self) -> int:
        return len(self.corrective) + len(self.fallback)


@dataclass(frozen=True)
class LocalSFTConfig:
    epochs: int = 0
    learning_rate: float = 0.01
    corrective_coefficient: float = 1.0
    fallback_coefficient: float = 1.0
    max_grad_norm: float = 1.0

    def __post_init__(self) -> None:
        if self.epochs < 0:
            raise ValueError("local SFT epochs must be non-negative")
        if self.learning_rate <= 0.0:
            raise ValueError("local SFT learning rate must be positive")
        if self.corrective_coefficient < 0.0 or self.fallback_coefficient < 0.0:
            raise ValueError("local SFT coefficients must be non-negative")
        if self.max_grad_norm <= 0.0:
            raise ValueError("local SFT max_grad_norm must be positive")


def _finite_vector(values: tuple[float, ...]) -> bool:
    if not values:
        return False
    try:
        return all(math.isfinite(value) for value in values)
    except TypeError:
        return False


def validate_episode_for_distillation(episode: EpisodeRollout) -> None:
    """Reject slice corruption before any Teacher action becomes a target."""

    if not episode.decisions or not episode.transitions:
        raise ValueError("distillation requires a non-empty completed episode")
    if any(row.terminated or row.truncated for row in episode.transitions[:-1]):
        raise ValueError("episode cannot continue after a terminal primitive step")
    final_transition = episode.transitions[-1]
    if final_transition.terminated == final_transition.truncated:
        raise ValueError("completed episode must end in exactly one terminal condition")
    for row in episode.transitions:
        if not _finite_vector(row.observation) or not _finite_vector(
            row.next_observation
        ):
            raise ValueError("transition observations must be finite and non-empty")
        if not isinstance(row.action, int):
            raise ValueError("transition actions must be integers")
        if not all(
            math.isfinite(value)
            for value in (row.env_reward, row.query_cost, row.execution_cost)
        ):
            raise ValueError("transition rewards and costs must be finite")
        if row.source is ActionSource.STUDENT and row.execution_cost != 0.0:
            raise ValueError("Student transitions cannot carry Teacher execution cost")

    ledger_counts = (
        episode.teacher_costs.probe_count,
        episode.teacher_costs.query_count,
        episode.teacher_costs.generated_teacher_steps,
        episode.teacher_costs.executed_teacher_steps,
    )
    if any(type(value) is not int or value < 0 for value in ledger_counts):
        raise ValueError("Teacher ledger counts must be non-negative integers")
    ledger_costs = (
        episode.teacher_costs.query_cost,
        episode.teacher_costs.execution_cost,
    )
    if any(not math.isfinite(value) or value < 0.0 for value in ledger_costs):
        raise ValueError("Teacher ledger costs must be finite and non-negative")
    expected_start = 0
    for decision_index, decision in enumerate(episode.decisions):
        if decision.decision_id != decision_index:
            raise ValueError("decision ids must be contiguous within an episode")
        if decision.transition_start != expected_start:
            raise ValueError("decision transition ranges must be contiguous and ordered")
        if not 0 <= decision.transition_start < decision.transition_stop <= len(
            episode.transitions
        ):
            raise ValueError("decision transition range is out of bounds")
        rows = episode.transitions[decision.transition_start : decision.transition_stop]
        if any(row.decision_id != decision.decision_id for row in rows):
            raise ValueError("transition decision id does not match its slice")
        if rows[0].observation != decision.observation:
            raise ValueError("decision and first primitive observation are misaligned")
        if any(
            previous.next_observation != current.observation
            for previous, current in zip(rows, rows[1:])
        ):
            raise ValueError("primitive observations are not contiguous")
        proposal = decision.student_proposal
        if not isinstance(proposal.action, int) or not all(
            math.isfinite(value) for value in (proposal.log_prob, proposal.value)
        ):
            raise ValueError("Student proposal metadata must be finite and integral")
        for candidate in decision.candidates:
            if not all(isinstance(action, int) for action in candidate.actions):
                raise ValueError("candidate actions must be integers")
            if not all(
                math.isfinite(value)
                for value in (
                    candidate.estimated_task_value,
                    candidate.query_cost,
                    candidate.execution_cost,
                )
            ):
                raise ValueError("candidate scores and costs must be finite")
        candidate_kinds = {candidate.kind for candidate in decision.candidates}
        expected_kinds = (
            {
                OptionKind.STUDENT,
                OptionKind.TEACHER_CORRECTION,
                OptionKind.TEACHER_RECOVERY,
            }
            if decision.probed
            else {OptionKind.STUDENT}
        )
        if (
            candidate_kinds != expected_kinds
            or len(decision.candidates) != len(expected_kinds)
        ):
            raise ValueError("candidate set does not match the probe decision")
        student_candidate = next(
            candidate
            for candidate in decision.candidates
            if candidate.kind is OptionKind.STUDENT
        )
        if (
            student_candidate.actions != (proposal.action,)
            or student_candidate.preview_steps != 1
        ):
            raise ValueError("Student candidate must match the sampled proposal")
        if any(
            not math.isclose(
                candidate.query_cost, rows[0].query_cost, abs_tol=1e-9
            )
            for candidate in decision.candidates
        ):
            raise ValueError("candidate query costs disagree with actual execution")
        if any(row.query_cost != 0.0 for row in rows[1:]):
            raise ValueError("query cost may only be attached to the first primitive step")
        if not decision.probed and rows[0].query_cost != 0.0:
            raise ValueError("an unprobed decision cannot carry query cost")
        if any(row.selected_option is not decision.selected_option for row in rows):
            raise ValueError("transition option does not match its decision")

        expected_source = (
            ActionSource.STUDENT
            if decision.selected_option is OptionKind.STUDENT
            else ActionSource.TEACHER
        )
        if any(row.source is not expected_source for row in rows):
            raise ValueError("transition source does not match the selected option")
        if decision.selected_option in (
            OptionKind.STUDENT,
            OptionKind.TEACHER_CORRECTION,
        ) and len(rows) != 1:
            raise ValueError("S and T decisions must execute exactly one primitive step")

        selected_candidates = [
            candidate
            for candidate in decision.candidates
            if candidate.kind is decision.selected_option
        ]
        if len(selected_candidates) != 1:
            raise ValueError("each decision must retain exactly one selected candidate")
        selected_candidate = selected_candidates[0]
        actual_actions = tuple(row.action for row in rows)
        if actual_actions != selected_candidate.previewed_actions:
            raise ValueError("actual actions do not match the previewed candidate prefix")
        if selected_candidate.preview_steps != len(rows):
            raise ValueError("candidate preview length does not match actual execution")
        if (
            selected_candidate.terminated != rows[-1].terminated
            or selected_candidate.truncated != rows[-1].truncated
        ):
            raise ValueError("candidate terminal flags do not match actual execution")

        expected_start = decision.transition_stop
        if decision_index + 1 < len(episode.decisions):
            next_decision = episode.decisions[decision_index + 1]
            if rows[-1].next_observation != next_decision.observation:
                raise ValueError("adjacent decision-boundary observations are misaligned")

    if expected_start != len(episode.transitions):
        raise ValueError("decision ranges must cover every actual transition exactly once")
    probed_decisions = sum(decision.probed for decision in episode.decisions)
    expected_generated_steps = sum(
        len(candidate.actions)
        for decision in episode.decisions
        if decision.probed
        for candidate in decision.candidates
        if candidate.kind is not OptionKind.STUDENT
    )
    if episode.teacher_costs.probe_count != probed_decisions:
        raise ValueError("probe ledger disagrees with decisions")
    if episode.teacher_costs.query_count != probed_decisions:
        raise ValueError("query ledger disagrees with decisions")
    if episode.teacher_costs.generated_teacher_steps != expected_generated_steps:
        raise ValueError("generated-step ledger disagrees with candidates")

    query_cost = sum(row.query_cost for row in episode.transitions)
    execution_cost = sum(row.execution_cost for row in episode.transitions)
    executed_teacher_steps = sum(
        row.source is ActionSource.TEACHER for row in episode.transitions
    )
    if not math.isclose(query_cost, episode.teacher_costs.query_cost, abs_tol=1e-9):
        raise ValueError("query-cost ledger disagrees with actual transitions")
    if not math.isclose(execution_cost, episode.teacher_costs.execution_cost, abs_tol=1e-9):
        raise ValueError("execution-cost ledger disagrees with actual transitions")
    if executed_teacher_steps != episode.teacher_costs.executed_teacher_steps:
        raise ValueError("teacher-step ledger disagrees with actual transitions")


def extract_local_sft_examples(
    episodes: Sequence[EpisodeRollout],
    *,
    collection_id: int = 0,
) -> LocalSFTDataset:
    if collection_id < 0:
        raise ValueError("collection_id must be non-negative")
    corrective: list[LocalSFTExample] = []
    fallback: list[LocalSFTExample] = []
    for episode_index, episode in enumerate(episodes):
        validate_episode_for_distillation(episode)
        for decision in episode.decisions:
            rows = episode.transitions[
                decision.transition_start : decision.transition_stop
            ]
            if decision.selected_option is OptionKind.TEACHER_CORRECTION:
                row = rows[0]
                corrective.append(
                    LocalSFTExample(
                        observation=row.observation,
                        target_action=row.action,
                        kind=LocalSFTKind.CORRECTIVE,
                        episode_index=episode_index,
                        decision_id=decision.decision_id,
                        primitive_offset=0,
                        segment_length=1,
                        weight=1.0,
                        collection_id=collection_id,
                    )
                )
            elif decision.selected_option is OptionKind.TEACHER_RECOVERY:
                segment_weight = 1.0 / len(rows)
                for primitive_offset, row in enumerate(rows):
                    fallback.append(
                        LocalSFTExample(
                            observation=row.observation,
                            target_action=row.action,
                            kind=LocalSFTKind.FALLBACK,
                            episode_index=episode_index,
                            decision_id=decision.decision_id,
                            primitive_offset=primitive_offset,
                            segment_length=len(rows),
                            weight=segment_weight,
                            collection_id=collection_id,
                        )
                    )
    dataset = LocalSFTDataset(tuple(corrective), tuple(fallback))
    expected_teacher_steps = sum(
        episode.teacher_costs.executed_teacher_steps for episode in episodes
    )
    if dataset.total_examples != expected_teacher_steps:
        raise ValueError("each executed Teacher step must produce exactly one local SFT example")
    return dataset


def _weighted_action_loss(
    model: ActorCritic,
    examples: tuple[LocalSFTExample, ...],
) -> torch.Tensor:
    observations = torch.tensor(
        [example.observation for example in examples],
        dtype=torch.float32,
        device=model.device,
    )
    actions = torch.tensor(
        [example.target_action for example in examples],
        dtype=torch.long,
        device=model.device,
    )
    weights = torch.tensor(
        [example.weight for example in examples],
        dtype=torch.float32,
        device=model.device,
    )
    with torch.no_grad():
        features = model.encoder(observations)
    logits = model.actor_head(features.detach())
    per_example = nn.functional.cross_entropy(logits, actions, reduction="none")
    return (per_example * weights).sum() / weights.sum()


def _diagnostic_loss(
    model: ActorCritic,
    examples: tuple[LocalSFTExample, ...],
) -> float:
    if not examples:
        return 0.0
    with torch.no_grad():
        return float(_weighted_action_loss(model, examples))


def local_sft_update(
    model: ActorCritic,
    dataset: LocalSFTDataset,
    config: LocalSFTConfig,
) -> dict[str, float]:
    metrics = {
        "local_sft_loss": 0.0,
        "corrective_sft_loss": 0.0,
        "fallback_sft_loss": 0.0,
        "trained_corrective_sft_examples": float(len(dataset.corrective)),
        "trained_fallback_sft_examples": float(len(dataset.fallback)),
        "local_sft_optimizer_steps": 0.0,
        "corrective_nll_before": _diagnostic_loss(model, dataset.corrective),
        "corrective_nll_after": 0.0,
        "fallback_nll_before": _diagnostic_loss(model, dataset.fallback),
        "fallback_nll_after": 0.0,
    }
    active_corrective = bool(dataset.corrective) and config.corrective_coefficient > 0.0
    active_fallback = bool(dataset.fallback) and config.fallback_coefficient > 0.0
    if config.epochs == 0 or not (active_corrective or active_fallback):
        metrics["corrective_nll_after"] = metrics["corrective_nll_before"]
        metrics["fallback_nll_after"] = metrics["fallback_nll_before"]
        return metrics

    # This update deliberately uses stateless SGD. PPO owns a long-lived Adam
    # optimizer, whose actor-head moments are cleared by the training loop after
    # an SFT update. Giving the same parameters to a second Adam would leave two
    # incompatible moment histories and make checkpoint/resume ambiguous.
    optimizer = torch.optim.SGD(model.actor_head.parameters(), lr=config.learning_rate)
    for _ in range(config.epochs):
        zero = torch.zeros((), dtype=torch.float32, device=model.device)
        corrective_loss = (
            _weighted_action_loss(model, dataset.corrective) if active_corrective else zero
        )
        fallback_loss = (
            _weighted_action_loss(model, dataset.fallback) if active_fallback else zero
        )
        total = (
            config.corrective_coefficient * corrective_loss
            + config.fallback_coefficient * fallback_loss
        )
        if not bool(torch.isfinite(total)):
            raise FloatingPointError("non-finite local SFT loss")
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        nn.utils.clip_grad_norm_(
            model.actor_head.parameters(),
            config.max_grad_norm,
            error_if_nonfinite=True,
        )
        optimizer.step()
        metrics["local_sft_loss"] += float(total.detach())
        metrics["corrective_sft_loss"] += float(corrective_loss.detach())
        metrics["fallback_sft_loss"] += float(fallback_loss.detach())
        metrics["local_sft_optimizer_steps"] += 1.0

    metrics["local_sft_loss"] /= config.epochs
    metrics["corrective_sft_loss"] /= config.epochs
    metrics["fallback_sft_loss"] /= config.epochs
    metrics["corrective_nll_after"] = _diagnostic_loss(model, dataset.corrective)
    metrics["fallback_nll_after"] = _diagnostic_loss(model, dataset.fallback)
    return metrics
