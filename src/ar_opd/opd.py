"""Ephemeral Student-only OPD for discrete environment actions.

An OPD dataset belongs to exactly one fresh no-probe collection. Consume it
immediately for the configured local epochs, then discard it. It is deliberately
not a replay-buffer or checkpoint payload: reusing stale annotated states would
silently turn this on-policy objective into offline distillation.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import torch
from torch import nn

from ar_opd.core import ActionSource, EpisodeRollout, OptionKind
from ar_opd.distillation import validate_episode_for_distillation
from ar_opd.models import ActorCritic
from ar_opd.toy_env import ChainAction


@dataclass(frozen=True)
class ActionDistributionAnnotation:
    probabilities: tuple[float, ...]
    query_cost: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "probabilities",
            validate_probability_distribution(self.probabilities),
        )
        if isinstance(self.query_cost, bool) or not isinstance(
            self.query_cost, int | float
        ):
            raise TypeError("annotation query cost must be a real number")
        if self.query_cost < 0.0 or not math.isfinite(self.query_cost):
            raise ValueError("annotation query cost must be finite and non-negative")


class FullActionDistributionAnnotator(Protocol):
    action_size: int

    def annotate(
        self, observation: tuple[float, ...]
    ) -> ActionDistributionAnnotation: ...


@dataclass(frozen=True)
class OPDAnnotationLedger:
    """Teacher annotation resources, intentionally separate from PPO rewards."""

    query_count: int = 0
    scored_actions: int = 0
    query_cost: float = 0.0

    def __post_init__(self) -> None:
        if (
            isinstance(self.query_count, bool)
            or not isinstance(self.query_count, int)
            or isinstance(self.scored_actions, bool)
            or not isinstance(self.scored_actions, int)
        ):
            raise TypeError("annotation counts must be integers")
        if self.query_count < 0 or self.scored_actions < 0:
            raise ValueError("annotation counts must be non-negative")
        if isinstance(self.query_cost, bool) or not isinstance(
            self.query_cost, int | float
        ):
            raise TypeError("annotation query cost must be a real number")
        if self.query_cost < 0.0 or not math.isfinite(self.query_cost):
            raise ValueError("annotation query cost must be finite and non-negative")


class OPDAnnotationError(RuntimeError):
    """Annotation failure carrying every successfully accounted query so far."""

    def __init__(self, message: str, partial_ledger: OPDAnnotationLedger) -> None:
        super().__init__(message)
        self.partial_ledger = partial_ledger


@dataclass(frozen=True)
class OPDExample:
    observation: tuple[float, ...]
    teacher_probabilities: tuple[float, ...]
    student_proposal_action: int
    collection_id: int
    episode_index: int
    decision_id: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "observation", tuple(float(x) for x in self.observation))
        if not self.observation or any(
            not math.isfinite(value) for value in self.observation
        ):
            raise ValueError("OPD observation must contain finite values")
        object.__setattr__(
            self,
            "teacher_probabilities",
            validate_probability_distribution(self.teacher_probabilities),
        )
        if isinstance(self.student_proposal_action, bool) or not isinstance(
            self.student_proposal_action, int
        ):
            raise TypeError("Student proposal action must be an integer")
        if not 0 <= self.student_proposal_action < len(self.teacher_probabilities):
            raise ValueError("Student proposal action is outside the Teacher support")
        indices = (self.collection_id, self.episode_index, self.decision_id)
        if any(isinstance(value, bool) or not isinstance(value, int) for value in indices):
            raise TypeError("OPD provenance indices must be integers")
        if any(value < 0 for value in indices):
            raise ValueError("OPD provenance indices must be non-negative")


@dataclass(frozen=True)
class OPDDataset:
    """One fresh collection; never replay or serialize this dataset."""

    collection_id: int
    examples: tuple[OPDExample, ...] = ()

    def __post_init__(self) -> None:
        if isinstance(self.collection_id, bool) or not isinstance(self.collection_id, int):
            raise TypeError("OPD collection_id must be an integer")
        if self.collection_id < 0:
            raise ValueError("OPD collection_id must be non-negative")
        object.__setattr__(self, "examples", tuple(self.examples))
        if any(not isinstance(row, OPDExample) for row in self.examples):
            raise TypeError("OPD dataset examples must be OPDExample records")
        if any(row.collection_id != self.collection_id for row in self.examples):
            raise ValueError("every OPD example must match the dataset collection_id")
        action_sizes = {len(row.teacher_probabilities) for row in self.examples}
        if len(action_sizes) > 1:
            raise ValueError("all OPD examples must share one action dimension")

    def __len__(self) -> int:
        return len(self.examples)

    @property
    def action_size(self) -> int | None:
        return len(self.examples[0].teacher_probabilities) if self.examples else None


@dataclass(frozen=True)
class OPDExtraction:
    dataset: OPDDataset
    ledger: OPDAnnotationLedger


@dataclass(frozen=True)
class OPDConfig:
    epochs: int = 0
    learning_rate: float = 0.01
    target_temperature: float = 1.0
    max_grad_norm: float = 1.0

    def __post_init__(self) -> None:
        if isinstance(self.epochs, bool) or not isinstance(self.epochs, int):
            raise TypeError("OPD epochs must be an integer")
        if self.epochs < 0:
            raise ValueError("OPD epochs must be non-negative")
        if isinstance(self.learning_rate, bool) or not isinstance(
            self.learning_rate, int | float
        ):
            raise TypeError("OPD learning rate must be a real number")
        if self.learning_rate <= 0.0 or not math.isfinite(self.learning_rate):
            raise ValueError("OPD learning rate must be finite and positive")
        _validate_target_temperature(self.target_temperature)
        if isinstance(self.max_grad_norm, bool) or not isinstance(
            self.max_grad_norm, int | float
        ):
            raise TypeError("OPD max_grad_norm must be a real number")
        if self.max_grad_norm <= 0.0 or not math.isfinite(self.max_grad_norm):
            raise ValueError("OPD max_grad_norm must be finite and positive")


class ToyOracleDistributionAnnotator:
    """A stateless smoothed oracle over the toy environment's two actions."""

    action_size = len(ChainAction)

    def __init__(
        self,
        *,
        preferred_probability: float = 0.95,
        query_cost: float = 0.0,
    ) -> None:
        if isinstance(preferred_probability, bool) or not isinstance(
            preferred_probability, int | float
        ):
            raise TypeError("preferred_probability must be a real number")
        if not 0.5 < preferred_probability < 1.0:
            raise ValueError("preferred_probability must lie strictly between 0.5 and 1")
        if isinstance(query_cost, bool) or not isinstance(query_cost, int | float):
            raise TypeError("annotation query cost must be a real number")
        if query_cost < 0.0 or not math.isfinite(query_cost):
            raise ValueError("annotation query cost must be finite and non-negative")
        self.preferred_probability = preferred_probability
        self.query_cost = query_cost

    def annotate(self, observation: tuple[float, ...]) -> ActionDistributionAnnotation:
        if len(observation) != 3 or any(not math.isfinite(value) for value in observation):
            raise ValueError("toy annotation requires a finite encoded toy observation")
        preferred = ChainAction.REPAIR if observation[1] >= 0.5 else ChainAction.ADVANCE
        other_probability = 1.0 - self.preferred_probability
        probabilities = [other_probability] * self.action_size
        probabilities[int(preferred)] = self.preferred_probability
        return ActionDistributionAnnotation(tuple(probabilities), self.query_cost)


def validate_probability_distribution(
    probabilities: Sequence[float],
    *,
    action_size: int | None = None,
    tolerance: float = 1e-6,
) -> tuple[float, ...]:
    """Validate and return an immutable categorical probability vector."""

    values = tuple(probabilities)
    if action_size is not None and (
        isinstance(action_size, bool) or not isinstance(action_size, int)
    ):
        raise TypeError("action_size must be an integer")
    if action_size is not None and action_size < 2:
        raise ValueError("action_size must be at least two")
    expected_size = action_size if action_size is not None else len(values)
    if expected_size < 2 or len(values) != expected_size:
        raise ValueError("probability vector has the wrong action dimension")
    if any(isinstance(value, bool) or not isinstance(value, int | float) for value in values):
        raise TypeError("probabilities must be real numbers")
    converted = tuple(float(value) for value in values)
    if any(not math.isfinite(value) or value < 0.0 for value in converted):
        raise ValueError("probabilities must be finite and non-negative")
    total = sum(converted)
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=tolerance):
        raise ValueError("probabilities must sum to one")
    if not any(value > 0.0 for value in converted):
        raise ValueError("probability distribution must have non-empty support")
    return tuple(value / total for value in converted)


def _validate_target_temperature(target_temperature: float) -> None:
    if isinstance(target_temperature, bool) or not isinstance(
        target_temperature, int | float
    ):
        raise TypeError("target_temperature must be a real number")
    if target_temperature <= 0.0 or not math.isfinite(target_temperature):
        raise ValueError("target_temperature must be finite and positive")


def temperature_scale_distribution(
    probabilities: Sequence[float], target_temperature: float
) -> tuple[float, ...]:
    """Apply q_tau(a) proportional to q(a) ** (1 / tau) to the target only."""

    values = validate_probability_distribution(probabilities)
    _validate_target_temperature(target_temperature)
    tensor = torch.tensor(values, dtype=torch.float64)
    log_values = torch.full_like(tensor, -torch.inf)
    positive = tensor > 0.0
    log_values[positive] = tensor[positive].log() / target_temperature
    scaled = torch.softmax(log_values, dim=-1)
    return tuple(float(value) for value in scaled.tolist())


def _validate_student_only_episode(
    episode: EpisodeRollout, *, expected_action_size: int
) -> None:
    validate_episode_for_distillation(episode)
    ledger = episode.teacher_costs
    if (
        ledger.probe_count != 0
        or ledger.query_count != 0
        or ledger.generated_teacher_steps != 0
        or ledger.executed_teacher_steps != 0
        or ledger.query_cost != 0.0
        or ledger.execution_cost != 0.0
    ):
        raise ValueError("OPD extraction requires a zero-Teacher rollout ledger")
    if not episode.decisions or not episode.transitions:
        raise ValueError("OPD extraction requires a non-empty actual rollout")

    expected_start = 0
    for expected_id, decision in enumerate(episode.decisions):
        if decision.decision_id != expected_id:
            raise ValueError("Student-only decision ids must be contiguous")
        if decision.probed:
            raise ValueError("OPD extraction rejects probed or hybrid rollouts")
        if decision.selected_option is not OptionKind.STUDENT:
            raise ValueError("Student-only OPD requires every selected option to be S")
        if decision.transition_start != expected_start or decision.duration != 1:
            raise ValueError("Student-only decisions must cover one contiguous primitive step")
        if decision.transition_stop > len(episode.transitions):
            raise ValueError("decision transition range is out of bounds")
        if len(decision.candidates) != 1:
            raise ValueError("unprobed Student-only decisions must retain only S")
        candidate = decision.candidates[0]
        if candidate.kind is not OptionKind.STUDENT:
            raise ValueError("unprobed Student-only candidate must be S")
        if candidate.query_cost != 0.0 or candidate.execution_cost != 0.0:
            raise ValueError("Student-only candidate cannot contain Teacher cost")

        row = episode.transitions[decision.transition_start]
        if row.decision_id != decision.decision_id or row.observation != decision.observation:
            raise ValueError("decision state is not aligned with its actual transition")
        if row.source is not ActionSource.STUDENT or row.selected_option is not OptionKind.STUDENT:
            raise ValueError("Student-only OPD rejects Teacher transition rows")
        if row.query_cost != 0.0 or row.execution_cost != 0.0:
            raise ValueError("Student-only transition cannot contain Teacher cost")
        if row.action != decision.student_proposal.action:
            raise ValueError("actual S action must match the Student proposal")
        if not 0 <= row.action < expected_action_size:
            raise ValueError("Student action lies outside the expected action dimension")
        if candidate.previewed_actions != (row.action,) or candidate.preview_steps != 1:
            raise ValueError("actual S action must match its candidate preview")
        expected_start = decision.transition_stop

    if expected_start != len(episode.transitions):
        raise ValueError("Student-only decisions must cover every actual transition")


def extract_student_only_opd(
    episodes: Sequence[EpisodeRollout],
    annotator: FullActionDistributionAnnotator,
    *,
    expected_action_size: int,
    collection_id: int,
) -> OPDExtraction:
    """Validate a fresh collection completely, then annotate its actual states."""

    if (
        isinstance(expected_action_size, bool)
        or not isinstance(expected_action_size, int)
        or expected_action_size < 2
    ):
        raise ValueError("expected_action_size must be an integer of at least two")
    if isinstance(collection_id, bool) or not isinstance(collection_id, int):
        raise TypeError("collection_id must be an integer")
    if collection_id < 0:
        raise ValueError("collection_id must be non-negative")
    annotator_action_size = annotator.action_size
    if (
        isinstance(annotator_action_size, bool)
        or not isinstance(annotator_action_size, int)
    ):
        raise TypeError("annotator action_size must be an integer")
    if annotator_action_size != expected_action_size:
        raise ValueError("annotator action_size does not match expected_action_size")

    episodes = tuple(episodes)

    # This pass is deliberately query-free. A corrupt later episode must not
    # make earlier annotations disappear without an accounting ledger.
    for episode in episodes:
        _validate_student_only_episode(
            episode, expected_action_size=expected_action_size
        )

    examples: list[OPDExample] = []
    query_count = 0
    query_cost = 0.0
    scored_actions = 0

    def partial_ledger() -> OPDAnnotationLedger:
        return OPDAnnotationLedger(query_count, scored_actions, query_cost)

    for episode_index, episode in enumerate(episodes):
        for decision in episode.decisions:
            try:
                annotation = annotator.annotate(decision.observation)
            except Exception as error:
                raise OPDAnnotationError(
                    "full-action annotation failed", partial_ledger()
                ) from error
            if not isinstance(annotation, ActionDistributionAnnotation):
                raise OPDAnnotationError(
                    "annotator returned an invalid result type", partial_ledger()
                )

            # A returned annotation consumed a query even when its schema is
            # malformed, so account it before expected-dimension validation.
            query_count += 1
            query_cost += annotation.query_cost
            scored_actions += len(annotation.probabilities)
            try:
                probabilities = validate_probability_distribution(
                    annotation.probabilities,
                    action_size=expected_action_size,
                )
                row = OPDExample(
                    observation=decision.observation,
                    teacher_probabilities=probabilities,
                    student_proposal_action=decision.student_proposal.action,
                    collection_id=collection_id,
                    episode_index=episode_index,
                    decision_id=decision.decision_id,
                )
            except Exception as error:
                raise OPDAnnotationError(
                    "full-action annotation was invalid", partial_ledger()
                ) from error
            examples.append(row)
    return OPDExtraction(
        dataset=OPDDataset(collection_id=collection_id, examples=tuple(examples)),
        ledger=partial_ledger(),
    )


def _opd_tensors(
    model: ActorCritic,
    dataset: OPDDataset,
    target_temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    action_size = model.actor_head.out_features
    observations: list[tuple[float, ...]] = []
    targets: list[tuple[float, ...]] = []
    for example in dataset.examples:
        validate_probability_distribution(
            example.teacher_probabilities, action_size=action_size
        )
        observations.append(example.observation)
        targets.append(
            temperature_scale_distribution(
                example.teacher_probabilities, target_temperature
            )
        )
    return (
        torch.tensor(observations, dtype=torch.float32, device=model.device),
        torch.tensor(targets, dtype=torch.float32, device=model.device),
    )


def _actor_logits(model: ActorCritic, observations: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        features = model.encoder(observations)
    return model.actor_head(features.detach())


def opd_forward_kl(
    model: ActorCritic,
    dataset: OPDDataset,
    *,
    target_temperature: float = 1.0,
) -> torch.Tensor:
    """Compute KL(q_target_temperature || pi_student).

    Temperature is applied only to the frozen Teacher target. Student logits
    remain at the policy's execution temperature; this is not bilateral KD.
    """

    _validate_target_temperature(target_temperature)
    if not dataset.examples:
        return model.actor_head.weight.sum() * 0.0
    observations, targets = _opd_tensors(model, dataset, target_temperature)
    student_log_probs = torch.log_softmax(_actor_logits(model, observations), dim=-1)
    positive = targets > 0.0
    teacher_terms = torch.zeros_like(targets)
    teacher_terms[positive] = targets[positive] * targets[positive].log()
    cross_terms = targets * student_log_probs
    return (teacher_terms - cross_terms).sum(dim=-1).mean()


def _diagnostics(
    model: ActorCritic,
    dataset: OPDDataset,
    target_temperature: float,
) -> dict[str, float]:
    if not dataset.examples:
        return {
            "kl": 0.0,
            "student_entropy": 0.0,
            "teacher_entropy": 0.0,
            "disagreement": 0.0,
            "proposal_disagreement": 0.0,
        }
    observations, targets = _opd_tensors(model, dataset, target_temperature)
    with torch.no_grad():
        logits = _actor_logits(model, observations)
        student_log_probs = torch.log_softmax(logits, dim=-1)
        student_probs = student_log_probs.exp()
        positive = targets > 0.0
        teacher_log_probs = torch.zeros_like(targets)
        teacher_log_probs[positive] = targets[positive].log()
        kl = (targets * (teacher_log_probs - student_log_probs)).sum(dim=-1).mean()
        student_entropy = -(student_probs * student_log_probs).sum(dim=-1).mean()
        teacher_entropy = -(targets * teacher_log_probs).sum(dim=-1).mean()
        teacher_actions = targets.argmax(dim=-1)
        disagreement = (teacher_actions != logits.argmax(dim=-1)).float().mean()
        proposals = torch.tensor(
            [example.student_proposal_action for example in dataset.examples],
            dtype=torch.long,
            device=model.device,
        )
        proposal_disagreement = (teacher_actions != proposals).float().mean()
    return {
        "kl": float(kl),
        "student_entropy": float(student_entropy),
        "teacher_entropy": float(teacher_entropy),
        "disagreement": float(disagreement),
        "proposal_disagreement": float(proposal_disagreement),
    }


def opd_update(
    model: ActorCritic,
    dataset: OPDDataset,
    config: OPDConfig,
    *,
    expected_collection_id: int,
) -> dict[str, float]:
    """Immediately consume one fresh collection using actor-head-only SGD."""

    if (
        isinstance(expected_collection_id, bool)
        or not isinstance(expected_collection_id, int)
        or expected_collection_id < 0
    ):
        raise ValueError("expected_collection_id must be a non-negative integer")
    if dataset.collection_id != expected_collection_id:
        raise ValueError("OPD dataset collection_id does not match the current collection")
    before = _diagnostics(model, dataset, config.target_temperature)
    metrics = {
        "opd_examples": float(len(dataset)),
        "opd_optimizer_steps": 0.0,
        "opd_loss": 0.0,
        "opd_kl_before": before["kl"],
        "opd_kl_after": before["kl"],
        "opd_student_entropy_before": before["student_entropy"],
        "opd_student_entropy_after": before["student_entropy"],
        "opd_teacher_entropy": before["teacher_entropy"],
        "opd_disagreement_before": before["disagreement"],
        "opd_disagreement_after": before["disagreement"],
        "opd_proposal_disagreement": before["proposal_disagreement"],
    }
    if config.epochs == 0 or not dataset.examples:
        return metrics

    parameters = tuple(model.actor_head.parameters())
    for _ in range(config.epochs):
        model.actor_head.zero_grad(set_to_none=True)
        loss = opd_forward_kl(
            model,
            dataset,
            target_temperature=config.target_temperature,
        )
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("non-finite OPD forward KL")
        loss.backward()
        nn.utils.clip_grad_norm_(
            parameters, config.max_grad_norm, error_if_nonfinite=True
        )
        with torch.no_grad():
            for parameter in parameters:
                if parameter.grad is not None:
                    parameter.add_(parameter.grad, alpha=-config.learning_rate)
        metrics["opd_loss"] += float(loss.detach())
        metrics["opd_optimizer_steps"] += 1.0
    model.actor_head.zero_grad(set_to_none=True)

    metrics["opd_loss"] /= config.epochs
    after = _diagnostics(model, dataset, config.target_temperature)
    metrics["opd_kl_after"] = after["kl"]
    metrics["opd_student_entropy_after"] = after["student_entropy"]
    metrics["opd_disagreement_after"] = after["disagreement"]
    return metrics
