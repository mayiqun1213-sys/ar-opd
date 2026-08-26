"""Runtime ownership and pluggable policy/Teacher/evaluator contracts."""

from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic, Protocol, TypeVar

import torch

from ar_opd.core import OptionCandidate, StudentProposal, TeacherProposal
from ar_opd.environment import EnvironmentSpec, RolloutEnvironment

if TYPE_CHECKING:
    from ar_opd.rollout import RolloutConfig


ObservationT = TypeVar("ObservationT")


class StudentModel(Protocol):
    def act(
        self,
        observation: tuple[float, ...],
        *,
        deterministic: bool = False,
        generator: torch.Generator | None = None,
    ) -> StudentProposal: ...

    def value(self, observation: tuple[float, ...]) -> float: ...


class TeacherPolicy(Protocol[ObservationT]):
    def propose(
        self,
        observation: ObservationT,
        recovery_horizon: int,
    ) -> TeacherProposal: ...


class CandidateEvaluator(Protocol[ObservationT]):
    """Build exact candidates without changing the online environment state."""

    def build_candidates(
        self,
        environment: RolloutEnvironment[ObservationT],
        student_proposal: StudentProposal,
        student_model: StudentModel,
        config: RolloutConfig,
        teacher_proposal: TeacherProposal | None,
    ) -> tuple[OptionCandidate, ...]: ...

    def close(self) -> None: ...


class GatePolicy(Protocol):
    def choose(self, candidates: tuple[OptionCandidate, ...]) -> OptionCandidate: ...


@dataclass(frozen=True)
class EpisodeComponents(Generic[ObservationT]):
    """Fresh resources owned by exactly one adapter context."""

    environment: RolloutEnvironment[ObservationT]
    teacher: TeacherPolicy[ObservationT] | None
    candidate_evaluator: CandidateEvaluator[ObservationT]


class EnvironmentAdapter(Protocol[ObservationT]):
    """Factory and cleanup boundary for one seeded episode at a time."""

    @property
    def spec(self) -> EnvironmentSpec: ...

    def open_episode(
        self,
        *,
        seed: int,
        require_teacher: bool,
    ) -> AbstractContextManager[EpisodeComponents[ObservationT]]: ...
