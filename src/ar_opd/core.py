"""Core records and invariants for step-level AR-OPD rollouts."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum


class OptionKind(str, Enum):
    """High-level choice made at an environment decision point."""

    STUDENT = "S"
    TEACHER_CORRECTION = "T"
    TEACHER_RECOVERY = "F"


class ActionSource(str, Enum):
    STUDENT = "student"
    TEACHER = "teacher"


@dataclass(frozen=True)
class StudentProposal:
    """The Student action sampled before the value gate is evaluated."""

    action: int
    log_prob: float
    value: float

    def __post_init__(self) -> None:
        if isinstance(self.action, bool) or not isinstance(self.action, int):
            raise TypeError("Student action must be an integer")
        for name, value in (("log_prob", self.log_prob), ("value", self.value)):
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise TypeError(f"Student {name} must be a real number")
            if not math.isfinite(value):
                raise ValueError(f"Student {name} must be finite")


@dataclass(frozen=True)
class TeacherProposal:
    correction_actions: tuple[int, ...]
    recovery_actions: tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "correction_actions", tuple(self.correction_actions))
        object.__setattr__(self, "recovery_actions", tuple(self.recovery_actions))
        if len(self.correction_actions) != 1:
            raise ValueError("teacher correction must contain exactly one action")
        if not self.recovery_actions:
            raise ValueError("teacher recovery must contain at least one action")
        if any(
            isinstance(action, bool) or not isinstance(action, int)
            for action in (*self.correction_actions, *self.recovery_actions)
        ):
            raise TypeError("Teacher actions must be integers")

    @property
    def generated_steps(self) -> int:
        return len(self.correction_actions) + len(self.recovery_actions)


@dataclass(frozen=True)
class OptionCandidate:
    kind: OptionKind
    actions: tuple[int, ...]
    preview_steps: int
    estimated_task_value: float
    query_cost: float
    execution_cost: float
    terminated: bool
    truncated: bool

    def __post_init__(self) -> None:
        if not self.actions:
            raise ValueError("a candidate must contain at least one action")
        if not 1 <= self.preview_steps <= len(self.actions):
            raise ValueError("preview_steps must cover a non-empty action prefix")
        if self.query_cost < 0 or self.execution_cost < 0:
            raise ValueError("teacher costs must be non-negative")

    @property
    def net_score(self) -> float:
        return self.estimated_task_value - self.query_cost - self.execution_cost

    @property
    def previewed_actions(self) -> tuple[int, ...]:
        return self.actions[: self.preview_steps]


@dataclass(frozen=True)
class Transition:
    """One primitive environment step from the actually executed trajectory.

    Student policy metadata intentionally lives on the decision record rather
    than here, so an executed Teacher action cannot be mistaken for a PPO
    action.
    """

    decision_id: int
    observation: tuple[float, ...]
    action: int
    next_observation: tuple[float, ...]
    env_reward: float
    query_cost: float
    execution_cost: float
    terminated: bool
    truncated: bool
    source: ActionSource
    selected_option: OptionKind

    def __post_init__(self) -> None:
        if self.query_cost < 0 or self.execution_cost < 0:
            raise ValueError("transition costs must be non-negative")
        if self.source is ActionSource.STUDENT:
            if self.selected_option is not OptionKind.STUDENT:
                raise ValueError("only a selected S option can execute a Student action")
        elif self.selected_option is OptionKind.STUDENT:
            raise ValueError("a selected S option cannot execute a Teacher action")

    @property
    def net_reward(self) -> float:
        return self.env_reward - self.query_cost - self.execution_cost


@dataclass(frozen=True)
class DecisionRecord:
    decision_id: int
    observation: tuple[float, ...]
    student_proposal: StudentProposal
    probed: bool
    candidates: tuple[OptionCandidate, ...]
    selected_option: OptionKind
    transition_start: int
    transition_stop: int

    def __post_init__(self) -> None:
        if self.transition_stop <= self.transition_start:
            raise ValueError("each decision must execute at least one primitive step")

    @property
    def duration(self) -> int:
        return self.transition_stop - self.transition_start


@dataclass
class TeacherCostLedger:
    probe_count: int = 0
    query_count: int = 0
    generated_teacher_steps: int = 0
    executed_teacher_steps: int = 0
    query_cost: float = 0.0
    execution_cost: float = 0.0

    @property
    def total_cost(self) -> float:
        return self.query_cost + self.execution_cost


@dataclass
class EpisodeRollout:
    transitions: list[Transition]
    decisions: list[DecisionRecord]
    teacher_costs: TeacherCostLedger = field(default_factory=TeacherCostLedger)
    success: bool = False

    @property
    def task_return(self) -> float:
        return sum(row.env_reward for row in self.transitions)

    @property
    def net_return(self) -> float:
        return sum(row.net_reward for row in self.transitions)

    @property
    def actor_rows(self) -> int:
        return len(self.decisions)
