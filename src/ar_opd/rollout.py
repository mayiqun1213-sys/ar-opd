"""Step-level S/T/F candidate evaluation, gating, and rollout collection."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import TypeVar, cast

import torch

from ar_opd.core import (
    ActionSource,
    DecisionRecord,
    EpisodeRollout,
    OptionCandidate,
    OptionKind,
    StudentProposal,
    TeacherCostLedger,
    TeacherProposal,
    Transition,
)
from ar_opd.environment import (
    BranchableRolloutEnvironment,
    EnvironmentSpec,
    EnvironmentStep,
    RolloutEnvironment,
    validate_encoded_observation,
    validate_environment_action,
    validate_environment_spec,
    validate_environment_step,
)
from ar_opd.runtime import (
    CandidateEvaluator,
    EnvironmentAdapter,
    GatePolicy,
    StudentModel,
    TeacherPolicy,
)


ObservationT = TypeVar("ObservationT")


@dataclass(frozen=True)
class RolloutConfig:
    gamma: float = 0.97
    probe_probability: float = 1.0
    recovery_horizon: int = 2
    teacher_query_cost: float = 0.01
    teacher_execution_cost: float = 0.02

    def __post_init__(self) -> None:
        if not 0.0 <= self.probe_probability <= 1.0:
            raise ValueError("probe_probability must be in [0, 1]")
        if not 0.0 < self.gamma <= 1.0:
            raise ValueError("gamma must be in (0, 1]")
        if self.recovery_horizon < 1:
            raise ValueError("recovery_horizon must be positive")
        if self.teacher_query_cost < 0 or self.teacher_execution_cost < 0:
            raise ValueError("teacher costs must be non-negative")


class ValueGate:
    """A parameter-free net-value gate with conservative tie breaking."""

    _priority = (
        OptionKind.STUDENT,
        OptionKind.TEACHER_CORRECTION,
        OptionKind.TEACHER_RECOVERY,
    )

    def choose(self, candidates: tuple[OptionCandidate, ...]) -> OptionCandidate:
        by_kind = {candidate.kind: candidate for candidate in candidates}
        if len(by_kind) != len(candidates):
            raise ValueError("candidate option kinds must be unique")
        ordered = [by_kind[kind] for kind in self._priority if kind in by_kind]
        if not ordered:
            raise ValueError("at least one candidate is required")
        if any(not math.isfinite(candidate.net_score) for candidate in ordered):
            raise ValueError("candidate scores must be finite")
        return max(ordered, key=lambda candidate: candidate.net_score)


class CounterfactualCandidateEvaluator:
    """Toy evaluator: preview actions on exact branches and bootstrap with V(s)."""

    @staticmethod
    def validate_environment(environment: RolloutEnvironment[ObservationT]) -> None:
        validate_environment_spec(environment.spec)
        if not callable(getattr(environment, "clone", None)):
            raise TypeError(
                "counterfactual evaluation requires a branchable environment; "
                "inject another CandidateEvaluator for online-only environments"
            )

    def build_candidates(
        self,
        environment: RolloutEnvironment[ObservationT],
        student_proposal: StudentProposal,
        student_model: StudentModel,
        config: RolloutConfig,
        teacher_proposal: TeacherProposal | None,
    ) -> tuple[OptionCandidate, ...]:
        self.validate_environment(environment)
        query_cost = config.teacher_query_cost if teacher_proposal is not None else 0.0
        candidates = [
            self._preview(
                environment,
                kind=OptionKind.STUDENT,
                actions=(student_proposal.action,),
                student_model=student_model,
                config=config,
                query_cost=query_cost,
            )
        ]
        if teacher_proposal is not None:
            candidates.extend(
                (
                    self._preview(
                        environment,
                        kind=OptionKind.TEACHER_CORRECTION,
                        actions=teacher_proposal.correction_actions,
                        student_model=student_model,
                        config=config,
                        query_cost=query_cost,
                    ),
                    self._preview(
                        environment,
                        kind=OptionKind.TEACHER_RECOVERY,
                        actions=teacher_proposal.recovery_actions,
                        student_model=student_model,
                        config=config,
                        query_cost=query_cost,
                    ),
                )
            )
        return tuple(candidates)

    @staticmethod
    def _preview(
        environment: RolloutEnvironment[ObservationT],
        *,
        kind: OptionKind,
        actions: tuple[int, ...],
        student_model: StudentModel,
        config: RolloutConfig,
        query_cost: float,
    ) -> OptionCandidate:
        online_spec = validate_environment_spec(environment.spec)
        for action in actions:
            validate_environment_action(action, online_spec)
        branchable = cast(BranchableRolloutEnvironment[ObservationT], environment)
        branch = branchable.clone()
        if branch is environment:
            raise RuntimeError("environment clone must be independent")
        try:
            branch_spec = validate_environment_spec(branch.spec)
            if branch_spec != online_spec:
                raise ValueError("environment clone changed the model-facing spec")
            discounted_reward = 0.0
            discounted_execution_cost = 0.0
            preview_steps = 0
            terminated = False
            truncated = False
            last_result: EnvironmentStep[ObservationT] | None = None
            for index, action in enumerate(actions):
                result = validate_environment_step(branch.step(action))
                discount = config.gamma**index
                discounted_reward += discount * result.reward
                if kind is not OptionKind.STUDENT:
                    discounted_execution_cost += (
                        discount * config.teacher_execution_cost
                    )
                preview_steps += 1
                terminated = result.terminated
                truncated = result.truncated
                last_result = result
                if terminated or truncated:
                    break

            # Current PPO semantics bootstrap neither task termination nor
            # adapter truncation. Changing truncation bootstrap is an algorithm
            # milestone, not part of this behavior-preserving refactor.
            if not terminated and not truncated:
                if last_result is None:
                    raise AssertionError("candidate preview executed no actions")
                encoded = validate_encoded_observation(branch, last_result.observation)
                bootstrap = student_model.value(encoded)
                if isinstance(bootstrap, bool) or not isinstance(
                    bootstrap, int | float
                ):
                    raise TypeError("Student value must be a real number")
                if not math.isfinite(bootstrap):
                    raise ValueError("Student value must be finite")
                discounted_reward += (config.gamma**preview_steps) * bootstrap
            return OptionCandidate(
                kind=kind,
                actions=actions,
                preview_steps=preview_steps,
                estimated_task_value=discounted_reward,
                query_cost=query_cost,
                execution_cost=discounted_execution_cost,
                terminated=terminated,
                truncated=truncated,
            )
        finally:
            branch.close()

    def close(self) -> None:
        """No persistent scratch resources; branches close after every preview."""


class RolloutCollector:
    def __init__(
        self,
        config: RolloutConfig,
        *,
        seed: int = 0,
        evaluator: CandidateEvaluator[ObservationT] | None = None,
        gate: GatePolicy | None = None,
        torch_generator: torch.Generator | None = None,
    ) -> None:
        self.config = config
        self._random = random.Random(seed)
        self._evaluator = (
            evaluator if evaluator is not None else CounterfactualCandidateEvaluator()
        )
        self._gate = gate if gate is not None else ValueGate()
        self._torch_generator = torch_generator

    def collect_episode(
        self,
        environment: RolloutEnvironment[ObservationT],
        student_model: StudentModel,
        teacher: TeacherPolicy[ObservationT] | None,
        *,
        deterministic_student: bool = False,
    ) -> EpisodeRollout:
        environment_spec = validate_environment_spec(environment.spec)
        if self.config.probe_probability > 0.0 and teacher is None:
            raise ValueError("probing requires a Teacher policy")
        environment_validator = getattr(
            self._evaluator, "validate_environment", None
        )
        if callable(environment_validator):
            environment_validator(environment)

        raw_observation = environment.reset()
        encoded = validate_encoded_observation(environment, raw_observation)
        transitions: list[Transition] = []
        decisions: list[DecisionRecord] = []
        ledger = TeacherCostLedger()
        terminated = False
        truncated = False
        success = False

        while not terminated and not truncated:
            decision_id = len(decisions)
            decision_observation = encoded
            proposed_student = student_model.act(
                encoded,
                deterministic=deterministic_student,
                generator=self._torch_generator,
            )
            if not isinstance(proposed_student, StudentProposal):
                raise TypeError("Student model must return StudentProposal")
            student_proposal = StudentProposal(
                action=proposed_student.action,
                log_prob=proposed_student.log_prob,
                value=proposed_student.value,
            )
            validate_environment_action(student_proposal.action, environment_spec)
            probed = self._random.random() < self.config.probe_probability
            teacher_proposal = None
            if probed:
                if teacher is None:
                    raise AssertionError("Teacher disappeared after validation")
                ledger.probe_count += 1
                proposed_teacher = teacher.propose(
                    raw_observation,
                    self.config.recovery_horizon,
                )
                if not isinstance(proposed_teacher, TeacherProposal):
                    raise TypeError("Teacher must return TeacherProposal")
                teacher_proposal = TeacherProposal(
                    correction_actions=proposed_teacher.correction_actions,
                    recovery_actions=proposed_teacher.recovery_actions,
                )
                for action in (
                    *teacher_proposal.correction_actions,
                    *teacher_proposal.recovery_actions,
                ):
                    validate_environment_action(action, environment_spec)
                ledger.query_count += 1
                ledger.generated_teacher_steps += len(
                    teacher_proposal.correction_actions
                ) + len(teacher_proposal.recovery_actions)
                ledger.query_cost += self.config.teacher_query_cost

            candidates = self._evaluator.build_candidates(
                environment,
                student_proposal,
                student_model,
                self.config,
                teacher_proposal,
            )
            candidates = self._validate_candidates(
                candidates,
                student_proposal=student_proposal,
                teacher_proposal=teacher_proposal,
                environment_spec=environment_spec,
            )
            selected = self._gate.choose(candidates)
            if not any(selected is candidate for candidate in candidates):
                raise ValueError("gate must select one of the supplied candidates")
            transition_start = len(transitions)
            source = (
                ActionSource.STUDENT
                if selected.kind is OptionKind.STUDENT
                else ActionSource.TEACHER
            )

            for action_index, action in enumerate(selected.previewed_actions):
                observation = encoded
                query_cost = (
                    self.config.teacher_query_cost
                    if probed and action_index == 0
                    else 0.0
                )
                execution_cost = (
                    self.config.teacher_execution_cost
                    if source is ActionSource.TEACHER
                    else 0.0
                )
                result = validate_environment_step(environment.step(action))
                next_observation = validate_encoded_observation(
                    environment, result.observation
                )
                if source is ActionSource.TEACHER:
                    ledger.executed_teacher_steps += 1
                    ledger.execution_cost += execution_cost
                transitions.append(
                    Transition(
                        decision_id=decision_id,
                        observation=observation,
                        action=action,
                        next_observation=next_observation,
                        env_reward=result.reward,
                        query_cost=query_cost,
                        execution_cost=execution_cost,
                        terminated=result.terminated,
                        truncated=result.truncated,
                        source=source,
                        selected_option=selected.kind,
                    )
                )
                raw_observation = result.observation
                encoded = next_observation
                terminated = result.terminated
                truncated = result.truncated
                success = result.success
                if terminated or truncated:
                    break

            executed_steps = len(transitions) - transition_start
            if executed_steps != selected.preview_steps:
                raise ValueError(
                    "selected candidate preview length differs from execution"
                )
            if (
                terminated is not selected.terminated
                or truncated is not selected.truncated
            ):
                raise ValueError(
                    "selected candidate terminal state differs from execution"
                )

            decisions.append(
                DecisionRecord(
                    decision_id=decision_id,
                    observation=decision_observation,
                    student_proposal=student_proposal,
                    probed=probed,
                    candidates=candidates,
                    selected_option=selected.kind,
                    transition_start=transition_start,
                    transition_stop=len(transitions),
                )
            )

        return EpisodeRollout(
            transitions=transitions,
            decisions=decisions,
            teacher_costs=ledger,
            success=success,
        )

    def _validate_candidates(
        self,
        candidates: tuple[OptionCandidate, ...],
        *,
        student_proposal: StudentProposal,
        teacher_proposal: TeacherProposal | None,
        environment_spec: EnvironmentSpec,
    ) -> tuple[OptionCandidate, ...]:
        """Validate evaluator output before the gate or PPO can consume it."""

        if not isinstance(candidates, tuple) or not candidates:
            raise TypeError("CandidateEvaluator must return a non-empty tuple")

        expected_actions = {OptionKind.STUDENT: (student_proposal.action,)}
        if teacher_proposal is not None:
            expected_actions.update(
                {
                    OptionKind.TEACHER_CORRECTION: teacher_proposal.correction_actions,
                    OptionKind.TEACHER_RECOVERY: teacher_proposal.recovery_actions,
                }
            )

        by_kind: dict[OptionKind, OptionCandidate] = {}
        canonical_candidates: list[OptionCandidate] = []
        expected_query_cost = (
            self.config.teacher_query_cost if teacher_proposal is not None else 0.0
        )
        for candidate in candidates:
            if not isinstance(candidate, OptionCandidate):
                raise TypeError("candidate rows must be OptionCandidate records")
            candidate = OptionCandidate(
                kind=candidate.kind,
                actions=candidate.actions,
                preview_steps=candidate.preview_steps,
                estimated_task_value=candidate.estimated_task_value,
                query_cost=candidate.query_cost,
                execution_cost=candidate.execution_cost,
                terminated=candidate.terminated,
                truncated=candidate.truncated,
            )
            canonical_candidates.append(candidate)
            if not isinstance(candidate.kind, OptionKind):
                raise TypeError("candidate kind must be an OptionKind")
            if candidate.kind in by_kind:
                raise ValueError("candidate option kinds must be unique")
            by_kind[candidate.kind] = candidate
            if not isinstance(candidate.actions, tuple):
                raise TypeError("candidate actions must be a tuple")
            for action in candidate.actions:
                validate_environment_action(action, environment_spec)
            expected = expected_actions.get(candidate.kind)
            if expected is None:
                raise ValueError("candidate kind is inconsistent with the probe")
            if candidate.actions != expected:
                raise ValueError("candidate actions differ from the sampled proposal")
            if (
                isinstance(candidate.preview_steps, bool)
                or not isinstance(candidate.preview_steps, int)
            ):
                raise TypeError("candidate preview_steps must be an integer")
            for name, value in (
                ("estimated_task_value", candidate.estimated_task_value),
                ("query_cost", candidate.query_cost),
                ("execution_cost", candidate.execution_cost),
            ):
                if isinstance(value, bool) or not isinstance(value, int | float):
                    raise TypeError(f"candidate {name} must be a real number")
                if not math.isfinite(value):
                    raise ValueError(f"candidate {name} must be finite")
            if (
                type(candidate.terminated) is not bool
                or type(candidate.truncated) is not bool
            ):
                raise TypeError("candidate outcome flags must be booleans")
            if candidate.terminated and candidate.truncated:
                raise ValueError("candidate cannot terminate and truncate together")
            if (
                not candidate.terminated
                and not candidate.truncated
                and candidate.preview_steps != len(candidate.actions)
            ):
                raise ValueError("an unfinished candidate must preview every action")
            if not math.isclose(
                candidate.query_cost,
                expected_query_cost,
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise ValueError("candidate query cost differs from the probe cost")
            expected_execution_cost = 0.0
            if candidate.kind is not OptionKind.STUDENT:
                expected_execution_cost = sum(
                    (self.config.gamma**index)
                    * self.config.teacher_execution_cost
                    for index in range(candidate.preview_steps)
                )
            if not math.isclose(
                candidate.execution_cost,
                expected_execution_cost,
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    "candidate execution cost differs from its previewed actions"
                )

        if set(by_kind) != set(expected_actions):
            raise ValueError("candidate set is inconsistent with the probe")
        return tuple(canonical_candidates)


def collect_episodes(
    adapter: EnvironmentAdapter[ObservationT],
    student_model: StudentModel,
    config: RolloutConfig,
    *,
    count: int,
    seed: int,
    deterministic_student: bool,
    generator: torch.Generator | None,
) -> list[EpisodeRollout]:
    """Collect independently owned, explicitly seeded episode sessions."""

    if isinstance(count, bool) or not isinstance(count, int):
        raise TypeError("episode count must be an integer")
    if count < 1:
        raise ValueError("episode count must be positive")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("episode seed must be an integer")
    adapter_spec = validate_environment_spec(adapter.spec)

    episodes: list[EpisodeRollout] = []
    require_teacher = config.probe_probability > 0.0
    for episode_index in range(count):
        episode_seed = seed + episode_index
        with adapter.open_episode(
            seed=episode_seed,
            require_teacher=require_teacher,
        ) as components:
            episode_spec = validate_environment_spec(components.environment.spec)
            if episode_spec != adapter_spec:
                raise ValueError("episode environment spec differs from adapter spec")
            if require_teacher and components.teacher is None:
                raise ValueError("adapter omitted the required Teacher policy")
            if components.candidate_evaluator is None:
                raise ValueError("adapter omitted the required candidate evaluator")
            collector = RolloutCollector(
                config,
                seed=episode_seed,
                evaluator=components.candidate_evaluator,
                torch_generator=generator,
            )
            episodes.append(
                collector.collect_episode(
                    components.environment,
                    student_model,
                    components.teacher,
                    deterministic_student=deterministic_student,
                )
            )
    return episodes
