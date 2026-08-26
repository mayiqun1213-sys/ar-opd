"""Step-level S/T/F candidate evaluation, gating, and rollout collection."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Protocol

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
from ar_opd.teacher import OracleTeacher
from ar_opd.toy_env import JammedChainEnv


class StudentModel(Protocol):
    def act(
        self,
        observation: tuple[float, ...],
        *,
        deterministic: bool = False,
        generator: torch.Generator | None = None,
    ) -> StudentProposal: ...

    def value(self, observation: tuple[float, ...]) -> float: ...


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
    """Toy evaluator: preview actions on clones and bootstrap with PPO V(s)."""

    def build_candidates(
        self,
        env: JammedChainEnv,
        student_proposal: StudentProposal,
        student_model: StudentModel,
        config: RolloutConfig,
        teacher_proposal: TeacherProposal | None,
    ) -> tuple[OptionCandidate, ...]:
        query_cost = config.teacher_query_cost if teacher_proposal is not None else 0.0
        candidates = [
            self._preview(
                env,
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
                        env,
                        kind=OptionKind.TEACHER_CORRECTION,
                        actions=teacher_proposal.correction_actions,
                        student_model=student_model,
                        config=config,
                        query_cost=query_cost,
                    ),
                    self._preview(
                        env,
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
        env: JammedChainEnv,
        *,
        kind: OptionKind,
        actions: tuple[int, ...],
        student_model: StudentModel,
        config: RolloutConfig,
        query_cost: float,
    ) -> OptionCandidate:
        branch = env.clone()
        discounted_reward = 0.0
        discounted_execution_cost = 0.0
        preview_steps = 0
        terminated = False
        truncated = False
        for index, action in enumerate(actions):
            result = branch.step(action)
            discount = config.gamma**index
            discounted_reward += discount * result.reward
            if kind is not OptionKind.STUDENT:
                discounted_execution_cost += discount * config.teacher_execution_cost
            preview_steps += 1
            terminated = result.terminated
            truncated = result.truncated
            if terminated or truncated:
                break

        # The toy observation contains the finite-horizon clock. There is no
        # continuation after either terminal condition, so both bootstrap to 0.
        if not terminated and not truncated:
            bootstrap = student_model.value(branch.encode_observation())
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


class RolloutCollector:
    def __init__(
        self,
        config: RolloutConfig,
        *,
        seed: int = 0,
        evaluator: CounterfactualCandidateEvaluator | None = None,
        gate: ValueGate | None = None,
        torch_generator: torch.Generator | None = None,
    ) -> None:
        self.config = config
        self._random = random.Random(seed)
        self._evaluator = evaluator or CounterfactualCandidateEvaluator()
        self._gate = gate or ValueGate()
        self._torch_generator = torch_generator

    def collect_episode(
        self,
        env: JammedChainEnv,
        student_model: StudentModel,
        teacher: OracleTeacher,
        *,
        deterministic_student: bool = False,
    ) -> EpisodeRollout:
        env.reset()
        transitions: list[Transition] = []
        decisions: list[DecisionRecord] = []
        ledger = TeacherCostLedger()

        while not env.done:
            decision_id = len(decisions)
            encoded = env.encode_observation()
            student_proposal = student_model.act(
                encoded,
                deterministic=deterministic_student,
                generator=self._torch_generator,
            )
            probed = self._random.random() < self.config.probe_probability
            teacher_proposal = None
            if probed:
                ledger.probe_count += 1
                teacher_proposal = teacher.propose(env.observation, self.config.recovery_horizon)
                ledger.query_count += 1
                ledger.generated_teacher_steps += teacher_proposal.generated_steps
                ledger.query_cost += self.config.teacher_query_cost

            candidates = self._evaluator.build_candidates(
                env, student_proposal, student_model, self.config, teacher_proposal
            )
            selected = self._gate.choose(candidates)
            transition_start = len(transitions)
            source = (
                ActionSource.STUDENT
                if selected.kind is OptionKind.STUDENT
                else ActionSource.TEACHER
            )

            for action_index, action in enumerate(selected.actions):
                observation = env.encode_observation()
                query_cost = (
                    self.config.teacher_query_cost if probed and action_index == 0 else 0.0
                )
                execution_cost = (
                    self.config.teacher_execution_cost
                    if source is ActionSource.TEACHER
                    else 0.0
                )
                result = env.step(action)
                if source is ActionSource.TEACHER:
                    ledger.executed_teacher_steps += 1
                    ledger.execution_cost += execution_cost
                transitions.append(
                    Transition(
                        decision_id=decision_id,
                        observation=observation,
                        action=action,
                        next_observation=env.encode_observation(result.observation),
                        env_reward=result.reward,
                        query_cost=query_cost,
                        execution_cost=execution_cost,
                        terminated=result.terminated,
                        truncated=result.truncated,
                        source=source,
                        selected_option=selected.kind,
                    )
                )
                if result.terminated or result.truncated:
                    break

            decisions.append(
                DecisionRecord(
                    decision_id=decision_id,
                    observation=encoded,
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
            success=env.success,
        )
