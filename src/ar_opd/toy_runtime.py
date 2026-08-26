"""Toy-specific runtime assembly and full-distribution oracle annotation."""

from __future__ import annotations

import math
from collections.abc import Iterator
from contextlib import contextmanager

from ar_opd.opd import ActionDistributionAnnotation
from ar_opd.rollout import CounterfactualCandidateEvaluator
from ar_opd.runtime import EpisodeComponents
from ar_opd.teacher import OracleTeacher
from ar_opd.toy_env import ChainAction, JammedChainConfig, JammedChainEnv, ToyObservation


class ToyRuntimeAdapter:
    """Create fresh deterministic toy resources with explicit episode ownership."""

    spec = JammedChainEnv.spec

    def __init__(self, config: JammedChainConfig | None = None) -> None:
        self.config = config or JammedChainConfig()

    @contextmanager
    def open_episode(
        self,
        *,
        seed: int,
        require_teacher: bool,
    ) -> Iterator[EpisodeComponents[ToyObservation]]:
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("episode seed must be an integer")
        if type(require_teacher) is not bool:
            raise TypeError("require_teacher must be a boolean")

        environment = JammedChainEnv(self.config, episode_seed=seed)
        evaluator: CounterfactualCandidateEvaluator | None = None
        try:
            evaluator = CounterfactualCandidateEvaluator()
            teacher = OracleTeacher(self.config) if require_teacher else None
            yield EpisodeComponents(environment, teacher, evaluator)
        finally:
            try:
                if evaluator is not None:
                    evaluator.close()
            finally:
                environment.close()


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
