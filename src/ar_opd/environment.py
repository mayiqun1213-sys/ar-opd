"""Framework-neutral environment contracts for step-level AR-OPD rollouts."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar, runtime_checkable


ObservationT = TypeVar("ObservationT")


@dataclass(frozen=True)
class EnvironmentSpec:
    """Fixed model-facing tensor and dense integer-action dimensions."""

    observation_size: int
    action_size: int

    def __post_init__(self) -> None:
        dimensions = (self.observation_size, self.action_size)
        if any(isinstance(value, bool) or not isinstance(value, int) for value in dimensions):
            raise TypeError("environment dimensions must be integers")
        if any(value < 1 for value in dimensions):
            raise ValueError("environment dimensions must be positive")


@dataclass(frozen=True)
class EnvironmentStep(Generic[ObservationT]):
    """One raw environment outcome with explicit terminal semantics."""

    observation: ObservationT
    reward: float
    terminated: bool
    truncated: bool
    success: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.reward, bool) or not isinstance(self.reward, int | float):
            raise TypeError("environment reward must be a real number")
        if not math.isfinite(self.reward):
            raise ValueError("environment reward must be finite")
        flags = (self.terminated, self.truncated, self.success)
        if any(type(value) is not bool for value in flags):
            raise TypeError("environment outcome flags must be booleans")
        if self.terminated and self.truncated:
            raise ValueError("an environment step cannot terminate and truncate together")
        if self.success and not self.terminated:
            raise ValueError("a successful environment step must terminate the episode")


def validate_environment_spec(spec: EnvironmentSpec) -> EnvironmentSpec:
    """Re-run fixed-dimension invariants for adapter-supplied specs."""

    if not isinstance(spec, EnvironmentSpec):
        raise TypeError("environment must expose a valid EnvironmentSpec")
    return EnvironmentSpec(
        observation_size=spec.observation_size,
        action_size=spec.action_size,
    )


@runtime_checkable
class RolloutEnvironment(Protocol[ObservationT]):
    """Online environment required by the generic rollout collector.

    The environment owns raw observations and dense integer actions. Text or
    dynamically enumerated backends require a separate policy-facing codec
    implementing this contract.
    """

    @property
    def spec(self) -> EnvironmentSpec: ...

    def reset(self) -> ObservationT: ...

    def encode_observation(
        self, observation: ObservationT
    ) -> tuple[float, ...]: ...

    def step(self, action: int) -> EnvironmentStep[ObservationT]: ...

    def close(self) -> None: ...


@runtime_checkable
class BranchableRolloutEnvironment(RolloutEnvironment[ObservationT], Protocol[ObservationT]):
    """Optional independent exact-current-state branch used by candidate evaluation."""

    def clone(self) -> BranchableRolloutEnvironment[ObservationT]: ...


def validate_encoded_observation(
    environment: RolloutEnvironment[ObservationT],
    observation: ObservationT,
) -> tuple[float, ...]:
    """Fail closed on variable, malformed, or non-finite model inputs."""

    spec = validate_environment_spec(environment.spec)
    encoded = environment.encode_observation(observation)
    if not isinstance(encoded, tuple):
        raise TypeError("encoded observation must be a tuple")
    if len(encoded) != spec.observation_size:
        raise ValueError("encoded observation has the wrong dimension")
    if any(
        isinstance(value, bool) or not isinstance(value, int | float)
        for value in encoded
    ):
        raise TypeError("encoded observation values must be real numbers")
    values = tuple(float(value) for value in encoded)
    if any(not math.isfinite(value) for value in values):
        raise ValueError("encoded observation values must be finite")
    return values


def validate_environment_action(action: int, spec: EnvironmentSpec) -> int:
    """Return one dense model-facing action after strict schema validation."""

    spec = validate_environment_spec(spec)
    if isinstance(action, bool) or not isinstance(action, int):
        raise TypeError("environment action must be an integer")
    if not 0 <= action < spec.action_size:
        raise ValueError("environment action is outside the action space")
    return action


def validate_environment_step(
    result: EnvironmentStep[ObservationT],
) -> EnvironmentStep[ObservationT]:
    """Reject structural lookalikes that bypass EnvironmentStep invariants."""

    if not isinstance(result, EnvironmentStep):
        raise TypeError("environment step must return EnvironmentStep")
    return EnvironmentStep(
        observation=result.observation,
        reward=result.reward,
        terminated=result.terminated,
        truncated=result.truncated,
        success=result.success,
    )
