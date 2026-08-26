"""A deterministic recovery environment used for the first runnable loop."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

from ar_opd.environment import EnvironmentSpec, EnvironmentStep


class ChainAction(IntEnum):
    ADVANCE = 0
    REPAIR = 1


@dataclass(frozen=True)
class JammedChainConfig:
    goal_position: int = 5
    trap_positions: tuple[int, ...] = (2, 4)
    max_steps: int = 16
    step_reward: float = -0.01
    repair_reward: float = -0.03
    invalid_reward: float = -0.20
    unnecessary_repair_reward: float = -0.10
    success_reward: float = 1.0

    def __post_init__(self) -> None:
        if self.goal_position < 1:
            raise ValueError("goal_position must be positive")
        if self.max_steps < 1:
            raise ValueError("max_steps must be positive")
        if any(
            position <= 0 or position >= self.goal_position
            for position in self.trap_positions
        ):
            raise ValueError("traps must be strictly between the start and goal")


@dataclass(frozen=True)
class ToyObservation:
    position: int
    jammed: bool
    steps_elapsed: int
    goal_position: int
    max_steps: int

    def encode(self) -> tuple[float, float, float]:
        return (
            self.position / self.goal_position,
            float(self.jammed),
            max(0.0, (self.max_steps - self.steps_elapsed) / self.max_steps),
        )


@dataclass(frozen=True)
class ToyStepResult(EnvironmentStep[ToyObservation]):
    pass


class JammedChainEnv:
    """A short chain where an oracle can correct or recover from jams."""

    spec = EnvironmentSpec(observation_size=3, action_size=len(ChainAction))
    observation_size = spec.observation_size
    action_size = spec.action_size

    def __init__(
        self,
        config: JammedChainConfig | None = None,
        *,
        episode_seed: int | None = None,
    ) -> None:
        if episode_seed is not None and (
            isinstance(episode_seed, bool) or not isinstance(episode_seed, int)
        ):
            raise TypeError("episode_seed must be an integer or None")
        self.config = config or JammedChainConfig()
        self.episode_seed = episode_seed
        self._position = 0
        self._jammed = False
        self._steps = 0
        self._terminated = False
        self._truncated = False
        self._closed = False

    @property
    def closed(self) -> bool:
        return self._closed

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("environment is closed")

    def reset(self) -> ToyObservation:
        self._ensure_open()
        self._position = 0
        self._jammed = False
        self._steps = 0
        self._terminated = False
        self._truncated = False
        return self.observation

    @property
    def observation(self) -> ToyObservation:
        self._ensure_open()
        return ToyObservation(
            position=self._position,
            jammed=self._jammed,
            steps_elapsed=self._steps,
            goal_position=self.config.goal_position,
            max_steps=self.config.max_steps,
        )

    @property
    def success(self) -> bool:
        self._ensure_open()
        return self._terminated and self._position >= self.config.goal_position

    @property
    def done(self) -> bool:
        self._ensure_open()
        return self._terminated or self._truncated

    def encode_observation(
        self, observation: ToyObservation | None = None
    ) -> tuple[float, ...]:
        self._ensure_open()
        if observation is not None and not isinstance(observation, ToyObservation):
            raise TypeError("toy observation has the wrong type")
        return (observation or self.observation).encode()

    def clone(self) -> JammedChainEnv:
        self._ensure_open()
        clone = JammedChainEnv(self.config, episode_seed=self.episode_seed)
        clone._position = self._position
        clone._jammed = self._jammed
        clone._steps = self._steps
        clone._terminated = self._terminated
        clone._truncated = self._truncated
        return clone

    def step(self, action: int) -> ToyStepResult:
        self._ensure_open()
        if self.done:
            raise RuntimeError("step called after the episode ended")
        if isinstance(action, bool) or not isinstance(action, int):
            raise TypeError("toy action must be an integer")
        try:
            action_kind = ChainAction(action)
        except ValueError as error:
            raise ValueError(f"unknown action: {action}") from error

        if self._jammed:
            if action_kind is ChainAction.REPAIR:
                self._jammed = False
                reward = self.config.repair_reward
            else:
                reward = self.config.invalid_reward
        elif action_kind is ChainAction.ADVANCE:
            self._position += 1
            if self._position >= self.config.goal_position:
                reward = self.config.success_reward
            else:
                reward = self.config.step_reward
                if self._position in self.config.trap_positions:
                    self._jammed = True
        else:
            reward = self.config.unnecessary_repair_reward

        self._steps += 1
        self._terminated = self._position >= self.config.goal_position
        self._truncated = self._steps >= self.config.max_steps and not self._terminated
        return ToyStepResult(
            observation=self.observation,
            reward=reward,
            terminated=self._terminated,
            truncated=self._truncated,
            success=self.success,
        )

    def close(self) -> None:
        self._closed = True
