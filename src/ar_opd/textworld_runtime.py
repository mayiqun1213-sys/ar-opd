"""Dependency-free TextWorld runtime, replay, and candidate-evaluation core.

This module intentionally has no Java or TextWorldExpress import. A concrete
backend only has to translate reset/step results into the strict records below.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from ar_opd.core import OptionCandidate, OptionKind, StudentProposal, TeacherProposal
from ar_opd.environment import (
    EnvironmentSpec,
    EnvironmentStep,
    RolloutEnvironment,
    validate_environment_action,
    validate_environment_spec,
)
from ar_opd.runtime import EpisodeComponents, StudentModel, TeacherPolicy

if TYPE_CHECKING:
    from ar_opd.rollout import RolloutConfig


_ACTION_CODEC_SCHEMA = "ar-opd-fixed-vocabulary-action-codec-v1"
_ENCODER_SCHEMA = "ar-opd-stable-text-observation-encoder-v1"
_BOUNDARY_SCHEMA = "ar-opd-textworld-boundary-v1"
_REPLAY_STEP_SCHEMA = "ar-opd-textworld-replay-step-v1"
_REPLAY_CURSOR_SCHEMA = "ar-opd-textworld-replay-cursor-v1"
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_GAME_PARAM_KEY_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9_]*")
_GAME_PARAM_VALUE_PATTERN = re.compile(r"[+-]?\d+")
_TEXTWORLD_FOLDS = frozenset(("train", "dev", "test"))
_SIGNED_INT32_MIN = -(2**31)
_SIGNED_INT32_MAX = 2**31 - 1


def _sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_sha256(value: str, name: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{name} must be a string")
    if _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _require_int(value: int, name: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _require_signed_int32(value: int, name: str) -> int:
    value = _require_int(value, name)
    if not _SIGNED_INT32_MIN <= value <= _SIGNED_INT32_MAX:
        raise ValueError(f"{name} must fit a signed 32-bit integer")
    return value


def _require_float(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{name} must be a real number")
    canonical = float(value)
    if not math.isfinite(canonical):
        raise ValueError(f"{name} must be finite")
    return canonical


def _require_string(
    value: str,
    name: str,
    *,
    allow_empty: bool = False,
    canonical_whitespace: bool = False,
) -> str:
    if type(value) is not str:
        raise TypeError(f"{name} must be a string")
    if not allow_empty and not value:
        raise ValueError(f"{name} must not be empty")
    if canonical_whitespace and value != value.strip():
        raise ValueError(f"{name} must not have leading or trailing whitespace")
    if "\x00" in value:
        raise ValueError(f"{name} must not contain NUL")
    return value


def _canonical_commands(
    commands: tuple[str, ...],
    name: str,
    *,
    allow_empty: bool,
) -> tuple[str, ...]:
    if isinstance(commands, (str, bytes)):
        raise TypeError(f"{name} must be an iterable of command strings")
    try:
        canonical = tuple(commands)
    except TypeError as error:
        raise TypeError(f"{name} must be an iterable of command strings") from error
    if not allow_empty and not canonical:
        raise ValueError(f"{name} must not be empty")
    for command in canonical:
        _require_string(command, name, canonical_whitespace=True)
    if len(set(canonical)) != len(canonical):
        raise ValueError(f"{name} must not contain duplicate commands")
    return canonical


def _canonical_game_params(value: str) -> str:
    value = _require_string(value, "game_params", allow_empty=True)
    if not value:
        return ""
    parsed: dict[str, int] = {}
    for raw_item in value.split(","):
        item = raw_item.strip()
        if item.count("=") != 1:
            raise ValueError("game_params entries must use key=integer syntax")
        key, raw_number = (part.strip() for part in item.split("=", 1))
        if _GAME_PARAM_KEY_PATTERN.fullmatch(key) is None:
            raise ValueError(f"invalid game_params key: {key!r}")
        if _GAME_PARAM_VALUE_PATTERN.fullmatch(raw_number) is None:
            raise ValueError(f"game_params value for {key!r} must be an integer")
        if key in parsed:
            raise ValueError(f"duplicate game_params key: {key!r}")
        parsed[key] = _require_signed_int32(
            int(raw_number),
            f"game_params value for {key!r}",
        )
    return ",".join(f"{key}={parsed[key]}" for key in sorted(parsed))


def _action_codec_sha256(vocabulary: tuple[str, ...]) -> str:
    return _sha256(
        {
            "schema": _ACTION_CODEC_SCHEMA,
            "vocabulary": list(vocabulary),
        }
    )


def _encoder_sha256(observation_size: int) -> str:
    return _sha256(
        {
            "schema": _ENCODER_SCHEMA,
            "observation_size": observation_size,
        }
    )


class TaskOutcome(str, Enum):
    """Derived task status, deliberately separate from a local step limit."""

    ACTIVE = "active"
    SUCCESS = "success"
    FAILURE = "failure"


def _derive_task_outcome(
    *,
    score: float,
    task_success: bool,
    task_failure: bool,
) -> TaskOutcome:
    score = _require_float(score, "normalized task score")
    if type(task_success) is not bool or type(task_failure) is not bool:
        raise TypeError("task outcome flags must be booleans")
    if task_success and task_failure:
        raise ValueError("task success and failure flags cannot both be true")
    if task_failure:
        return TaskOutcome.FAILURE
    if task_success or score >= 1.0:
        return TaskOutcome.SUCCESS
    return TaskOutcome.ACTIVE


@dataclass(frozen=True)
class TextWorldRuntimeConfig:
    """Immutable policy and backend ABI shared by every episode."""

    game_name: str
    backend_identity: str
    action_vocabulary: tuple[str, ...]
    observation_size: int
    project_max_steps: int
    game_fold: str = "train"
    game_params: str = ""
    generate_gold_path: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "game_name",
            _require_string(
                self.game_name,
                "game_name",
                canonical_whitespace=True,
            ),
        )
        object.__setattr__(
            self,
            "backend_identity",
            _require_string(
                self.backend_identity,
                "backend_identity",
                canonical_whitespace=True,
            ),
        )
        object.__setattr__(
            self,
            "game_fold",
            _require_string(
                self.game_fold,
                "game_fold",
                canonical_whitespace=True,
            ),
        )
        if self.game_fold not in _TEXTWORLD_FOLDS:
            raise ValueError("game_fold must be one of: train, dev, test")
        object.__setattr__(
            self,
            "game_params",
            _canonical_game_params(self.game_params),
        )
        action_vocabulary = _canonical_commands(
            self.action_vocabulary,
            "action_vocabulary",
            allow_empty=False,
        )
        if "help" in action_vocabulary:
            raise ValueError("the TextWorldExpress help command is not a policy action")
        object.__setattr__(
            self,
            "action_vocabulary",
            action_vocabulary,
        )
        object.__setattr__(
            self,
            "observation_size",
            _require_int(self.observation_size, "observation_size", minimum=1),
        )
        object.__setattr__(
            self,
            "project_max_steps",
            _require_int(self.project_max_steps, "project_max_steps", minimum=1),
        )
        if type(self.generate_gold_path) is not bool:
            raise TypeError("generate_gold_path must be a boolean")

    @property
    def action_codec_sha256(self) -> str:
        return _action_codec_sha256(self.action_vocabulary)

    @property
    def encoder_sha256(self) -> str:
        return _encoder_sha256(self.observation_size)

    @property
    def abi_sha256(self) -> str:
        return _sha256(
            {
                "action_codec_sha256": self.action_codec_sha256,
                "backend_identity": self.backend_identity,
                "encoder_sha256": self.encoder_sha256,
                "game_fold": self.game_fold,
                "game_name": self.game_name,
                "game_params": self.game_params,
                "generate_gold_path": self.generate_gold_path,
                "project_max_steps": self.project_max_steps,
                "schema": "ar-opd-textworld-runtime-config-v1",
            }
        )


def _snapshot_config(config: TextWorldRuntimeConfig) -> TextWorldRuntimeConfig:
    if not isinstance(config, TextWorldRuntimeConfig):
        raise TypeError("config must be TextWorldRuntimeConfig")
    return TextWorldRuntimeConfig(
        game_name=config.game_name,
        backend_identity=config.backend_identity,
        action_vocabulary=config.action_vocabulary,
        observation_size=config.observation_size,
        project_max_steps=config.project_max_steps,
        game_fold=config.game_fold,
        game_params=config.game_params,
        generate_gold_path=config.generate_gold_path,
    )


@dataclass(frozen=True)
class TextWorldEpisodeSpec:
    """Complete explicit reset identity for one episode."""

    config: TextWorldRuntimeConfig
    seed: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "config", _snapshot_config(self.config))
        object.__setattr__(
            self,
            "seed",
            _require_signed_int32(self.seed, "episode seed"),
        )

    @property
    def abi_sha256(self) -> str:
        return _sha256(
            {
                "config_sha256": self.config.abi_sha256,
                "schema": "ar-opd-textworld-episode-v1",
                "seed": self.seed,
            }
        )

    @property
    def game_name(self) -> str:
        return self.config.game_name

    @property
    def game_fold(self) -> str:
        return self.config.game_fold

    @property
    def game_params(self) -> str:
        return self.config.game_params

    @property
    def generate_gold_path(self) -> bool:
        return self.config.generate_gold_path


def _snapshot_episode(spec: TextWorldEpisodeSpec) -> TextWorldEpisodeSpec:
    if not isinstance(spec, TextWorldEpisodeSpec):
        raise TypeError("episode spec must be TextWorldEpisodeSpec")
    return TextWorldEpisodeSpec(config=spec.config, seed=spec.seed)


@dataclass(frozen=True)
class BackendBoundary:
    """Canonical observable state returned by a low-level text backend."""

    text: str
    look: str
    inventory: str
    valid_actions: tuple[str, ...]
    score_raw: float
    score: float
    task_description: str
    task_success: bool = False
    task_failure: bool = False
    state_token: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "text",
            _require_string(self.text, "backend observation", allow_empty=True),
        )
        object.__setattr__(
            self,
            "look",
            _require_string(self.look, "backend look", allow_empty=True),
        )
        object.__setattr__(
            self,
            "inventory",
            _require_string(self.inventory, "backend inventory", allow_empty=True),
        )
        object.__setattr__(
            self,
            "valid_actions",
            _canonical_commands(
                self.valid_actions,
                "backend valid_actions",
                allow_empty=True,
            ),
        )
        object.__setattr__(
            self,
            "score_raw",
            _require_float(self.score_raw, "backend raw score"),
        )
        object.__setattr__(self, "score", _require_float(self.score, "backend score"))
        object.__setattr__(
            self,
            "task_description",
            _require_string(
                self.task_description,
                "backend task_description",
                allow_empty=True,
            ),
        )
        _derive_task_outcome(
            score=self.score,
            task_success=self.task_success,
            task_failure=self.task_failure,
        )
        object.__setattr__(
            self,
            "state_token",
            _require_string(self.state_token, "backend state_token", allow_empty=True),
        )

    @property
    def task_outcome(self) -> TaskOutcome:
        return _derive_task_outcome(
            score=self.score,
            task_success=self.task_success,
            task_failure=self.task_failure,
        )


def _snapshot_backend_boundary(boundary: BackendBoundary) -> BackendBoundary:
    if not isinstance(boundary, BackendBoundary):
        raise TypeError("backend must return BackendBoundary")
    return BackendBoundary(
        text=boundary.text,
        look=boundary.look,
        inventory=boundary.inventory,
        valid_actions=boundary.valid_actions,
        score_raw=boundary.score_raw,
        score=boundary.score,
        task_description=boundary.task_description,
        task_success=boundary.task_success,
        task_failure=boundary.task_failure,
        state_token=boundary.state_token,
    )


@dataclass(frozen=True)
class BackendTransition:
    """One raw backend result; runtime derives task outcome from its boundary."""

    boundary: BackendBoundary
    done: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "boundary", _snapshot_backend_boundary(self.boundary))
        if type(self.done) is not bool:
            raise TypeError("backend done must be a boolean")


def _snapshot_backend_transition(result: BackendTransition) -> BackendTransition:
    if not isinstance(result, BackendTransition):
        raise TypeError("backend step must return BackendTransition")
    return BackendTransition(
        boundary=result.boundary,
        done=result.done,
    )


@runtime_checkable
class TextWorldBackend(Protocol):
    """Minimal raw surface implemented by fake and future JVM wrappers.

    A concrete wrapper must return every canonical boundary field and keep any
    upstream step cap above ``project_max_steps``.
    """

    def reset(self, episode_spec: TextWorldEpisodeSpec) -> BackendBoundary: ...

    def step(self, command: str) -> BackendTransition: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class ActionChoice:
    """One globally stable dense action ID and its exact backend command."""

    action_id: int
    command: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "action_id",
            _require_int(self.action_id, "action_id", minimum=0),
        )
        object.__setattr__(
            self,
            "command",
            _require_string(
                self.command,
                "action command",
                canonical_whitespace=True,
            ),
        )


def _snapshot_choice(choice: ActionChoice) -> ActionChoice:
    if not isinstance(choice, ActionChoice):
        raise TypeError("action choices must be ActionChoice records")
    return ActionChoice(action_id=choice.action_id, command=choice.command)


@dataclass(frozen=True)
class ActionView:
    """The dynamic valid subset of one fixed action vocabulary."""

    choices: tuple[ActionChoice, ...]
    mask: tuple[bool, ...]
    codec_sha256: str

    def __post_init__(self) -> None:
        if isinstance(self.choices, (str, bytes)):
            raise TypeError("action choices must be an iterable")
        try:
            choices = tuple(_snapshot_choice(choice) for choice in self.choices)
        except TypeError as error:
            raise TypeError("action choices must be an iterable") from error
        try:
            mask = tuple(self.mask)
        except TypeError as error:
            raise TypeError("action mask must be an iterable") from error
        if not mask:
            raise ValueError("action mask must not be empty")
        if any(type(value) is not bool for value in mask):
            raise TypeError("action mask values must be booleans")
        _require_sha256(self.codec_sha256, "codec_sha256")
        ids = tuple(choice.action_id for choice in choices)
        if ids != tuple(sorted(ids)) or len(set(ids)) != len(ids):
            raise ValueError("action choices must have unique ascending IDs")
        if any(action_id >= len(mask) for action_id in ids):
            raise ValueError("action choice lies outside the fixed action space")
        expected_ids = tuple(index for index, valid in enumerate(mask) if valid)
        if ids != expected_ids:
            raise ValueError("action choices and action mask disagree")
        object.__setattr__(self, "choices", choices)
        object.__setattr__(self, "mask", mask)

    @property
    def action_size(self) -> int:
        return len(self.mask)

    @property
    def action_ids(self) -> tuple[int, ...]:
        return tuple(choice.action_id for choice in self.choices)

    def command_for(self, action_id: int) -> str:
        action_id = _require_int(action_id, "action_id", minimum=0)
        if action_id >= len(self.mask):
            raise ValueError("action_id lies outside the fixed action space")
        if not self.mask[action_id]:
            raise ValueError("action_id is not valid at the current boundary")
        by_id = {choice.action_id: choice.command for choice in self.choices}
        return by_id[action_id]


def _snapshot_action_view(view: ActionView) -> ActionView:
    if not isinstance(view, ActionView):
        raise TypeError("action view must be ActionView")
    return ActionView(
        choices=view.choices,
        mask=view.mask,
        codec_sha256=view.codec_sha256,
    )


class FixedVocabularyActionCodec:
    """Map exact commands to stable IDs while exposing a dynamic valid mask."""

    def __init__(self, vocabulary: tuple[str, ...]) -> None:
        self._vocabulary = _canonical_commands(
            vocabulary,
            "action vocabulary",
            allow_empty=False,
        )
        if "help" in self._vocabulary:
            raise ValueError("the TextWorldExpress help command is not a policy action")
        self._ids = {command: index for index, command in enumerate(self._vocabulary)}
        self._abi_sha256 = _action_codec_sha256(self._vocabulary)

    @property
    def vocabulary(self) -> tuple[str, ...]:
        return self._vocabulary

    @property
    def action_size(self) -> int:
        return len(self._vocabulary)

    @property
    def abi_sha256(self) -> str:
        return self._abi_sha256

    def bind(self, valid_commands: tuple[str, ...]) -> ActionView:
        commands = _canonical_commands(
            valid_commands,
            "valid commands",
            allow_empty=True,
        )
        unknown = tuple(command for command in commands if command not in self._ids)
        if unknown:
            raise ValueError(f"valid command is absent from the fixed vocabulary: {unknown[0]!r}")
        ids = sorted(self._ids[command] for command in commands)
        choices = tuple(ActionChoice(action_id, self._vocabulary[action_id]) for action_id in ids)
        valid_ids = set(ids)
        mask = tuple(index in valid_ids for index in range(self.action_size))
        return ActionView(
            choices=choices,
            mask=mask,
            codec_sha256=self.abi_sha256,
        )

    def decode(self, action_id: int, view: ActionView) -> str:
        view = _snapshot_action_view(view)
        if view.codec_sha256 != self.abi_sha256 or view.action_size != self.action_size:
            raise ValueError("action view belongs to a different codec ABI")
        command = view.command_for(action_id)
        if self._vocabulary[action_id] != command:
            raise ValueError("action view changed a stable action mapping")
        return command


@dataclass(frozen=True)
class TextWorldObservation:
    """Immutable raw observation delivered to a Teacher policy."""

    text: str
    look: str
    inventory: str
    action_view: ActionView
    score_raw: float
    score: float
    task_description: str
    steps_elapsed: int
    task_success: bool = False
    task_failure: bool = False
    truncated: bool = False
    state_token: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "text",
            _require_string(self.text, "observation text", allow_empty=True),
        )
        object.__setattr__(
            self,
            "look",
            _require_string(self.look, "observation look", allow_empty=True),
        )
        object.__setattr__(
            self,
            "inventory",
            _require_string(self.inventory, "observation inventory", allow_empty=True),
        )
        object.__setattr__(self, "action_view", _snapshot_action_view(self.action_view))
        object.__setattr__(
            self,
            "score_raw",
            _require_float(self.score_raw, "observation raw score"),
        )
        object.__setattr__(self, "score", _require_float(self.score, "observation score"))
        object.__setattr__(
            self,
            "task_description",
            _require_string(
                self.task_description,
                "observation task_description",
                allow_empty=True,
            ),
        )
        object.__setattr__(
            self,
            "steps_elapsed",
            _require_int(self.steps_elapsed, "steps_elapsed", minimum=0),
        )
        _derive_task_outcome(
            score=self.score,
            task_success=self.task_success,
            task_failure=self.task_failure,
        )
        if type(self.truncated) is not bool:
            raise TypeError("truncated must be a boolean")
        if self.task_outcome is not TaskOutcome.ACTIVE and self.truncated:
            raise ValueError("a task terminal observation cannot also be truncated")
        object.__setattr__(
            self,
            "state_token",
            _require_string(self.state_token, "state_token", allow_empty=True),
        )

    @property
    def valid_actions(self) -> tuple[ActionChoice, ...]:
        return self.action_view.choices

    @property
    def action_mask(self) -> tuple[bool, ...]:
        return self.action_view.mask

    @property
    def task_outcome(self) -> TaskOutcome:
        return _derive_task_outcome(
            score=self.score,
            task_success=self.task_success,
            task_failure=self.task_failure,
        )

    @property
    def terminated(self) -> bool:
        return self.task_outcome is not TaskOutcome.ACTIVE

    @property
    def success(self) -> bool:
        return self.task_outcome is TaskOutcome.SUCCESS


def _snapshot_observation(observation: TextWorldObservation) -> TextWorldObservation:
    if not isinstance(observation, TextWorldObservation):
        raise TypeError("observation must be TextWorldObservation")
    return TextWorldObservation(
        text=observation.text,
        look=observation.look,
        inventory=observation.inventory,
        action_view=observation.action_view,
        score_raw=observation.score_raw,
        score=observation.score,
        task_description=observation.task_description,
        steps_elapsed=observation.steps_elapsed,
        task_success=observation.task_success,
        task_failure=observation.task_failure,
        truncated=observation.truncated,
        state_token=observation.state_token,
    )


class StableTextObservationEncoder:
    """A deterministic dependency-free feature hash for integration smoke tests."""

    _token_pattern = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)

    def __init__(self, observation_size: int) -> None:
        self._observation_size = _require_int(
            observation_size,
            "observation_size",
            minimum=1,
        )
        self._abi_sha256 = _encoder_sha256(self._observation_size)

    @property
    def observation_size(self) -> int:
        return self._observation_size

    @property
    def abi_sha256(self) -> str:
        return self._abi_sha256

    def encode(self, observation: TextWorldObservation) -> tuple[float, ...]:
        observation = _snapshot_observation(observation)
        features = ["bias"]
        for field_name, text in (
            ("text", observation.text),
            ("look", observation.look),
            ("inventory", observation.inventory),
            ("task", observation.task_description),
        ):
            features.extend(
                f"{field_name}:{token}"
                for token in self._token_pattern.findall(text.casefold())
            )
        features.extend(
            f"action:{choice.action_id}:{choice.command}"
            for choice in observation.valid_actions
        )
        features.extend(
            (
                f"score_raw:{observation.score_raw.hex()}",
                f"score:{observation.score.hex()}",
                f"steps:{observation.steps_elapsed}",
                f"task_success:{int(observation.task_success)}",
                f"task_failure:{int(observation.task_failure)}",
                f"outcome:{observation.task_outcome.value}",
                f"truncated:{int(observation.truncated)}",
            )
        )
        # state_token is debug/replay metadata. TextWorldExpress has no such
        # policy observation, so the fake backend must not teach from it.
        values = [0.0] * self.observation_size
        for feature in features:
            digest = hashlib.sha256(feature.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:8], "big") % self.observation_size
            sign = 1.0 if digest[8] & 1 else -1.0
            values[bucket] += sign
        norm = math.sqrt(sum(value * value for value in values))
        if norm == 0.0:
            values[0] = 1.0
            norm = 1.0
        return tuple(value / norm for value in values)


@dataclass(frozen=True)
class BoundaryFingerprint:
    """Versioned observable boundary identity used by deterministic replay."""

    episode_sha256: str
    steps_elapsed: int
    observation_sha256: str
    task_outcome: TaskOutcome
    truncated: bool
    digest: str

    def __post_init__(self) -> None:
        _require_sha256(self.episode_sha256, "episode_sha256")
        object.__setattr__(
            self,
            "steps_elapsed",
            _require_int(self.steps_elapsed, "steps_elapsed", minimum=0),
        )
        _require_sha256(self.observation_sha256, "observation_sha256")
        if not isinstance(self.task_outcome, TaskOutcome):
            raise TypeError("fingerprint task_outcome must be TaskOutcome")
        object.__setattr__(self, "task_outcome", TaskOutcome(self.task_outcome.value))
        if type(self.truncated) is not bool:
            raise TypeError("fingerprint truncated must be a boolean")
        if self.task_outcome is not TaskOutcome.ACTIVE and self.truncated:
            raise ValueError("terminal fingerprint cannot also be truncated")
        _require_sha256(self.digest, "boundary digest")
        expected_digest = _sha256(
            {
                "episode_sha256": self.episode_sha256,
                "observation_sha256": self.observation_sha256,
                "schema": _BOUNDARY_SCHEMA,
                "steps_elapsed": self.steps_elapsed,
                "task_outcome": self.task_outcome.value,
                "truncated": self.truncated,
            }
        )
        if self.digest != expected_digest:
            raise ValueError("boundary digest does not match its fingerprint fields")

    @classmethod
    def create(
        cls,
        episode_spec: TextWorldEpisodeSpec,
        observation: TextWorldObservation,
    ) -> BoundaryFingerprint:
        episode_spec = _snapshot_episode(episode_spec)
        observation = _snapshot_observation(observation)
        if observation.action_view.codec_sha256 != episode_spec.config.action_codec_sha256:
            raise ValueError("observation action codec differs from the episode ABI")
        observation_sha256 = _sha256(
            {
                "action_mask": list(observation.action_mask),
                "codec_sha256": observation.action_view.codec_sha256,
                "inventory": observation.inventory,
                "look": observation.look,
                "score": observation.score.hex(),
                "score_raw": observation.score_raw.hex(),
                "state_token": observation.state_token,
                "steps_elapsed": observation.steps_elapsed,
                "task_description": observation.task_description,
                "task_failure": observation.task_failure,
                "task_outcome": observation.task_outcome.value,
                "task_success": observation.task_success,
                "text": observation.text,
                "truncated": observation.truncated,
                "valid_actions": [
                    [choice.action_id, choice.command]
                    for choice in observation.valid_actions
                ],
            }
        )
        digest = _sha256(
            {
                "episode_sha256": episode_spec.abi_sha256,
                "observation_sha256": observation_sha256,
                "schema": _BOUNDARY_SCHEMA,
                "steps_elapsed": observation.steps_elapsed,
                "task_outcome": observation.task_outcome.value,
                "truncated": observation.truncated,
            }
        )
        return cls(
            episode_sha256=episode_spec.abi_sha256,
            steps_elapsed=observation.steps_elapsed,
            observation_sha256=observation_sha256,
            task_outcome=observation.task_outcome,
            truncated=observation.truncated,
            digest=digest,
        )

    @property
    def terminated(self) -> bool:
        return self.task_outcome is not TaskOutcome.ACTIVE

    @property
    def success(self) -> bool:
        return self.task_outcome is TaskOutcome.SUCCESS


def _snapshot_fingerprint(fingerprint: BoundaryFingerprint) -> BoundaryFingerprint:
    if not isinstance(fingerprint, BoundaryFingerprint):
        raise TypeError("boundary must be BoundaryFingerprint")
    return BoundaryFingerprint(
        episode_sha256=fingerprint.episode_sha256,
        steps_elapsed=fingerprint.steps_elapsed,
        observation_sha256=fingerprint.observation_sha256,
        task_outcome=fingerprint.task_outcome,
        truncated=fingerprint.truncated,
        digest=fingerprint.digest,
    )


@dataclass(frozen=True)
class ReplayStep:
    """One exact online transition in an immutable replay trace."""

    before: BoundaryFingerprint
    action_id: int
    command: str
    reward: float
    terminated: bool
    truncated: bool
    success: bool
    after: BoundaryFingerprint
    step_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        before = _snapshot_fingerprint(self.before)
        after = _snapshot_fingerprint(self.after)
        action_id = _require_int(self.action_id, "replay action_id", minimum=0)
        command = _require_string(
            self.command,
            "replay command",
            canonical_whitespace=True,
        )
        reward = _require_float(self.reward, "replay reward")
        flags = (self.terminated, self.truncated, self.success)
        if any(type(value) is not bool for value in flags):
            raise TypeError("replay outcome flags must be booleans")
        if self.terminated and self.truncated:
            raise ValueError("replay step cannot terminate and truncate together")
        if self.success and not self.terminated:
            raise ValueError("successful replay step must terminate")
        if before.terminated or before.truncated:
            raise ValueError("a replay step cannot start after an episode boundary")
        if after.episode_sha256 != before.episode_sha256:
            raise ValueError("a replay step cannot cross episode identities")
        if after.steps_elapsed != before.steps_elapsed + 1:
            raise ValueError("replay step boundary indices must be consecutive")
        if self.terminated != after.terminated:
            raise ValueError("replay terminated flag differs from its after boundary")
        if self.truncated != after.truncated:
            raise ValueError("replay truncated flag differs from its after boundary")
        if self.success != after.success:
            raise ValueError("replay success flag differs from its after boundary")
        object.__setattr__(self, "before", before)
        object.__setattr__(self, "after", after)
        object.__setattr__(self, "action_id", action_id)
        object.__setattr__(self, "command", command)
        object.__setattr__(self, "reward", reward)
        object.__setattr__(
            self,
            "step_sha256",
            _sha256(
                {
                    "action_id": action_id,
                    "after": after.digest,
                    "before": before.digest,
                    "command": command,
                    "reward": reward.hex(),
                    "schema": _REPLAY_STEP_SCHEMA,
                    "success": self.success,
                    "terminated": self.terminated,
                    "truncated": self.truncated,
                }
            ),
        )

    @property
    def digest(self) -> str:
        return self.step_sha256


def _snapshot_replay_step(step: ReplayStep) -> ReplayStep:
    if not isinstance(step, ReplayStep):
        raise TypeError("trace rows must be ReplayStep records")
    return ReplayStep(
        before=step.before,
        action_id=step.action_id,
        command=step.command,
        reward=step.reward,
        terminated=step.terminated,
        truncated=step.truncated,
        success=step.success,
        after=step.after,
    )


@dataclass(frozen=True)
class ReplayCursor:
    """A self-validating immutable reset/replay recipe."""

    episode_spec: TextWorldEpisodeSpec
    initial: BoundaryFingerprint
    steps: tuple[ReplayStep, ...]
    trace_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        episode_spec = _snapshot_episode(self.episode_spec)
        initial = _snapshot_fingerprint(self.initial)
        if isinstance(self.steps, (str, bytes)):
            raise TypeError("replay steps must be an iterable")
        try:
            steps = tuple(_snapshot_replay_step(step) for step in self.steps)
        except TypeError as error:
            raise TypeError("replay steps must be an iterable") from error
        if initial.episode_sha256 != episode_spec.abi_sha256:
            raise ValueError("initial boundary differs from the episode identity")
        if initial.steps_elapsed != 0:
            raise ValueError("initial replay boundary must have step index zero")
        if initial.terminated or initial.truncated:
            raise ValueError("initial replay boundary must be active")
        previous = initial
        chain = _sha256(
            {
                "episode_sha256": episode_spec.abi_sha256,
                "initial": initial.digest,
                "schema": _REPLAY_CURSOR_SCHEMA,
            }
        )
        for step in steps:
            if step.before != previous:
                raise ValueError("replay trace boundaries are not adjacent")
            chain = _sha256(
                {
                    "previous": chain,
                    "schema": _REPLAY_CURSOR_SCHEMA,
                    "step": step.step_sha256,
                }
            )
            previous = step.after
        object.__setattr__(self, "episode_spec", episode_spec)
        object.__setattr__(self, "initial", initial)
        object.__setattr__(self, "steps", steps)
        object.__setattr__(self, "trace_sha256", chain)

    @property
    def boundary(self) -> BoundaryFingerprint:
        return self.steps[-1].after if self.steps else self.initial

    @property
    def digest(self) -> str:
        return self.trace_sha256


def _snapshot_cursor(cursor: ReplayCursor) -> ReplayCursor:
    if not isinstance(cursor, ReplayCursor):
        raise TypeError("cursor must be ReplayCursor")
    return ReplayCursor(
        episode_spec=cursor.episode_spec,
        initial=cursor.initial,
        steps=cursor.steps,
    )


class ReplayMismatchError(RuntimeError):
    """Raised when reset/replay does not reconstruct an exact online boundary."""


class OpaqueBackendDoneError(RuntimeError):
    """Raised when a backend reports done without a classified task outcome."""


class EnvironmentFaultedError(RuntimeError):
    """Raised after a backend call failed to commit a validated boundary."""


class ReplayableTextWorldEnvironment(RolloutEnvironment[TextWorldObservation]):
    """Online dense-action wrapper with an exact reset/replay trace."""

    def __init__(
        self,
        backend: TextWorldBackend,
        episode_spec: TextWorldEpisodeSpec,
        *,
        codec: FixedVocabularyActionCodec | None = None,
        encoder: StableTextObservationEncoder | None = None,
    ) -> None:
        if not isinstance(backend, TextWorldBackend):
            raise TypeError("backend must implement TextWorldBackend")
        self._backend = backend
        self._episode_spec = _snapshot_episode(episode_spec)
        config = self._episode_spec.config
        supplied_codec = codec or FixedVocabularyActionCodec(config.action_vocabulary)
        if not isinstance(supplied_codec, FixedVocabularyActionCodec):
            raise TypeError("codec must be FixedVocabularyActionCodec")
        self._codec = FixedVocabularyActionCodec(supplied_codec.vocabulary)
        if self._codec.abi_sha256 != config.action_codec_sha256:
            raise ValueError("codec ABI differs from the episode configuration")
        supplied_encoder = encoder or StableTextObservationEncoder(
            config.observation_size
        )
        if not isinstance(supplied_encoder, StableTextObservationEncoder):
            raise TypeError("encoder must be StableTextObservationEncoder")
        self._encoder = StableTextObservationEncoder(
            supplied_encoder.observation_size
        )
        if self._encoder.abi_sha256 != config.encoder_sha256:
            raise ValueError("encoder ABI differs from the episode configuration")
        self._spec = EnvironmentSpec(
            observation_size=self._encoder.observation_size,
            action_size=self._codec.action_size,
        )
        self._closed = False
        self._faulted = False
        self._reset_called = False
        self._observation: TextWorldObservation | None = None
        self._initial: BoundaryFingerprint | None = None
        self._trace: list[ReplayStep] = []

    @property
    def spec(self) -> EnvironmentSpec:
        return validate_environment_spec(self._spec)

    @property
    def episode_spec(self) -> TextWorldEpisodeSpec:
        return _snapshot_episode(self._episode_spec)

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def faulted(self) -> bool:
        return self._faulted

    @property
    def observation(self) -> TextWorldObservation:
        return _snapshot_observation(self._require_observation())

    @property
    def action_view(self) -> ActionView:
        return _snapshot_action_view(self._require_observation().action_view)

    @property
    def available_actions(self) -> tuple[ActionChoice, ...]:
        return self.action_view.choices

    @property
    def action_mask(self) -> tuple[bool, ...]:
        return self.action_view.mask

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("TextWorld environment is closed")

    def _require_healthy(self) -> None:
        self._require_open()
        if self._faulted:
            raise EnvironmentFaultedError(
                "TextWorld environment is faulted after a backend error"
            )

    def _require_observation(self) -> TextWorldObservation:
        self._require_healthy()
        if self._observation is None:
            raise RuntimeError("TextWorld environment has not been reset")
        return self._observation

    def reset(self) -> TextWorldObservation:
        self._require_healthy()
        if self._reset_called:
            raise RuntimeError("online TextWorld environment may only be reset once")
        return self._reset_backend()

    def _reset_backend(self) -> TextWorldObservation:
        self._require_healthy()
        try:
            return self._reset_backend_call()
        except Exception:
            self._faulted = True
            raise

    def _reset_backend_call(self) -> TextWorldObservation:
        boundary = _snapshot_backend_boundary(
            self._backend.reset(_snapshot_episode(self._episode_spec))
        )
        if boundary.task_outcome is not TaskOutcome.ACTIVE:
            raise ValueError("backend reset must return an active task boundary")
        view = self._codec.bind(boundary.valid_actions)
        if not view.choices:
            raise ValueError("an active reset boundary must expose a valid action")
        observation = TextWorldObservation(
            text=boundary.text,
            look=boundary.look,
            inventory=boundary.inventory,
            action_view=view,
            score_raw=boundary.score_raw,
            score=boundary.score,
            task_description=boundary.task_description,
            steps_elapsed=0,
            task_success=boundary.task_success,
            task_failure=boundary.task_failure,
            truncated=False,
            state_token=boundary.state_token,
        )
        initial = BoundaryFingerprint.create(self._episode_spec, observation)
        self._reset_called = True
        self._observation = observation
        self._initial = initial
        self._trace = []
        return _snapshot_observation(observation)

    def encode_observation(
        self,
        observation: TextWorldObservation,
    ) -> tuple[float, ...]:
        self._require_healthy()
        observation = _snapshot_observation(observation)
        if observation.action_view.codec_sha256 != self._codec.abi_sha256:
            raise ValueError("observation belongs to another action codec")
        return self._encoder.encode(observation)

    def step(self, action: int) -> EnvironmentStep[TextWorldObservation]:
        current = self._require_observation()
        if current.terminated or current.truncated:
            raise RuntimeError("cannot step after an episode boundary")
        action = validate_environment_action(action, self._spec)
        command = self._codec.decode(action, current.action_view)
        before = BoundaryFingerprint.create(self._episode_spec, current)
        try:
            return self._step_backend_call(
                current=current,
                action=action,
                command=command,
                before=before,
            )
        except Exception:
            # Once the backend call starts it may have advanced even if the
            # returned boundary is malformed or validation raises. Keeping the
            # old wrapper cursor usable would silently fork wrapper/backend
            # state, so this environment is permanently poisoned.
            self._faulted = True
            raise

    def _step_backend_call(
        self,
        *,
        current: TextWorldObservation,
        action: int,
        command: str,
        before: BoundaryFingerprint,
    ) -> EnvironmentStep[TextWorldObservation]:
        backend_result = _snapshot_backend_transition(self._backend.step(command))

        task_outcome = backend_result.boundary.task_outcome
        if task_outcome is TaskOutcome.ACTIVE:
            if backend_result.done:
                raise OpaqueBackendDoneError(
                    "backend reported done without a task terminal classification"
                )
        elif not backend_result.done:
            raise RuntimeError("backend task terminal must also report done")

        steps_elapsed = current.steps_elapsed + 1
        terminated = task_outcome is not TaskOutcome.ACTIVE
        truncated = (
            not terminated
            and steps_elapsed >= self._episode_spec.config.project_max_steps
        )
        success = task_outcome is TaskOutcome.SUCCESS
        next_view = self._codec.bind(backend_result.boundary.valid_actions)
        if not terminated and not truncated and not next_view.choices:
            raise ValueError("an active boundary must expose a valid action")
        next_observation = TextWorldObservation(
            text=backend_result.boundary.text,
            look=backend_result.boundary.look,
            inventory=backend_result.boundary.inventory,
            action_view=next_view,
            score_raw=backend_result.boundary.score_raw,
            score=backend_result.boundary.score,
            task_description=backend_result.boundary.task_description,
            steps_elapsed=steps_elapsed,
            task_success=backend_result.boundary.task_success,
            task_failure=backend_result.boundary.task_failure,
            truncated=truncated,
            state_token=backend_result.boundary.state_token,
        )
        reward = _require_float(
            next_observation.score - current.score,
            "backend score delta",
        )
        result = EnvironmentStep(
            observation=next_observation,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            success=success,
        )
        after = BoundaryFingerprint.create(self._episode_spec, next_observation)
        replay_step = ReplayStep(
            before=before,
            action_id=action,
            command=command,
            reward=reward,
            terminated=result.terminated,
            truncated=result.truncated,
            success=result.success,
            after=after,
        )
        self._observation = next_observation
        self._trace.append(replay_step)
        return EnvironmentStep(
            observation=_snapshot_observation(next_observation),
            reward=result.reward,
            terminated=result.terminated,
            truncated=result.truncated,
            success=result.success,
        )

    def replay_cursor(self) -> ReplayCursor:
        self._require_observation()
        if self._initial is None:
            raise AssertionError("reset environment omitted its initial boundary")
        return ReplayCursor(
            episode_spec=self._episode_spec,
            initial=self._initial,
            steps=tuple(self._trace),
        )

    def restore(self, cursor: ReplayCursor) -> TextWorldObservation:
        """Destructively reset this instance and exactly replay one cursor."""

        self._require_healthy()
        cursor = _snapshot_cursor(cursor)
        if cursor.episode_spec != self._episode_spec:
            raise ValueError("replay cursor belongs to another episode")
        try:
            return self._restore_backend_call(cursor)
        except Exception:
            # Reset/replay is destructive. Any mismatch after reset begins
            # leaves the scratch backend at an untrusted boundary and it must
            # not be reused for another candidate.
            self._faulted = True
            raise

    def _restore_backend_call(self, cursor: ReplayCursor) -> TextWorldObservation:
        self._reset_backend_call()
        if self.replay_cursor().initial != cursor.initial:
            raise ReplayMismatchError("backend reset boundary differs from replay cursor")

        for index, expected in enumerate(cursor.steps):
            current_cursor = self.replay_cursor()
            if current_cursor.boundary != expected.before:
                raise ReplayMismatchError(
                    f"replay boundary mismatch before trace step {index}"
                )
            try:
                command = self._codec.decode(
                    expected.action_id,
                    self._require_observation().action_view,
                )
            except (TypeError, ValueError) as error:
                raise ReplayMismatchError(
                    f"recorded action is invalid at trace step {index}"
                ) from error
            if command != expected.command:
                raise ReplayMismatchError(
                    f"recorded command changed at trace step {index}"
                )
            try:
                self.step(expected.action_id)
            except Exception as error:
                raise ReplayMismatchError(
                    f"backend failed while replaying trace step {index}"
                ) from error
            actual = self._trace[-1]
            if actual != expected:
                raise ReplayMismatchError(
                    f"backend result differs at trace step {index}"
                )

        if self.replay_cursor() != cursor:
            raise ReplayMismatchError("reconstructed replay cursor differs from snapshot")
        return self.observation

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._backend.close()


def _snapshot_student_proposal(proposal: StudentProposal) -> StudentProposal:
    if not isinstance(proposal, StudentProposal):
        raise TypeError("Student model must return StudentProposal")
    return StudentProposal(
        action=proposal.action,
        log_prob=proposal.log_prob,
        value=proposal.value,
    )


def _snapshot_teacher_proposal(
    proposal: TeacherProposal | None,
) -> TeacherProposal | None:
    if proposal is None:
        return None
    if not isinstance(proposal, TeacherProposal):
        raise TypeError("Teacher must return TeacherProposal")
    return TeacherProposal(
        correction_actions=proposal.correction_actions,
        recovery_actions=proposal.recovery_actions,
    )


class ReplayCandidateEvaluator:
    """Exact S/T/F evaluator backed by one reusable reset/replay environment."""

    def __init__(
        self,
        scratch_environment: ReplayableTextWorldEnvironment,
    ) -> None:
        if not isinstance(scratch_environment, ReplayableTextWorldEnvironment):
            raise TypeError("scratch environment must be replayable TextWorld")
        self._scratch = scratch_environment
        self._closed = False

    @property
    def closed(self) -> bool:
        return self._closed

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("replay candidate evaluator is closed")

    def validate_environment(
        self,
        environment: RolloutEnvironment[TextWorldObservation],
    ) -> None:
        self._require_open()
        if not isinstance(environment, ReplayableTextWorldEnvironment):
            raise TypeError(
                "replay evaluation requires ReplayableTextWorldEnvironment"
            )
        if environment is self._scratch:
            raise ValueError("scratch environment must be independent from online")
        if environment.closed:
            raise RuntimeError("online TextWorld environment is closed")
        if self._scratch.closed:
            raise RuntimeError("scratch TextWorld environment is closed")
        if environment.episode_spec != self._scratch.episode_spec:
            raise ValueError("online and scratch environments use different episodes")
        if environment.spec != self._scratch.spec:
            raise ValueError("online and scratch environments use different specs")

    def build_candidates(
        self,
        environment: RolloutEnvironment[TextWorldObservation],
        student_proposal: StudentProposal,
        student_model: StudentModel,
        config: RolloutConfig,
        teacher_proposal: TeacherProposal | None,
    ) -> tuple[OptionCandidate, ...]:
        self.validate_environment(environment)
        online = environment
        student_proposal = _snapshot_student_proposal(student_proposal)
        teacher_proposal = _snapshot_teacher_proposal(teacher_proposal)
        gamma = _require_float(config.gamma, "rollout gamma")
        if not 0.0 < gamma <= 1.0:
            raise ValueError("rollout gamma must be in (0, 1]")
        query_unit_cost = _require_float(
            config.teacher_query_cost,
            "teacher query cost",
        )
        execution_unit_cost = _require_float(
            config.teacher_execution_cost,
            "teacher execution cost",
        )
        if query_unit_cost < 0.0 or execution_unit_cost < 0.0:
            raise ValueError("Teacher costs must be non-negative")
        validate_environment_action(student_proposal.action, online.spec)
        if teacher_proposal is not None:
            for action in (
                *teacher_proposal.correction_actions,
                *teacher_proposal.recovery_actions,
            ):
                validate_environment_action(action, online.spec)

        cursor = online.replay_cursor()
        if cursor.boundary.terminated or cursor.boundary.truncated:
            raise ValueError("cannot build candidates after an episode boundary")
        query_cost = query_unit_cost if teacher_proposal is not None else 0.0
        try:
            candidates = [
                self._preview(
                    cursor,
                    kind=OptionKind.STUDENT,
                    actions=(student_proposal.action,),
                    student_model=student_model,
                    gamma=gamma,
                    query_cost=query_cost,
                    execution_unit_cost=execution_unit_cost,
                )
            ]
            if teacher_proposal is not None:
                candidates.extend(
                    (
                        self._preview(
                            cursor,
                            kind=OptionKind.TEACHER_CORRECTION,
                            actions=teacher_proposal.correction_actions,
                            student_model=student_model,
                            gamma=gamma,
                            query_cost=query_cost,
                            execution_unit_cost=execution_unit_cost,
                        ),
                        self._preview(
                            cursor,
                            kind=OptionKind.TEACHER_RECOVERY,
                            actions=teacher_proposal.recovery_actions,
                            student_model=student_model,
                            gamma=gamma,
                            query_cost=query_cost,
                            execution_unit_cost=execution_unit_cost,
                        ),
                    )
                )
            return tuple(candidates)
        finally:
            if online.replay_cursor() != cursor:
                raise RuntimeError(
                    "candidate evaluation mutated the online TextWorld environment"
                )

    def _preview(
        self,
        cursor: ReplayCursor,
        *,
        kind: OptionKind,
        actions: tuple[int, ...],
        student_model: StudentModel,
        gamma: float,
        query_cost: float,
        execution_unit_cost: float,
    ) -> OptionCandidate:
        self._scratch.restore(cursor)
        if self._scratch.replay_cursor() != cursor:
            raise ReplayMismatchError("scratch restore did not reach the online cursor")
        discounted_reward = 0.0
        discounted_execution_cost = 0.0
        preview_steps = 0
        terminated = False
        truncated = False
        last_result: EnvironmentStep[TextWorldObservation] | None = None
        for index, action in enumerate(actions):
            action = validate_environment_action(action, self._scratch.spec)
            result = self._scratch.step(action)
            discount = gamma**index
            discounted_reward += discount * result.reward
            if kind is not OptionKind.STUDENT:
                discounted_execution_cost += discount * execution_unit_cost
            preview_steps += 1
            terminated = result.terminated
            truncated = result.truncated
            last_result = result
            if terminated or truncated:
                break
        if last_result is None:
            raise AssertionError("candidate preview executed no actions")
        if not terminated and not truncated:
            encoded = self._scratch.encode_observation(last_result.observation)
            bootstrap = _require_float(
                student_model.value(encoded),
                "Student value",
            )
            discounted_reward += (gamma**preview_steps) * bootstrap
        estimated_task_value = _require_float(
            discounted_reward,
            "candidate estimated task value",
        )
        execution_cost = _require_float(
            discounted_execution_cost,
            "candidate execution cost",
        )
        return OptionCandidate(
            kind=kind,
            actions=actions,
            preview_steps=preview_steps,
            estimated_task_value=estimated_task_value,
            query_cost=query_cost,
            execution_cost=execution_cost,
            terminated=terminated,
            truncated=truncated,
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._scratch.close()


BackendFactory = Callable[[], TextWorldBackend]
TeacherFactory = Callable[
    [TextWorldEpisodeSpec],
    TeacherPolicy[TextWorldObservation],
]


class TextWorldRuntimeAdapter:
    """Own one online backend and one reusable scratch backend per episode."""

    def __init__(
        self,
        config: TextWorldRuntimeConfig,
        *,
        backend_factory: BackendFactory,
        teacher_factory: TeacherFactory | None = None,
    ) -> None:
        self._config = _snapshot_config(config)
        if not callable(backend_factory):
            raise TypeError("backend_factory must be callable")
        if teacher_factory is not None and not callable(teacher_factory):
            raise TypeError("teacher_factory must be callable")
        self._backend_factory = backend_factory
        self._teacher_factory = teacher_factory
        self._codec = FixedVocabularyActionCodec(self._config.action_vocabulary)
        self._encoder = StableTextObservationEncoder(self._config.observation_size)
        self._spec = EnvironmentSpec(
            observation_size=self._encoder.observation_size,
            action_size=self._codec.action_size,
        )

    @property
    def config(self) -> TextWorldRuntimeConfig:
        return _snapshot_config(self._config)

    @property
    def spec(self) -> EnvironmentSpec:
        return validate_environment_spec(self._spec)

    def _new_backend(self) -> TextWorldBackend:
        backend = self._backend_factory()
        if not isinstance(backend, TextWorldBackend):
            close = getattr(backend, "close", None)
            try:
                if callable(close):
                    close()
            finally:
                raise TypeError("backend_factory must return TextWorldBackend")
        return backend

    @contextmanager
    def open_episode(
        self,
        *,
        seed: int,
        require_teacher: bool,
    ) -> Iterator[EpisodeComponents[TextWorldObservation]]:
        seed = _require_int(seed, "episode seed")
        if type(require_teacher) is not bool:
            raise TypeError("require_teacher must be a boolean")
        if require_teacher and self._teacher_factory is None:
            raise ValueError("a Teacher factory is required when probing is enabled")

        episode_spec = TextWorldEpisodeSpec(config=self._config, seed=seed)
        online_backend: TextWorldBackend | None = None
        scratch_backend: TextWorldBackend | None = None
        online_environment: ReplayableTextWorldEnvironment | None = None
        scratch_environment: ReplayableTextWorldEnvironment | None = None
        evaluator: ReplayCandidateEvaluator | None = None
        try:
            online_backend = self._new_backend()
            online_environment = ReplayableTextWorldEnvironment(
                online_backend,
                episode_spec,
                codec=self._codec,
                encoder=self._encoder,
            )
            scratch_backend = self._new_backend()
            if scratch_backend is online_backend:
                raise ValueError("backend_factory returned the online backend twice")
            scratch_environment = ReplayableTextWorldEnvironment(
                scratch_backend,
                episode_spec,
                codec=self._codec,
                encoder=self._encoder,
            )
            evaluator = ReplayCandidateEvaluator(scratch_environment)
            teacher = None
            if require_teacher:
                if self._teacher_factory is None:
                    raise AssertionError("Teacher factory disappeared after validation")
                teacher = self._teacher_factory(_snapshot_episode(episode_spec))
                if teacher is None or not callable(getattr(teacher, "propose", None)):
                    raise TypeError("teacher_factory must return a Teacher policy")
            yield EpisodeComponents(
                environment=online_environment,
                teacher=teacher,
                candidate_evaluator=evaluator,
            )
        finally:
            try:
                if evaluator is not None:
                    evaluator.close()
                elif scratch_environment is not None:
                    scratch_environment.close()
                elif (
                    scratch_backend is not None
                    and scratch_backend is not online_backend
                ):
                    scratch_backend.close()
            finally:
                if online_environment is not None:
                    online_environment.close()
                elif online_backend is not None:
                    online_backend.close()
