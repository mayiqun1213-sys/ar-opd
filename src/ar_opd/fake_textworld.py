"""Dependency-free TextWorld-like backend used by adapter integration tests."""

from __future__ import annotations

import math
from dataclasses import dataclass

from ar_opd.core import TeacherProposal
from ar_opd.textworld_runtime import (
    BackendBoundary,
    BackendTransition,
    TaskOutcome,
    TextWorldEpisodeSpec,
    TextWorldObservation,
)


FAKE_TEXTWORLD_BACKEND_IDENTITY = "ar-opd.fake-textworld-v1"
JAMMED_QUEST_ACTION_VOCABULARY = (
    "look",
    "go east",
    "repair cart",
    "wait",
    "go west",
)


def _validate_name(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


@dataclass(frozen=True)
class FakeTransitionSpec:
    """One immutable command edge in a fake game graph."""

    command: str
    next_state: str
    score_delta: float = 0.0
    done: bool = False
    task_outcome: TaskOutcome = TaskOutcome.ACTIVE

    def __post_init__(self) -> None:
        _validate_name(self.command, field="transition command")
        _validate_name(self.next_state, field="transition target")
        if isinstance(self.score_delta, bool) or not isinstance(
            self.score_delta, int | float
        ):
            raise TypeError("transition score delta must be a real number")
        if not math.isfinite(self.score_delta):
            raise ValueError("transition score delta must be finite")
        if type(self.done) is not bool:
            raise TypeError("transition done flag must be a boolean")
        if not isinstance(self.task_outcome, TaskOutcome):
            raise TypeError("transition outcome must be a TaskOutcome")
        if self.done != (self.task_outcome is not TaskOutcome.ACTIVE):
            raise ValueError("transition done must agree with its declared task outcome")

    @property
    def outcome(self) -> TaskOutcome:
        """Concise alias useful in fault-injection assertions."""

        return self.task_outcome


@dataclass(frozen=True)
class FakeStateSpec:
    """One immutable state with a deliberately ordered dynamic action menu."""

    name: str
    text: str
    look: str
    inventory: str
    valid_actions: tuple[str, ...]
    transitions: tuple[FakeTransitionSpec, ...]

    def __post_init__(self) -> None:
        _validate_name(self.name, field="state name")
        _validate_name(self.text, field="state text")
        _validate_name(self.look, field="state look")
        _validate_name(self.inventory, field="state inventory")
        object.__setattr__(self, "valid_actions", tuple(self.valid_actions))
        object.__setattr__(self, "transitions", tuple(self.transitions))
        if any(not isinstance(command, str) or not command for command in self.valid_actions):
            raise ValueError("state actions must be non-empty strings")
        if len(set(self.valid_actions)) != len(self.valid_actions):
            raise ValueError("state actions must be unique")
        if any(not isinstance(edge, FakeTransitionSpec) for edge in self.transitions):
            raise TypeError("state transitions must be FakeTransitionSpec records")
        commands = tuple(edge.command for edge in self.transitions)
        if len(set(commands)) != len(commands):
            raise ValueError("state transition commands must be unique")
        if set(commands) != set(self.valid_actions):
            raise ValueError("state transitions must exactly cover its valid actions")

    def transition_for(self, command: str) -> FakeTransitionSpec | None:
        return next((edge for edge in self.transitions if edge.command == command), None)


@dataclass(frozen=True)
class FakeTextWorldGame:
    """An immutable, validated graph shared safely by independent backends."""

    name: str
    task_description: str
    initial_state: str
    states: tuple[FakeStateSpec, ...]

    def __post_init__(self) -> None:
        _validate_name(self.name, field="game name")
        _validate_name(self.task_description, field="game task description")
        _validate_name(self.initial_state, field="initial state")
        object.__setattr__(self, "states", tuple(self.states))
        if not self.states:
            raise ValueError("a fake game must contain at least one state")
        if any(not isinstance(state, FakeStateSpec) for state in self.states):
            raise TypeError("game states must be FakeStateSpec records")
        names = tuple(state.name for state in self.states)
        if len(set(names)) != len(names):
            raise ValueError("game state names must be unique")
        if self.initial_state not in names:
            raise ValueError("game initial state is missing")
        targets = {
            edge.next_state
            for state in self.states
            for edge in state.transitions
        }
        missing = targets.difference(names)
        if missing:
            raise ValueError(f"game transitions reference missing states: {sorted(missing)!r}")

    def state(self, name: str) -> FakeStateSpec:
        for state in self.states:
            if state.name == name:
                return state
        raise KeyError(name)


class FakeTextWorldBackend:
    """Strict deterministic backend with observable lifecycle and call history."""

    def __init__(self, game: FakeTextWorldGame) -> None:
        if not isinstance(game, FakeTextWorldGame):
            raise TypeError("fake backend requires a FakeTextWorldGame")
        self.game = game
        self.reset_calls: list[TextWorldEpisodeSpec] = []
        self.step_attempts: list[str] = []
        self.step_calls: list[str] = []
        self.close_count = 0
        self._closed = False
        self._episode: TextWorldEpisodeSpec | None = None
        self._state_name: str | None = None
        self._score = 0.0
        self._done = False
        self._task_outcome = TaskOutcome.ACTIVE

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def episode_spec(self) -> TextWorldEpisodeSpec | None:
        return self._episode

    @property
    def state_name(self) -> str | None:
        return self._state_name

    @property
    def score(self) -> float:
        return self._score

    @property
    def done(self) -> bool:
        return self._done

    @property
    def task_outcome(self) -> TaskOutcome:
        return self._task_outcome

    @property
    def boundary(self) -> BackendBoundary:
        self._ensure_ready()
        if self._state_name is None:
            raise AssertionError("ready backend has no state")
        return self._make_boundary(
            self.game.state(self._state_name),
            self._score,
            self._task_outcome,
        )

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("fake backend is closed")

    def _ensure_ready(self) -> None:
        self._ensure_open()
        if self._episode is None or self._state_name is None:
            raise RuntimeError("fake backend must be reset before use")

    def _make_boundary(
        self,
        state: FakeStateSpec,
        score: float,
        task_outcome: TaskOutcome,
    ) -> BackendBoundary:
        return BackendBoundary(
            text=state.text,
            look=state.look,
            inventory=state.inventory,
            valid_actions=state.valid_actions,
            score_raw=score,
            score=score,
            task_description=self.game.task_description,
            task_success=task_outcome is TaskOutcome.SUCCESS,
            task_failure=task_outcome is TaskOutcome.FAILURE,
            state_token=state.name,
        )

    def reset(self, episode: TextWorldEpisodeSpec) -> BackendBoundary:
        self._ensure_open()
        if not isinstance(episode, TextWorldEpisodeSpec):
            raise TypeError("fake backend reset requires TextWorldEpisodeSpec")
        if episode.config.game_name != self.game.name:
            raise ValueError("episode game name does not match fake game")
        if episode.config.backend_identity != FAKE_TEXTWORLD_BACKEND_IDENTITY:
            raise ValueError("episode backend identity does not match fake backend")
        state = self.game.state(self.game.initial_state)
        boundary = self._make_boundary(state, 0.0, TaskOutcome.ACTIVE)
        self._episode = episode
        self._state_name = state.name
        self._score = 0.0
        self._done = False
        self._task_outcome = TaskOutcome.ACTIVE
        self.reset_calls.append(episode)
        return boundary

    def step(self, command: str) -> BackendTransition:
        self._ensure_ready()
        if not isinstance(command, str) or not command:
            raise TypeError("fake backend command must be a non-empty string")
        self.step_attempts.append(command)
        if self._done:
            raise RuntimeError("fake backend cannot step after done")
        if self._state_name is None:
            raise AssertionError("ready backend has no state")
        current = self.game.state(self._state_name)
        edge = current.transition_for(command)
        if edge is None:
            raise ValueError("command is not valid at the current fake state")

        next_state = self.game.state(edge.next_state)
        next_score = self._score + float(edge.score_delta)
        if not math.isfinite(next_score):
            raise ValueError("fake backend cumulative score became non-finite")
        boundary = self._make_boundary(next_state, next_score, edge.task_outcome)
        result = BackendTransition(
            boundary=boundary,
            done=edge.done,
        )

        self._state_name = next_state.name
        self._score = next_score
        self._done = edge.done
        self._task_outcome = boundary.task_outcome
        self.step_calls.append(command)
        return result

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self.close_count += 1


class FakeTextWorldBackendFactory:
    """Create independent fake backends and retain them for lifecycle assertions."""

    def __init__(self, game: FakeTextWorldGame | None = None) -> None:
        game = JAMMED_QUEST_GAME if game is None else game
        if not isinstance(game, FakeTextWorldGame):
            raise TypeError("fake backend factory requires a FakeTextWorldGame")
        self.game = game
        self.instances: list[FakeTextWorldBackend] = []

    @property
    def call_count(self) -> int:
        return len(self.instances)

    def __call__(self) -> FakeTextWorldBackend:
        backend = FakeTextWorldBackend(self.game)
        self.instances.append(backend)
        return backend


def _edge(
    command: str,
    next_state: str,
    score_delta: float = 0.0,
    *,
    done: bool = False,
    task_outcome: TaskOutcome = TaskOutcome.ACTIVE,
) -> FakeTransitionSpec:
    return FakeTransitionSpec(
        command=command,
        next_state=next_state,
        score_delta=score_delta,
        done=done,
        task_outcome=task_outcome,
    )


JAMMED_QUEST_GAME = FakeTextWorldGame(
    name="jammed-quest",
    task_description="Deliver the cart through the eastern passage.",
    initial_state="start",
    states=(
        FakeStateSpec(
            name="start",
            text="A delivery cart waits beside the eastern passage.",
            look="The depot is east; the cart and a repair kit are here.",
            inventory="You are carrying nothing.",
            valid_actions=("wait", "go east", "look"),
            transitions=(
                _edge("wait", "start", -0.05),
                _edge("go east", "jammed", 0.10),
                _edge("look", "start"),
            ),
        ),
        FakeStateSpec(
            name="jammed",
            text="The cart is jammed in the passage and blocks progress east.",
            look="The damaged axle must be repaired before the cart can pass.",
            inventory="You have access to the cart's repair kit.",
            valid_actions=("go west", "repair cart", "go east", "look"),
            transitions=(
                _edge("go west", "start", -0.02),
                _edge("repair cart", "repaired", -0.02),
                _edge("go east", "jammed", -0.25),
                _edge("look", "jammed"),
            ),
        ),
        FakeStateSpec(
            name="repaired",
            text="The repaired cart can now roll through the eastern passage.",
            look="The route east to the depot is clear.",
            inventory="You are carrying nothing.",
            valid_actions=("look", "go west", "go east", "wait"),
            transitions=(
                _edge("look", "repaired"),
                _edge("go west", "start", -0.02),
                _edge(
                    "go east",
                    "goal",
                    1.0,
                    done=True,
                    task_outcome=TaskOutcome.SUCCESS,
                ),
                _edge("wait", "repaired", -0.05),
            ),
        ),
        FakeStateSpec(
            name="goal",
            text="The delivery reaches the eastern depot.",
            look="The repaired cart is safely parked at the depot.",
            inventory="You are carrying nothing.",
            valid_actions=(),
            transitions=(),
        ),
    ),
)


class JammedQuestOracleTeacher:
    """State-aware oracle using global vocabulary IDs, never menu positions."""

    def __init__(self, episode_spec: TextWorldEpisodeSpec) -> None:
        if not isinstance(episode_spec, TextWorldEpisodeSpec):
            raise TypeError("jammed Teacher requires TextWorldEpisodeSpec")
        if episode_spec.config.game_name != JAMMED_QUEST_GAME.name:
            raise ValueError("jammed Teacher received another game")
        if (
            episode_spec.config.action_vocabulary
            != JAMMED_QUEST_ACTION_VOCABULARY
        ):
            raise ValueError("jammed Teacher requires the standard action vocabulary")
        self.episode_spec = episode_spec
        self.call_count = 0

    def propose(
        self,
        observation: TextWorldObservation,
        recovery_horizon: int,
    ) -> TeacherProposal:
        if not isinstance(observation, TextWorldObservation):
            raise TypeError("jammed Teacher requires TextWorldObservation")
        if isinstance(recovery_horizon, bool) or not isinstance(recovery_horizon, int):
            raise TypeError("recovery horizon must be an integer")
        if recovery_horizon < 1:
            raise ValueError("recovery horizon must be positive")

        go_east = JAMMED_QUEST_ACTION_VOCABULARY.index("go east")
        repair = JAMMED_QUEST_ACTION_VOCABULARY.index("repair cart")
        valid_ids = set(observation.action_view.action_ids)
        if repair in valid_ids:
            recovery = (repair, go_east)
        elif go_east in valid_ids:
            # S is already the best one-step choice here; a costly Teacher
            # prefix should not displace it before an actual jam.
            recovery = (go_east,)
        else:
            raise RuntimeError("jammed Teacher cannot act from this observation")
        self.call_count += 1
        return TeacherProposal(
            correction_actions=(recovery[0],),
            recovery_actions=recovery[:recovery_horizon],
        )


class JammedQuestTeacherFactory:
    """Observable factory that is safe to pass to TextWorldRuntimeAdapter."""

    def __init__(self) -> None:
        self.episode_specs: list[TextWorldEpisodeSpec] = []
        self.instances: list[JammedQuestOracleTeacher] = []

    @property
    def call_count(self) -> int:
        return len(self.instances)

    def __call__(self, episode_spec: TextWorldEpisodeSpec) -> JammedQuestOracleTeacher:
        teacher = JammedQuestOracleTeacher(episode_spec)
        self.episode_specs.append(episode_spec)
        self.instances.append(teacher)
        return teacher
