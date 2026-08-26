import math
import unittest
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from types import SimpleNamespace

import torch

from ar_opd.core import (
    ActionSource,
    OptionCandidate,
    OptionKind,
    StudentProposal,
    TeacherProposal,
)
from ar_opd.environment import (
    EnvironmentSpec,
    EnvironmentStep,
    validate_encoded_observation,
    validate_environment_action,
    validate_environment_spec,
)
from ar_opd.models import ActorCritic
from ar_opd.rollout import (
    CounterfactualCandidateEvaluator,
    RolloutCollector,
    RolloutConfig,
    collect_episodes,
)
from ar_opd.runtime import EpisodeComponents
from ar_opd.teacher import OracleTeacher
from ar_opd.toy_env import ChainAction, JammedChainConfig, JammedChainEnv
from ar_opd.toy_runtime import ToyRuntimeAdapter


class _ActionZeroStudent:
    def act(self, observation, *, deterministic=False, generator=None):
        return StudentProposal(action=0, log_prob=-0.25, value=0.0)

    def value(self, observation):
        return 0.0


class _ExplodingStudent(_ActionZeroStudent):
    def act(self, observation, *, deterministic=False, generator=None):
        raise RuntimeError("student failed")


class _InvalidValueStudent(_ActionZeroStudent):
    def __init__(self, value) -> None:
        self._value = value

    def value(self, observation):
        return self._value


class _UncheckedStudentProposal(StudentProposal):
    def __post_init__(self) -> None:
        pass


class _InvalidProposalStudent(_ActionZeroStudent):
    def act(self, observation, *, deterministic=False, generator=None):
        return _UncheckedStudentProposal(action=0, log_prob=math.nan, value=0.0)


class _OnlineOnlyEnvironment:
    """Minimal one-step environment that intentionally has no clone method."""

    spec = EnvironmentSpec(observation_size=2, action_size=2)

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed
        self.reset_count = 0
        self.step_count = 0
        self.close_count = 0
        self.closed = False
        self._done = False

    def reset(self) -> int:
        if self.closed:
            raise RuntimeError("environment is closed")
        self.reset_count += 1
        self._done = False
        return self.seed

    def encode_observation(self, observation: int) -> tuple[float, ...]:
        if self.closed:
            raise RuntimeError("environment is closed")
        return (float(observation), float(self.step_count))

    def step(self, action: int) -> EnvironmentStep[int]:
        if self.closed:
            raise RuntimeError("environment is closed")
        if self._done:
            raise RuntimeError("step called after the episode ended")
        validate_environment_action(action, self.spec)
        self.step_count += 1
        self._done = True
        return EnvironmentStep(
            observation=self.seed + 1,
            reward=1.0,
            terminated=True,
            truncated=False,
            success=True,
        )

    def close(self) -> None:
        self.close_count += 1
        self.closed = True


class _StudentOnlyEvaluator:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.teacher_proposals = []
        self.close_count = 0
        self.closed = False

    def build_candidates(
        self,
        environment,
        student_proposal,
        student_model,
        config,
        teacher_proposal,
    ):
        self.teacher_proposals.append(teacher_proposal)
        if self.fail:
            raise RuntimeError("evaluator failed")
        return (
            OptionCandidate(
                kind=OptionKind.STUDENT,
                actions=(student_proposal.action,),
                preview_steps=1,
                estimated_task_value=1.0,
                query_cost=0.0,
                execution_cost=0.0,
                terminated=True,
                truncated=False,
            ),
        )

    def close(self) -> None:
        self.close_count += 1
        self.closed = True


class _FalsyStudentOnlyEvaluator(_StudentOnlyEvaluator):
    def __bool__(self) -> bool:
        return False


class _FalsyGate:
    def __init__(self) -> None:
        self.called = False

    def __bool__(self) -> bool:
        return False

    def choose(self, candidates: tuple[OptionCandidate, ...]) -> OptionCandidate:
        self.called = True
        return candidates[0]


class _FixedCandidatesEvaluator:
    def __init__(self, candidates: tuple[OptionCandidate, ...]) -> None:
        self.candidates = candidates
        self.closed = False

    def build_candidates(
        self,
        environment,
        student_proposal,
        student_model,
        config,
        teacher_proposal,
    ) -> tuple[OptionCandidate, ...]:
        return self.candidates

    def close(self) -> None:
        self.closed = True


class _MisleadingCandidate(OptionCandidate):
    @property
    def previewed_actions(self) -> tuple[int, ...]:
        return (1,)


class _FixedTeacher:
    def propose(self, observation, recovery_horizon):
        return TeacherProposal(correction_actions=(1,), recovery_actions=(1, 0))


class _MalformedOnlineStepEnvironment(_OnlineOnlyEnvironment):
    def __init__(self, result: object, seed: int = 0) -> None:
        super().__init__(seed)
        self.result = result

    def step(self, action: int):
        validate_environment_action(action, self.spec)
        self.step_count += 1
        return self.result


class _UncheckedEnvironmentStep(EnvironmentStep[int]):
    def __post_init__(self) -> None:
        pass


class _UncheckedEnvironmentSpec(EnvironmentSpec):
    def __post_init__(self) -> None:
        pass


class _MalformedCloneEnvironment(_OnlineOnlyEnvironment):
    def __init__(self, branch_result: object) -> None:
        super().__init__()
        self.branch_result = branch_result
        self.last_branch: _MalformedOnlineStepEnvironment | None = None

    def clone(self) -> _MalformedOnlineStepEnvironment:
        self.last_branch = _MalformedOnlineStepEnvironment(self.branch_result, self.seed)
        return self.last_branch


def _probed_candidates(config: RolloutConfig) -> tuple[OptionCandidate, ...]:
    return (
        OptionCandidate(
            kind=OptionKind.STUDENT,
            actions=(0,),
            preview_steps=1,
            estimated_task_value=0.0,
            query_cost=config.teacher_query_cost,
            execution_cost=0.0,
            terminated=True,
            truncated=False,
        ),
        OptionCandidate(
            kind=OptionKind.TEACHER_CORRECTION,
            actions=(1,),
            preview_steps=1,
            estimated_task_value=0.0,
            query_cost=config.teacher_query_cost,
            execution_cost=config.teacher_execution_cost,
            terminated=True,
            truncated=False,
        ),
        OptionCandidate(
            kind=OptionKind.TEACHER_RECOVERY,
            actions=(1, 0),
            preview_steps=2,
            estimated_task_value=0.0,
            query_cost=config.teacher_query_cost,
            execution_cost=(1.0 + config.gamma) * config.teacher_execution_cost,
            terminated=True,
            truncated=False,
        ),
    )


class _TrackingAdapter:
    spec = _OnlineOnlyEnvironment.spec

    def __init__(self, *, evaluator_fails: bool = False) -> None:
        self.evaluator_fails = evaluator_fails
        self.seeds: list[int] = []
        self.require_teacher: list[bool] = []
        self.environments: list[_OnlineOnlyEnvironment] = []
        self.evaluators: list[_StudentOnlyEvaluator] = []

    @contextmanager
    def open_episode(
        self,
        *,
        seed: int,
        require_teacher: bool,
    ) -> Iterator[EpisodeComponents[int]]:
        environment = _OnlineOnlyEnvironment(seed)
        evaluator = _StudentOnlyEvaluator(fail=self.evaluator_fails)
        self.seeds.append(seed)
        self.require_teacher.append(require_teacher)
        self.environments.append(environment)
        self.evaluators.append(evaluator)
        try:
            yield EpisodeComponents(environment, None, evaluator)
        finally:
            try:
                evaluator.close()
            finally:
                environment.close()


class _MissingEvaluatorAdapter:
    spec = _OnlineOnlyEnvironment.spec

    def __init__(self) -> None:
        self.environment: _OnlineOnlyEnvironment | None = None

    @contextmanager
    def open_episode(self, *, seed: int, require_teacher: bool):
        self.environment = _OnlineOnlyEnvironment(seed)
        try:
            yield EpisodeComponents(self.environment, None, None)
        finally:
            self.environment.close()


class _EncodingEnvironment:
    spec = EnvironmentSpec(observation_size=2, action_size=2)

    def __init__(self, encoded) -> None:
        self.encoded = encoded

    def encode_observation(self, observation):
        return self.encoded


class EnvironmentSchemaTest(unittest.TestCase):
    def test_environment_spec_requires_positive_plain_integer_dimensions(self) -> None:
        self.assertEqual(EnvironmentSpec(3, 2), EnvironmentSpec(3, 2))
        for values in ((0, 2), (3, -1)):
            with self.subTest(values=values), self.assertRaises(ValueError):
                EnvironmentSpec(*values)

        unchecked = _UncheckedEnvironmentSpec(observation_size=0, action_size=2)
        with self.assertRaises(ValueError):
            validate_environment_spec(unchecked)
        for values in ((True, 2), (3.0, 2)):
            with self.subTest(values=values), self.assertRaises(TypeError):
                EnvironmentSpec(*values)

    def test_environment_step_enforces_finite_reward_and_terminal_schema(self) -> None:
        terminal = EnvironmentStep("done", 1, True, False, success=True)
        self.assertEqual(terminal.reward, 1)
        for reward in (math.inf, -math.inf, math.nan):
            with self.subTest(reward=reward), self.assertRaises(ValueError):
                EnvironmentStep("bad", reward, False, False)
        with self.assertRaises(TypeError):
            EnvironmentStep("bad", True, False, False)
        with self.assertRaises(TypeError):
            EnvironmentStep("bad", 0.0, 0, False)
        with self.assertRaises(ValueError):
            EnvironmentStep("bad", 0.0, True, True)
        with self.assertRaises(ValueError):
            EnvironmentStep("bad", 0.0, False, False, success=True)

    def test_encoded_observation_schema_is_fixed_real_and_finite(self) -> None:
        valid = _EncodingEnvironment((1, 2.5))
        self.assertEqual(validate_encoded_observation(valid, object()), (1.0, 2.5))

        invalid_rows = (
            ([1.0, 2.0], TypeError),
            ((1.0,), ValueError),
            ((1.0, True), TypeError),
            ((1.0, "2"), TypeError),
            ((1.0, math.nan), ValueError),
        )
        for encoded, error_type in invalid_rows:
            with self.subTest(encoded=encoded), self.assertRaises(error_type):
                validate_encoded_observation(_EncodingEnvironment(encoded), object())

    def test_action_schema_is_dense_zero_based_and_rejects_bool(self) -> None:
        spec = EnvironmentSpec(observation_size=2, action_size=2)
        self.assertEqual(validate_environment_action(0, spec), 0)
        self.assertEqual(validate_environment_action(1, spec), 1)
        for action in (-1, 2):
            with self.subTest(action=action), self.assertRaises(ValueError):
                validate_environment_action(action, spec)
        for action in (True, 1.0, "1"):
            with self.subTest(action=action), self.assertRaises(TypeError):
                validate_environment_action(action, spec)


class EnvironmentAdapterLifecycleTest(unittest.TestCase):
    def test_online_only_environment_rolls_out_with_injected_evaluator(self) -> None:
        environment = _OnlineOnlyEnvironment(seed=4)
        evaluator = _StudentOnlyEvaluator()
        self.assertFalse(hasattr(environment, "clone"))
        try:
            episode = RolloutCollector(
                RolloutConfig(probe_probability=0.0),
                evaluator=evaluator,
            ).collect_episode(environment, _ActionZeroStudent(), None)
        finally:
            evaluator.close()
            environment.close()

        self.assertTrue(episode.success)
        self.assertEqual(len(episode.transitions), 1)
        self.assertIs(episode.transitions[0].source, ActionSource.STUDENT)
        self.assertEqual(evaluator.teacher_proposals, [None])

    def test_collect_episodes_owns_fresh_seeded_student_only_sessions(self) -> None:
        adapter = _TrackingAdapter()
        episodes = collect_episodes(
            adapter,
            _ActionZeroStudent(),
            RolloutConfig(probe_probability=0.0),
            count=3,
            seed=17,
            deterministic_student=False,
            generator=torch.Generator().manual_seed(9),
        )

        self.assertEqual(adapter.seeds, [17, 18, 19])
        self.assertEqual(adapter.require_teacher, [False, False, False])
        self.assertEqual(len({id(env) for env in adapter.environments}), 3)
        self.assertEqual([env.reset_count for env in adapter.environments], [1, 1, 1])
        self.assertTrue(all(env.closed for env in adapter.environments))
        self.assertEqual([env.close_count for env in adapter.environments], [1, 1, 1])
        self.assertTrue(all(evaluator.closed for evaluator in adapter.evaluators))
        self.assertTrue(
            all(evaluator.teacher_proposals == [None] for evaluator in adapter.evaluators)
        )
        self.assertTrue(all(episode.success for episode in episodes))

    def test_adapter_cannot_silently_omit_candidate_evaluator(self) -> None:
        adapter = _MissingEvaluatorAdapter()
        with self.assertRaisesRegex(ValueError, "omitted.*candidate evaluator"):
            collect_episodes(
                adapter,
                _ActionZeroStudent(),
                RolloutConfig(probe_probability=0.0),
                count=1,
                seed=4,
                deterministic_student=True,
                generator=None,
            )

        self.assertIsNotNone(adapter.environment)
        self.assertEqual(adapter.environment.reset_count, 0)
        self.assertTrue(adapter.environment.closed)

    def test_student_failure_still_closes_episode_resources(self) -> None:
        adapter = _TrackingAdapter()
        with self.assertRaisesRegex(RuntimeError, "student failed"):
            collect_episodes(
                adapter,
                _ExplodingStudent(),
                RolloutConfig(probe_probability=0.0),
                count=1,
                seed=2,
                deterministic_student=False,
                generator=None,
            )

        self.assertTrue(adapter.environments[0].closed)
        self.assertTrue(adapter.evaluators[0].closed)
        self.assertEqual(adapter.environments[0].close_count, 1)
        self.assertEqual(adapter.evaluators[0].close_count, 1)

    def test_evaluator_failure_still_closes_episode_resources(self) -> None:
        adapter = _TrackingAdapter(evaluator_fails=True)
        with self.assertRaisesRegex(RuntimeError, "evaluator failed"):
            collect_episodes(
                adapter,
                _ActionZeroStudent(),
                RolloutConfig(probe_probability=0.0),
                count=1,
                seed=3,
                deterministic_student=False,
                generator=None,
            )

        self.assertTrue(adapter.environments[0].closed)
        self.assertTrue(adapter.evaluators[0].closed)
        self.assertEqual(adapter.environments[0].close_count, 1)
        self.assertEqual(adapter.evaluators[0].close_count, 1)

    def test_default_counterfactual_evaluator_rejects_online_only_env_before_reset(self) -> None:
        environment = _OnlineOnlyEnvironment()
        try:
            with self.assertRaisesRegex(TypeError, "requires a branchable environment"):
                RolloutCollector(RolloutConfig(probe_probability=0.0)).collect_episode(
                    environment,
                    _ActionZeroStudent(),
                    None,
                )
            self.assertEqual(environment.reset_count, 0)
        finally:
            environment.close()

    def _assert_candidates_rejected(
        self,
        candidates: tuple[OptionCandidate, ...],
        config: RolloutConfig,
        *,
        environment: _OnlineOnlyEnvironment | None = None,
        teacher=None,
    ) -> None:
        online_environment = environment or _OnlineOnlyEnvironment()
        evaluator = _FixedCandidatesEvaluator(candidates)
        try:
            with self.assertRaises((TypeError, ValueError)):
                RolloutCollector(config, evaluator=evaluator).collect_episode(
                    online_environment,
                    _ActionZeroStudent(),
                    teacher,
                )
        finally:
            evaluator.close()
            online_environment.close()
        self.assertTrue(evaluator.closed)

    def test_online_step_must_return_valid_environment_step(self) -> None:
        malformed_results = (
            _UncheckedEnvironmentStep(
                observation=1,
                reward=math.nan,
                terminated=True,
                truncated=False,
                success=True,
            ),
            SimpleNamespace(
                observation=1,
                reward=1.0,
                terminated=True,
                truncated=False,
                success=True,
            ),
            SimpleNamespace(
                observation=1,
                reward=math.nan,
                terminated=True,
                truncated=False,
                success=True,
            ),
        )
        for result in malformed_results:
            with self.subTest(reward=result.reward):
                environment = _MalformedOnlineStepEnvironment(result)
                evaluator = _StudentOnlyEvaluator()
                try:
                    with self.assertRaises((TypeError, ValueError)):
                        RolloutCollector(
                            RolloutConfig(probe_probability=0.0),
                            evaluator=evaluator,
                        ).collect_episode(environment, _ActionZeroStudent(), None)
                finally:
                    evaluator.close()
                    environment.close()

    def test_subclassed_student_proposal_is_revalidated_before_evaluation(self) -> None:
        environment = _OnlineOnlyEnvironment()
        evaluator = _StudentOnlyEvaluator()
        try:
            with self.assertRaisesRegex(ValueError, "log_prob must be finite"):
                RolloutCollector(
                    RolloutConfig(probe_probability=0.0),
                    evaluator=evaluator,
                ).collect_episode(environment, _InvalidProposalStudent(), None)
            self.assertEqual(evaluator.teacher_proposals, [])
            self.assertEqual(environment.step_count, 0)
        finally:
            evaluator.close()
            environment.close()

    def test_counterfactual_bootstrap_requires_a_finite_plain_real_value(self) -> None:
        for value, error_type in ((True, TypeError), (math.nan, ValueError)):
            with self.subTest(value=value):
                environment = JammedChainEnv()
                try:
                    with self.assertRaises(error_type):
                        RolloutCollector(
                            RolloutConfig(probe_probability=0.0)
                        ).collect_episode(environment, _InvalidValueStudent(value), None)
                finally:
                    environment.close()

    def test_candidate_subclasses_are_canonicalized_before_gate_and_execution(self) -> None:
        candidate = _MisleadingCandidate(
            kind=OptionKind.STUDENT,
            actions=(0,),
            preview_steps=1,
            estimated_task_value=1.0,
            query_cost=0.0,
            execution_cost=0.0,
            terminated=True,
            truncated=False,
        )
        environment = _OnlineOnlyEnvironment()
        evaluator = _FixedCandidatesEvaluator((candidate,))
        try:
            episode = RolloutCollector(
                RolloutConfig(probe_probability=0.0),
                evaluator=evaluator,
            ).collect_episode(environment, _ActionZeroStudent(), None)
        finally:
            evaluator.close()
            environment.close()

        self.assertEqual(episode.transitions[0].action, 0)
        self.assertIs(type(episode.decisions[0].candidates[0]), OptionCandidate)

    def test_counterfactual_preview_rejects_non_step_and_closes_branch(self) -> None:
        result = SimpleNamespace(
            observation=1,
            reward=math.nan,
            terminated=True,
            truncated=False,
            success=True,
        )
        environment = _MalformedCloneEnvironment(result)
        evaluator = CounterfactualCandidateEvaluator()
        student = _ActionZeroStudent()
        try:
            environment.reset()
            proposal = student.act(environment.encode_observation(0))
            with self.assertRaises((TypeError, ValueError)):
                evaluator.build_candidates(
                    environment,
                    proposal,
                    student,
                    RolloutConfig(probe_probability=0.0),
                    teacher_proposal=None,
                )
            self.assertIsNotNone(environment.last_branch)
            self.assertTrue(environment.last_branch.closed)
        finally:
            evaluator.close()
            environment.close()

    def test_unprobed_evaluator_cannot_inject_teacher_candidates(self) -> None:
        config = RolloutConfig(
            gamma=0.5,
            probe_probability=0.0,
            teacher_query_cost=0.1,
            teacher_execution_cost=0.2,
        )
        candidates = tuple(
            replace(candidate, query_cost=0.0)
            for candidate in _probed_candidates(config)
        )
        self._assert_candidates_rejected(candidates, config)

    def test_probed_evaluator_must_return_exactly_student_correction_recovery(self) -> None:
        config = RolloutConfig(
            gamma=0.5,
            probe_probability=1.0,
            teacher_query_cost=0.1,
            teacher_execution_cost=0.2,
        )
        candidates = _probed_candidates(config)
        self._assert_candidates_rejected(candidates[:-1], config, teacher=_FixedTeacher())

    def test_candidate_actions_must_match_the_proposals_exactly(self) -> None:
        config = RolloutConfig(
            gamma=0.5,
            probe_probability=1.0,
            teacher_query_cost=0.1,
            teacher_execution_cost=0.2,
        )
        base = _probed_candidates(config)
        mismatches = (
            (0, (1,)),
            (1, (0,)),
            (2, (0, 1)),
        )
        for index, actions in mismatches:
            with self.subTest(kind=base[index].kind):
                candidates = list(base)
                candidates[index] = replace(candidates[index], actions=actions)
                self._assert_candidates_rejected(
                    tuple(candidates),
                    config,
                    teacher=_FixedTeacher(),
                )

    def test_selected_candidate_terminal_metadata_must_match_execution(self) -> None:
        config = RolloutConfig(
            gamma=0.5,
            probe_probability=1.0,
            teacher_query_cost=0.1,
            teacher_execution_cost=0.2,
        )
        candidates = list(_probed_candidates(config))
        candidates[0] = replace(candidates[0], terminated=False)
        self._assert_candidates_rejected(
            tuple(candidates),
            config,
            teacher=_FixedTeacher(),
        )

    def test_selected_candidate_preview_steps_must_match_execution(self) -> None:
        config = RolloutConfig(
            gamma=0.5,
            probe_probability=1.0,
            teacher_query_cost=0.1,
            teacher_execution_cost=0.2,
        )
        candidates = list(_probed_candidates(config))
        candidates[2] = replace(candidates[2], estimated_task_value=10.0)
        self._assert_candidates_rejected(
            tuple(candidates),
            config,
            teacher=_FixedTeacher(),
        )

    def test_candidate_query_cost_must_match_probe_cost(self) -> None:
        config = RolloutConfig(
            gamma=0.5,
            probe_probability=1.0,
            teacher_query_cost=0.1,
            teacher_execution_cost=0.2,
        )
        candidates = list(_probed_candidates(config))
        candidates[0] = replace(candidates[0], query_cost=0.0)
        self._assert_candidates_rejected(
            tuple(candidates),
            config,
            teacher=_FixedTeacher(),
        )

    def test_teacher_execution_cost_must_be_discounted_over_preview_steps(self) -> None:
        config = RolloutConfig(
            gamma=0.5,
            probe_probability=1.0,
            teacher_query_cost=0.1,
            teacher_execution_cost=0.2,
        )
        candidates = list(_probed_candidates(config))
        candidates[2] = replace(candidates[2], execution_cost=0.4)
        self._assert_candidates_rejected(
            tuple(candidates),
            config,
            teacher=_FixedTeacher(),
        )

    def test_explicit_falsy_evaluator_and_gate_are_preserved(self) -> None:
        environment = _OnlineOnlyEnvironment()
        evaluator = _FalsyStudentOnlyEvaluator()
        gate = _FalsyGate()
        try:
            episode = RolloutCollector(
                RolloutConfig(probe_probability=0.0),
                evaluator=evaluator,
                gate=gate,
            ).collect_episode(environment, _ActionZeroStudent(), None)
        finally:
            evaluator.close()
            environment.close()

        self.assertTrue(episode.success)
        self.assertTrue(gate.called)
        self.assertEqual(evaluator.teacher_proposals, [None])

    def test_collect_episodes_accepts_negative_integer_seeds(self) -> None:
        adapter = _TrackingAdapter()
        episodes = collect_episodes(
            adapter,
            _ActionZeroStudent(),
            RolloutConfig(probe_probability=0.0),
            count=3,
            seed=-2,
            deterministic_student=True,
            generator=None,
        )
        self.assertEqual(adapter.seeds, [-2, -1, 0])
        self.assertEqual(len(episodes), 3)

    def test_seed_schema_rejects_bool_and_non_integer(self) -> None:
        for seed in (True, 1.5, "1"):
            with self.subTest(api="collect_episodes", seed=seed):
                with self.assertRaises(TypeError):
                    collect_episodes(
                        _TrackingAdapter(),
                        _ActionZeroStudent(),
                        RolloutConfig(probe_probability=0.0),
                        count=1,
                        seed=seed,
                        deterministic_student=True,
                        generator=None,
                    )
            with self.subTest(api="toy_runtime", seed=seed):
                with self.assertRaises(TypeError):
                    with ToyRuntimeAdapter().open_episode(
                        seed=seed,
                        require_teacher=False,
                    ):
                        pass


class ToyRuntimeAdapterTest(unittest.TestCase):
    def test_runtime_accepts_negative_integer_seed(self) -> None:
        with ToyRuntimeAdapter().open_episode(
            seed=-7,
            require_teacher=False,
        ) as components:
            self.assertEqual(components.environment.episode_seed, -7)

    def test_runtime_exposes_spec_seed_teacher_mode_and_idempotent_close(self) -> None:
        adapter = ToyRuntimeAdapter()
        self.assertEqual(adapter.spec, JammedChainEnv.spec)

        with adapter.open_episode(seed=23, require_teacher=False) as components:
            environment = components.environment
            evaluator = components.candidate_evaluator
            self.assertEqual(environment.spec, adapter.spec)
            self.assertEqual(environment.episode_seed, 23)
            self.assertIsNone(components.teacher)
        self.assertTrue(environment.closed)
        environment.close()
        environment.close()
        evaluator.close()
        evaluator.close()
        self.assertTrue(environment.closed)

        with adapter.open_episode(seed=24, require_teacher=True) as teacher_components:
            self.assertIsInstance(teacher_components.teacher, OracleTeacher)
        self.assertTrue(teacher_components.environment.closed)

    def test_toy_preview_does_not_mutate_online_state(self) -> None:
        config = JammedChainConfig(goal_position=3, trap_positions=(1,), max_steps=7)
        environment = JammedChainEnv(config, episode_seed=5)
        evaluator = CounterfactualCandidateEvaluator()
        student = _ActionZeroStudent()
        try:
            environment.reset()
            environment.step(int(ChainAction.ADVANCE))
            before = environment.observation
            proposal = student.act(environment.encode_observation(before))
            teacher_proposal = OracleTeacher(config).propose(before, recovery_horizon=2)

            candidates = evaluator.build_candidates(
                environment,
                proposal,
                student,
                RolloutConfig(probe_probability=1.0, recovery_horizon=2),
                teacher_proposal,
            )

            self.assertEqual(environment.observation, before)
            self.assertFalse(environment.done)
            self.assertEqual(
                {candidate.kind for candidate in candidates},
                {
                    OptionKind.STUDENT,
                    OptionKind.TEACHER_CORRECTION,
                    OptionKind.TEACHER_RECOVERY,
                },
            )
            actual = environment.step(int(ChainAction.REPAIR))
            self.assertEqual(actual.observation.steps_elapsed, before.steps_elapsed + 1)
        finally:
            evaluator.close()
            environment.close()

    def test_adapter_and_direct_toy_rollouts_are_bitwise_equivalent(self) -> None:
        toy_config = JammedChainConfig(
            goal_position=4,
            trap_positions=(1, 3),
            max_steps=10,
        )
        rollout_config = RolloutConfig(
            probe_probability=0.65,
            recovery_horizon=2,
            teacher_query_cost=0.013,
            teacher_execution_cost=0.027,
        )
        torch.manual_seed(41)
        model = ActorCritic(
            observation_size=JammedChainEnv.spec.observation_size,
            action_size=JammedChainEnv.spec.action_size,
            hidden_size=8,
        )
        adapter_generator = torch.Generator().manual_seed(812)
        direct_generator = torch.Generator().manual_seed(812)
        adapter_episodes = collect_episodes(
            ToyRuntimeAdapter(toy_config),
            model,
            rollout_config,
            count=3,
            seed=101,
            deterministic_student=False,
            generator=adapter_generator,
        )

        direct_episodes = []
        for index in range(3):
            episode_seed = 101 + index
            environment = JammedChainEnv(toy_config, episode_seed=episode_seed)
            evaluator = CounterfactualCandidateEvaluator()
            teacher = OracleTeacher(toy_config)
            try:
                direct_episodes.append(
                    RolloutCollector(
                        rollout_config,
                        seed=episode_seed,
                        evaluator=evaluator,
                        torch_generator=direct_generator,
                    ).collect_episode(
                        environment,
                        model,
                        teacher,
                        deterministic_student=False,
                    )
                )
            finally:
                evaluator.close()
                environment.close()

        self.assertEqual(adapter_episodes, direct_episodes)
        self.assertTrue(
            torch.equal(adapter_generator.get_state(), direct_generator.get_state())
        )

    def test_adapter_preserves_golden_student_then_recovery_trace_and_costs(self) -> None:
        toy_config = JammedChainConfig(
            goal_position=2,
            trap_positions=(1,),
            max_steps=6,
        )
        episode = collect_episodes(
            ToyRuntimeAdapter(toy_config),
            _ActionZeroStudent(),
            RolloutConfig(
                probe_probability=1.0,
                recovery_horizon=2,
                teacher_query_cost=0.01,
                teacher_execution_cost=0.02,
            ),
            count=1,
            seed=0,
            deterministic_student=True,
            generator=None,
        )[0]

        self.assertTrue(episode.success)
        self.assertEqual(
            [decision.selected_option for decision in episode.decisions],
            [OptionKind.STUDENT, OptionKind.TEACHER_RECOVERY],
        )
        self.assertEqual(
            [row.action for row in episode.transitions],
            [
                int(ChainAction.ADVANCE),
                int(ChainAction.REPAIR),
                int(ChainAction.ADVANCE),
            ],
        )
        self.assertEqual(
            [row.source for row in episode.transitions],
            [ActionSource.STUDENT, ActionSource.TEACHER, ActionSource.TEACHER],
        )
        self.assertEqual(
            [row.query_cost for row in episode.transitions],
            [0.01, 0.01, 0.0],
        )
        self.assertEqual(
            [row.execution_cost for row in episode.transitions],
            [0.0, 0.02, 0.02],
        )
        self.assertEqual(episode.teacher_costs.query_count, 2)
        self.assertEqual(episode.teacher_costs.generated_teacher_steps, 6)
        self.assertEqual(episode.teacher_costs.executed_teacher_steps, 2)
        self.assertAlmostEqual(episode.teacher_costs.query_cost, 0.02)
        self.assertAlmostEqual(episode.teacher_costs.execution_cost, 0.04)


if __name__ == "__main__":
    unittest.main()
