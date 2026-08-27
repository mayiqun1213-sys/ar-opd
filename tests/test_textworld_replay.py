import math
import unittest
from dataclasses import replace

import torch

from ar_opd.core import ActionSource, OptionKind, StudentProposal, TeacherProposal
from ar_opd.fake_textworld import (
    FAKE_TEXTWORLD_BACKEND_IDENTITY,
    JAMMED_QUEST_ACTION_VOCABULARY,
    JAMMED_QUEST_GAME,
    FakeTextWorldBackend,
    FakeTextWorldBackendFactory,
    JammedQuestTeacherFactory,
)
from ar_opd.models import ActorCritic
from ar_opd.ppo import PPOConfig, build_batch, ppo_update
from ar_opd.rollout import (
    RolloutConfig,
    collect_episodes,
)
from ar_opd.textworld_runtime import (
    BackendTransition,
    BoundaryFingerprint,
    EnvironmentFaultedError,
    FixedVocabularyActionCodec,
    OpaqueBackendDoneError,
    ReplayCandidateEvaluator,
    ReplayCursor,
    ReplayMismatchError,
    ReplayableTextWorldEnvironment,
    StableTextObservationEncoder,
    TaskOutcome,
    TextWorldEpisodeSpec,
    TextWorldObservation,
    TextWorldRuntimeAdapter,
    TextWorldRuntimeConfig,
)


class _EastStudent:
    def act(self, observation, *, deterministic=False, generator=None):
        return StudentProposal(action=1, log_prob=-0.5, value=0.0)

    def value(self, observation):
        return 0.0


class _ScoreDriftBackend(FakeTextWorldBackend):
    """Return a different replay reward while keeping the fake state deterministic."""

    def step(self, command: str) -> BackendTransition:
        result = super().step(command)
        if (
            command != "go east"
            or result.boundary.task_outcome is not TaskOutcome.ACTIVE
        ):
            return result
        return BackendTransition(
            boundary=replace(result.boundary, score=result.boundary.score + 0.01),
            done=result.done,
        )


class _OpaqueDoneBackend(FakeTextWorldBackend):
    def step(self, command: str) -> BackendTransition:
        result = super().step(command)
        return BackendTransition(
            boundary=result.boundary,
            done=True,
        )


class _MutationThenRaiseBackend(FakeTextWorldBackend):
    def step(self, command: str) -> BackendTransition:
        super().step(command)
        raise RuntimeError("backend failed after mutation")


class _FailureBackend(FakeTextWorldBackend):
    def step(self, command: str) -> BackendTransition:
        result = super().step(command)
        return BackendTransition(
            boundary=replace(result.boundary, task_failure=True),
            done=True,
        )


class _ScoreOnlySuccessBackend(FakeTextWorldBackend):
    def step(self, command: str) -> BackendTransition:
        result = super().step(command)
        boundary = replace(
            result.boundary,
            score_raw=1.0,
            score=1.0,
            task_success=False,
            task_failure=False,
        )
        return BackendTransition(boundary=boundary, done=True)


class _TerminalResetBackend(FakeTextWorldBackend):
    def reset(self, episode: TextWorldEpisodeSpec):
        boundary = super().reset(episode)
        return replace(boundary, task_failure=True)


class _EmptyResetBackend(FakeTextWorldBackend):
    def reset(self, episode: TextWorldEpisodeSpec):
        boundary = super().reset(episode)
        return replace(boundary, valid_actions=())


class _SecondBackendFailsFactory:
    def __init__(self) -> None:
        self.instances: list[FakeTextWorldBackend] = []

    def __call__(self):
        if self.instances:
            raise RuntimeError("scratch construction failed")
        backend = FakeTextWorldBackend(JAMMED_QUEST_GAME)
        self.instances.append(backend)
        return backend


def _config(*, max_steps: int = 8) -> TextWorldRuntimeConfig:
    return TextWorldRuntimeConfig(
        game_name=JAMMED_QUEST_GAME.name,
        backend_identity=FAKE_TEXTWORLD_BACKEND_IDENTITY,
        action_vocabulary=JAMMED_QUEST_ACTION_VOCABULARY,
        observation_size=8,
        project_max_steps=max_steps,
    )


def _episode(*, max_steps: int = 8, seed: int = 7) -> TextWorldEpisodeSpec:
    return TextWorldEpisodeSpec(config=_config(max_steps=max_steps), seed=seed)


class TextWorldCodecAndFingerprintTest(unittest.TestCase):
    def test_runtime_config_canonicalizes_and_binds_backend_abi(self) -> None:
        canonical = replace(
            _config(),
            game_fold="dev",
            game_params=" z=+02, a=-0 ",
        )
        self.assertEqual(canonical.game_params, "a=0,z=2")
        self.assertNotEqual(
            canonical.abi_sha256,
            replace(canonical, backend_identity="another-backend-v1").abi_sha256,
        )

        for invalid_fold in ("validation", " train"):
            with self.subTest(fold=invalid_fold), self.assertRaises(ValueError):
                replace(_config(), game_fold=invalid_fold)
        for invalid_params in ("missing", "a=1,a=2", "a=text", f"a={2**31}"):
            with self.subTest(params=invalid_params), self.assertRaises(ValueError):
                replace(_config(), game_params=invalid_params)
        with self.assertRaisesRegex(ValueError, "help command"):
            replace(
                _config(),
                action_vocabulary=JAMMED_QUEST_ACTION_VOCABULARY + ("help",),
            )

        for seed in (-(2**31), 2**31 - 1):
            self.assertEqual(_episode(seed=seed).seed, seed)
        for seed in (-(2**31) - 1, 2**31):
            with self.subTest(seed=seed), self.assertRaisesRegex(ValueError, "32-bit"):
                _episode(seed=seed)

    def test_fixed_ids_dynamic_masks_and_invalid_action_rejection(self) -> None:
        codec = FixedVocabularyActionCodec(JAMMED_QUEST_ACTION_VOCABULARY)
        first = codec.bind(("wait", "go east", "look"))
        permuted = codec.bind(("look", "wait", "go east"))
        self.assertEqual(first, permuted)
        self.assertEqual(first.action_ids, (0, 1, 3))
        self.assertEqual(first.mask, (True, True, False, True, False))
        self.assertEqual(first.command_for(1), "go east")
        with self.assertRaisesRegex(ValueError, "duplicate"):
            codec.bind(("look", "look"))
        with self.assertRaisesRegex(ValueError, "fixed vocabulary"):
            codec.bind(("look", "dance"))

        backend = FakeTextWorldBackend(JAMMED_QUEST_GAME)
        environment = ReplayableTextWorldEnvironment(backend, _episode())
        try:
            observation = environment.reset()
            self.assertEqual(observation.action_view, first)
            cursor = environment.replay_cursor()
            attempts = tuple(backend.step_attempts)
            for invalid_action in (2, len(JAMMED_QUEST_ACTION_VOCABULARY)):
                with self.subTest(action=invalid_action), self.assertRaises(ValueError):
                    environment.step(invalid_action)
                self.assertEqual(tuple(backend.step_attempts), attempts)
                self.assertEqual(environment.replay_cursor(), cursor)

            jammed = environment.step(1).observation
            self.assertEqual(jammed.action_mask, (True, True, True, False, True))
            self.assertEqual(
                tuple(choice.command for choice in jammed.valid_actions),
                ("look", "go east", "repair cart", "go west"),
            )
        finally:
            environment.close()

    def test_fingerprint_canonicalizes_menu_order_and_binds_semantics(self) -> None:
        spec = _episode(seed=11)
        codec = FixedVocabularyActionCodec(JAMMED_QUEST_ACTION_VOCABULARY)
        observation = TextWorldObservation(
            text="same text",
            look="same room",
            inventory="empty",
            action_view=codec.bind(("wait", "go east", "look")),
            score_raw=2.0,
            score=0.25,
            task_description="finish the task",
            steps_elapsed=0,
            state_token="start",
        )
        reordered = replace(
            observation,
            action_view=codec.bind(("look", "wait", "go east")),
        )
        fingerprint = BoundaryFingerprint.create(spec, observation)
        self.assertEqual(fingerprint, BoundaryFingerprint.create(spec, reordered))
        self.assertEqual(len(fingerprint.digest), 64)
        int(fingerprint.digest, 16)

        mutations = (
            replace(observation, text="same text "),
            replace(observation, look="another room"),
            replace(observation, inventory="carrying a key"),
            replace(observation, score_raw=3.0),
            replace(observation, score=0.5),
            replace(observation, task_description="another task"),
            replace(observation, task_failure=True),
            replace(observation, state_token="other"),
            replace(observation, action_view=codec.bind(("look", "wait"))),
        )
        for mutated in mutations:
            with self.subTest(observation=mutated):
                self.assertNotEqual(
                    fingerprint,
                    BoundaryFingerprint.create(spec, mutated),
                )

        encoder = StableTextObservationEncoder(8)
        self.assertEqual(
            encoder.encode(observation),
            encoder.encode(replace(observation, state_token="private-backend-state")),
        )
        with self.assertRaisesRegex(ValueError, "both be true"):
            replace(observation, task_success=True, task_failure=True)


class TextWorldReplayIntegrityTest(unittest.TestCase):
    def test_cursor_tampering_is_rejected_during_restore(self) -> None:
        spec = _episode(seed=13)
        online = ReplayableTextWorldEnvironment(
            FakeTextWorldBackend(JAMMED_QUEST_GAME),
            spec,
        )
        try:
            initial_observation = online.reset()
            initial_cursor = online.replay_cursor()
            with self.assertRaisesRegex(ValueError, "digest does not match"):
                replace(initial_cursor.initial, digest="0" * 64)
            bad_initial = BoundaryFingerprint.create(
                spec,
                replace(initial_observation, look="replay reset drift"),
            )
            bad_initial_cursor = ReplayCursor(
                episode_spec=spec,
                initial=bad_initial,
                steps=(),
            )

            def assert_restore_fails_closed(cursor: ReplayCursor) -> None:
                scratch = ReplayableTextWorldEnvironment(
                    FakeTextWorldBackend(JAMMED_QUEST_GAME),
                    spec,
                )
                try:
                    with self.assertRaises(ReplayMismatchError):
                        scratch.restore(cursor)
                    self.assertTrue(scratch.faulted)
                    with self.assertRaises(EnvironmentFaultedError):
                        scratch.replay_cursor()
                finally:
                    scratch.close()

            assert_restore_fails_closed(bad_initial_cursor)

            online.step(1)
            cursor = online.replay_cursor()
            step = cursor.steps[0]
            bad_after = BoundaryFingerprint.create(
                spec,
                replace(online.observation, inventory="replay step drift"),
            )
            tampered_steps = (
                replace(step, command="wait"),
                replace(step, reward=step.reward + 0.5),
                replace(step, after=bad_after),
            )
            for tampered in tampered_steps:
                bad_cursor = ReplayCursor(
                    episode_spec=spec,
                    initial=cursor.initial,
                    steps=(tampered,),
                )
                with self.subTest(step=tampered):
                    assert_restore_fails_closed(bad_cursor)
            self.assertEqual(online.replay_cursor(), cursor)
        finally:
            online.close()

    def test_single_scratch_is_reset_replayed_and_never_mutates_online(self) -> None:
        spec = _episode(seed=17)
        factory = FakeTextWorldBackendFactory()
        online = ReplayableTextWorldEnvironment(factory(), spec)
        scratch = ReplayableTextWorldEnvironment(factory(), spec)
        evaluator = ReplayCandidateEvaluator(scratch)
        student = _EastStudent()
        try:
            online.reset()
            online.step(1)
            cursor = online.replay_cursor()
            candidates = evaluator.build_candidates(
                online,
                student.act(online.encode_observation(online.observation)),
                student,
                RolloutConfig(),
                TeacherProposal(
                    correction_actions=(2,),
                    recovery_actions=(2, 1),
                ),
            )

            self.assertEqual(factory.call_count, 2)
            self.assertEqual(len(factory.instances[0].reset_calls), 1)
            self.assertEqual(len(factory.instances[1].reset_calls), 3)
            self.assertEqual(factory.instances[0].step_calls, ["go east"])
            self.assertEqual(online.replay_cursor(), cursor)
            self.assertEqual(
                tuple(candidate.kind for candidate in candidates),
                (
                    OptionKind.STUDENT,
                    OptionKind.TEACHER_CORRECTION,
                    OptionKind.TEACHER_RECOVERY,
                ),
            )
            self.assertTrue(candidates[-1].terminated)

            evaluator.build_candidates(
                online,
                student.act(online.encode_observation(online.observation)),
                student,
                RolloutConfig(),
                TeacherProposal(
                    correction_actions=(2,),
                    recovery_actions=(2, 1),
                ),
            )
            self.assertEqual(factory.call_count, 2)
            self.assertEqual(len(factory.instances[1].reset_calls), 6)
            self.assertEqual(online.replay_cursor(), cursor)
        finally:
            evaluator.close()
            evaluator.close()
            online.close()
            online.close()
        self.assertEqual([backend.close_count for backend in factory.instances], [1, 1])

    def test_replay_score_drift_is_detected_before_candidate_preview(self) -> None:
        spec = _episode(seed=19)
        online_backend = FakeTextWorldBackend(JAMMED_QUEST_GAME)
        scratch_backend = _ScoreDriftBackend(JAMMED_QUEST_GAME)
        online = ReplayableTextWorldEnvironment(online_backend, spec)
        scratch = ReplayableTextWorldEnvironment(scratch_backend, spec)
        evaluator = ReplayCandidateEvaluator(scratch)
        try:
            online.reset()
            online.step(1)
            cursor = online.replay_cursor()
            with self.assertRaisesRegex(ReplayMismatchError, "backend result differs"):
                evaluator.build_candidates(
                    online,
                    StudentProposal(action=1, log_prob=-0.5, value=0.0),
                    _EastStudent(),
                    RolloutConfig(probe_probability=0.0),
                    None,
                )
            self.assertEqual(online.replay_cursor(), cursor)
            self.assertEqual(scratch_backend.step_calls, ["go east"])
        finally:
            evaluator.close()
            online.close()


class TextWorldOutcomeAndLifecycleTest(unittest.TestCase):
    def test_terminal_and_empty_active_resets_fail_closed(self) -> None:
        cases = (
            (_TerminalResetBackend, "active task boundary"),
            (_EmptyResetBackend, "valid action"),
        )
        for backend_type, message in cases:
            with self.subTest(backend=backend_type.__name__):
                backend = backend_type(JAMMED_QUEST_GAME)
                environment = ReplayableTextWorldEnvironment(
                    backend,
                    _episode(seed=21),
                )
                try:
                    with self.assertRaisesRegex(ValueError, message):
                        environment.reset()
                finally:
                    environment.close()
                self.assertEqual(backend.close_count, 1)

    def test_opaque_done_fails_closed_without_committing_a_trace(self) -> None:
        backend = _OpaqueDoneBackend(JAMMED_QUEST_GAME)
        environment = ReplayableTextWorldEnvironment(backend, _episode(seed=23))
        try:
            environment.reset()
            with self.assertRaises(OpaqueBackendDoneError):
                environment.step(1)
            self.assertTrue(environment.faulted)
            self.assertEqual(backend.step_calls, ["go east"])
            for operation in (
                lambda: environment.observation,
                lambda: environment.replay_cursor(),
                lambda: environment.step(1),
                lambda: environment.reset(),
            ):
                with self.assertRaises(EnvironmentFaultedError):
                    operation()
        finally:
            environment.close()

    def test_backend_mutation_then_error_faults_but_local_rejection_does_not(self) -> None:
        local_backend = FakeTextWorldBackend(JAMMED_QUEST_GAME)
        local_environment = ReplayableTextWorldEnvironment(
            local_backend,
            _episode(seed=24),
        )
        try:
            local_environment.reset()
            with self.assertRaisesRegex(ValueError, "not valid"):
                local_environment.step(2)
            self.assertFalse(local_environment.faulted)
            self.assertEqual(local_backend.step_calls, [])
            self.assertFalse(local_environment.step(1).terminated)
        finally:
            local_environment.close()

        backend = _MutationThenRaiseBackend(JAMMED_QUEST_GAME)
        environment = ReplayableTextWorldEnvironment(backend, _episode(seed=25))
        try:
            environment.reset()
            with self.assertRaisesRegex(RuntimeError, "after mutation"):
                environment.step(1)
            self.assertTrue(environment.faulted)
            self.assertEqual(backend.state_name, "jammed")
            with self.assertRaises(EnvironmentFaultedError):
                environment.action_mask
        finally:
            environment.close()

    def test_task_terminal_precedes_cap_and_active_cap_truncates(self) -> None:
        success_backend = FakeTextWorldBackend(JAMMED_QUEST_GAME)
        success_env = ReplayableTextWorldEnvironment(
            success_backend,
            _episode(max_steps=3, seed=29),
        )
        try:
            success_env.reset()
            self.assertFalse(success_env.step(1).terminated)
            self.assertFalse(success_env.step(2).terminated)
            result = success_env.step(1)
            self.assertTrue(result.terminated)
            self.assertFalse(result.truncated)
            self.assertTrue(result.success)
        finally:
            success_env.close()

        score_backend = _ScoreOnlySuccessBackend(JAMMED_QUEST_GAME)
        score_env = ReplayableTextWorldEnvironment(
            score_backend,
            _episode(max_steps=1, seed=30),
        )
        try:
            score_env.reset()
            result = score_env.step(1)
            self.assertTrue(result.terminated)
            self.assertFalse(result.truncated)
            self.assertTrue(result.success)
            self.assertFalse(result.observation.task_success)
        finally:
            score_env.close()

        failure_backend = _FailureBackend(JAMMED_QUEST_GAME)
        failure_env = ReplayableTextWorldEnvironment(
            failure_backend,
            _episode(max_steps=1, seed=31),
        )
        try:
            failure_env.reset()
            result = failure_env.step(1)
            self.assertTrue(result.terminated)
            self.assertFalse(result.truncated)
            self.assertFalse(result.success)
        finally:
            failure_env.close()

        cap_backend = FakeTextWorldBackend(JAMMED_QUEST_GAME)
        cap_env = ReplayableTextWorldEnvironment(
            cap_backend,
            _episode(max_steps=1, seed=37),
        )
        try:
            cap_env.reset()
            result = cap_env.step(1)
            self.assertFalse(result.terminated)
            self.assertTrue(result.truncated)
            self.assertFalse(result.success)
            calls = tuple(cap_backend.step_calls)
            with self.assertRaisesRegex(RuntimeError, "episode boundary"):
                cap_env.step(1)
            self.assertEqual(tuple(cap_backend.step_calls), calls)
        finally:
            cap_env.close()

    def test_close_is_idempotent_and_partial_construction_is_cleaned_up(self) -> None:
        backend = FakeTextWorldBackend(JAMMED_QUEST_GAME)
        backend.close()
        backend.close()
        self.assertTrue(backend.closed)
        self.assertEqual(backend.close_count, 1)

        failing_factory = _SecondBackendFailsFactory()
        adapter = TextWorldRuntimeAdapter(
            _config(),
            backend_factory=failing_factory,
        )
        with self.assertRaisesRegex(RuntimeError, "scratch construction failed"):
            with adapter.open_episode(seed=41, require_teacher=False):
                self.fail("adapter yielded despite scratch construction failure")
        self.assertEqual(len(failing_factory.instances), 1)
        self.assertEqual(failing_factory.instances[0].close_count, 1)

        factory = FakeTextWorldBackendFactory()

        def exploding_teacher(episode_spec):
            raise RuntimeError("Teacher construction failed")

        adapter = TextWorldRuntimeAdapter(
            _config(),
            backend_factory=factory,
            teacher_factory=exploding_teacher,
        )
        with self.assertRaisesRegex(RuntimeError, "Teacher construction failed"):
            with adapter.open_episode(seed=43, require_teacher=True):
                self.fail("adapter yielded despite Teacher construction failure")
        self.assertEqual(factory.call_count, 2)
        self.assertTrue(all(item.closed for item in factory.instances))
        self.assertEqual([item.close_count for item in factory.instances], [1, 1])


class TextWorldRolloutIntegrationTest(unittest.TestCase):
    def test_hybrid_s_then_f_is_ppo_consumable(self) -> None:
        backend_factory = FakeTextWorldBackendFactory()
        teacher_factory = JammedQuestTeacherFactory()
        adapter = TextWorldRuntimeAdapter(
            _config(),
            backend_factory=backend_factory,
            teacher_factory=teacher_factory,
        )
        episodes = collect_episodes(
            adapter,
            _EastStudent(),
            RolloutConfig(
                probe_probability=1.0,
                recovery_horizon=2,
                teacher_query_cost=0.01,
                teacher_execution_cost=0.02,
            ),
            count=1,
            seed=47,
            deterministic_student=True,
            generator=None,
        )
        episode = episodes[0]
        self.assertTrue(episode.success)
        self.assertEqual(
            [decision.selected_option for decision in episode.decisions],
            [OptionKind.STUDENT, OptionKind.TEACHER_RECOVERY],
        )
        self.assertEqual([row.action for row in episode.transitions], [1, 2, 1])
        self.assertEqual(
            [row.source for row in episode.transitions],
            [ActionSource.STUDENT, ActionSource.TEACHER, ActionSource.TEACHER],
        )
        self.assertEqual(episode.teacher_costs.probe_count, 2)
        self.assertEqual(episode.teacher_costs.query_count, 2)
        self.assertEqual(episode.teacher_costs.generated_teacher_steps, 5)
        self.assertEqual(episode.teacher_costs.executed_teacher_steps, 2)
        self.assertAlmostEqual(episode.teacher_costs.query_cost, 0.02)
        self.assertAlmostEqual(episode.teacher_costs.execution_cost, 0.04)
        self.assertEqual(backend_factory.call_count, 2)
        self.assertEqual(len(backend_factory.instances[0].reset_calls), 1)
        self.assertEqual(len(backend_factory.instances[1].reset_calls), 6)
        self.assertTrue(all(item.closed for item in backend_factory.instances))
        self.assertEqual(teacher_factory.call_count, 1)

        torch.manual_seed(3)
        model = ActorCritic(8, len(JAMMED_QUEST_ACTION_VOCABULARY), hidden_size=8)
        ppo_config = PPOConfig(epochs=1)
        batch = build_batch(episodes, model, ppo_config)
        self.assertEqual(batch.actions.tolist(), [1, 1])
        self.assertTrue(bool(torch.isfinite(batch.advantages).all()))
        self.assertTrue(bool(torch.isfinite(batch.returns).all()))
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        metrics = ppo_update(model, optimizer, batch, ppo_config)
        self.assertTrue(all(math.isfinite(value) for value in metrics.values()))
        self.assertEqual(metrics["actor_rows"], 2.0)
        self.assertEqual(metrics["critic_rows"], 2.0)

    def test_student_only_does_not_construct_teacher_and_truncates(self) -> None:
        backend_factory = FakeTextWorldBackendFactory()
        teacher_factory = JammedQuestTeacherFactory()
        adapter = TextWorldRuntimeAdapter(
            _config(max_steps=3),
            backend_factory=backend_factory,
            teacher_factory=teacher_factory,
        )
        episode = collect_episodes(
            adapter,
            _EastStudent(),
            RolloutConfig(probe_probability=0.0),
            count=1,
            seed=53,
            deterministic_student=True,
            generator=None,
        )[0]
        self.assertEqual(teacher_factory.call_count, 0)
        self.assertFalse(episode.success)
        self.assertEqual(len(episode.decisions), 3)
        self.assertTrue(episode.transitions[-1].truncated)
        self.assertFalse(episode.transitions[-1].terminated)
        self.assertTrue(
            all(decision.selected_option is OptionKind.STUDENT for decision in episode.decisions)
        )
        self.assertTrue(all(len(decision.candidates) == 1 for decision in episode.decisions))
        self.assertEqual(episode.teacher_costs.total_cost, 0.0)
        self.assertEqual(backend_factory.call_count, 2)
        self.assertEqual(len(backend_factory.instances[1].reset_calls), 3)
        self.assertTrue(all(item.closed for item in backend_factory.instances))


if __name__ == "__main__":
    unittest.main()
