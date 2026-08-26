import unittest

from ar_opd.core import ActionSource, OptionKind, StudentProposal
from ar_opd.rollout import (
    CounterfactualCandidateEvaluator,
    RolloutCollector,
    RolloutConfig,
)
from ar_opd.teacher import OracleTeacher
from ar_opd.toy_env import ChainAction, JammedChainConfig, JammedChainEnv


class AdvanceStudent:
    def act(self, observation, *, deterministic=False, generator=None):
        return StudentProposal(action=int(ChainAction.ADVANCE), log_prob=-0.2, value=0.0)

    def value(self, observation):
        return 0.0


class LargeValueStudent(AdvanceStudent):
    def value(self, observation):
        return 999.0


class RolloutTest(unittest.TestCase):
    def test_recovery_expands_to_teacher_rows_and_costs_once(self) -> None:
        env = JammedChainEnv(
            JammedChainConfig(goal_position=2, trap_positions=(1,), max_steps=6)
        )
        teacher = OracleTeacher(env.config)
        collector = RolloutCollector(
            RolloutConfig(
                probe_probability=1.0,
                recovery_horizon=2,
                teacher_query_cost=0.01,
                teacher_execution_cost=0.02,
            ),
            seed=0,
        )
        episode = collector.collect_episode(env, AdvanceStudent(), teacher)

        self.assertTrue(episode.success)
        self.assertEqual(
            [decision.selected_option for decision in episode.decisions],
            [OptionKind.STUDENT, OptionKind.TEACHER_RECOVERY],
        )
        self.assertEqual(
            [row.source for row in episode.transitions],
            [ActionSource.STUDENT, ActionSource.TEACHER, ActionSource.TEACHER],
        )
        self.assertEqual(
            [row.action for row in episode.transitions[1:]],
            [int(ChainAction.REPAIR), int(ChainAction.ADVANCE)],
        )
        self.assertEqual(episode.actor_rows, 2)
        self.assertEqual(
            episode.decisions[1].student_proposal.action, int(ChainAction.ADVANCE)
        )
        self.assertAlmostEqual(episode.transitions[1].query_cost, 0.01)
        self.assertAlmostEqual(episode.transitions[2].query_cost, 0.0)
        self.assertEqual(episode.teacher_costs.query_count, 2)
        self.assertEqual(episode.teacher_costs.generated_teacher_steps, 6)
        self.assertEqual(episode.teacher_costs.executed_teacher_steps, 2)
        self.assertAlmostEqual(episode.teacher_costs.query_cost, 0.02)
        self.assertAlmostEqual(episode.teacher_costs.execution_cost, 0.04)
        self.assertAlmostEqual(
            episode.teacher_costs.query_cost,
            sum(row.query_cost for row in episode.transitions),
        )
        self.assertAlmostEqual(
            episode.teacher_costs.execution_cost,
            sum(row.execution_cost for row in episode.transitions),
        )

    def test_correction_discards_student_environment_action_but_keeps_proposal(self) -> None:
        env = JammedChainEnv(
            JammedChainConfig(goal_position=4, trap_positions=(1,), max_steps=8)
        )
        episode = RolloutCollector(
            RolloutConfig(probe_probability=1.0), seed=0
        ).collect_episode(env, AdvanceStudent(), OracleTeacher(env.config))

        correction = episode.decisions[1]
        actual = episode.transitions[correction.transition_start]
        self.assertIs(correction.selected_option, OptionKind.TEACHER_CORRECTION)
        self.assertEqual(correction.student_proposal.action, int(ChainAction.ADVANCE))
        self.assertEqual(actual.action, int(ChainAction.REPAIR))
        self.assertIs(actual.source, ActionSource.TEACHER)

    def test_no_probe_never_calls_teacher_or_charges_cost(self) -> None:
        env = JammedChainEnv(
            JammedChainConfig(goal_position=2, trap_positions=(1,), max_steps=4)
        )
        teacher = OracleTeacher(env.config)
        episode = RolloutCollector(
            RolloutConfig(probe_probability=0.0), seed=0
        ).collect_episode(env, AdvanceStudent(), teacher)

        self.assertEqual(teacher.call_count, 0)
        self.assertEqual(episode.teacher_costs.query_count, 0)
        self.assertEqual(episode.teacher_costs.executed_teacher_steps, 0)
        self.assertAlmostEqual(episode.teacher_costs.total_cost, 0.0)
        self.assertTrue(all(row.source is ActionSource.STUDENT for row in episode.transitions))
        self.assertEqual(episode.actor_rows, len(episode.decisions))

    def test_truncated_candidate_does_not_bootstrap(self) -> None:
        env = JammedChainEnv(
            JammedChainConfig(goal_position=2, trap_positions=(), max_steps=1)
        )
        env.reset()
        student = LargeValueStudent()
        proposal = student.act(env.encode_observation())
        candidate = CounterfactualCandidateEvaluator().build_candidates(
            env,
            proposal,
            student,
            RolloutConfig(probe_probability=0.0),
            teacher_proposal=None,
        )[0]
        self.assertTrue(candidate.truncated)
        self.assertAlmostEqual(candidate.estimated_task_value, env.config.step_reward)


if __name__ == "__main__":
    unittest.main()
