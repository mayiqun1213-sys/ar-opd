import copy
import math
import unittest
from dataclasses import replace

import torch

from ar_opd.core import StudentProposal
from ar_opd.distillation import (
    LocalSFTConfig,
    LocalSFTDataset,
    LocalSFTExample,
    LocalSFTKind,
    extract_local_sft_examples,
    local_sft_update,
    validate_episode_for_distillation,
)
from ar_opd.models import ActorCritic
from ar_opd.rollout import RolloutCollector, RolloutConfig
from ar_opd.teacher import OracleTeacher
from ar_opd.toy_env import ChainAction, JammedChainConfig, JammedChainEnv


class AdvanceStudent:
    def act(self, observation, *, deterministic=False, generator=None):
        return StudentProposal(action=int(ChainAction.ADVANCE), log_prob=-0.2, value=0.0)

    def value(self, observation):
        return 0.0


def collect(goal: int, traps: tuple[int, ...], max_steps: int = 8):
    env = JammedChainEnv(
        JammedChainConfig(goal_position=goal, trap_positions=traps, max_steps=max_steps)
    )
    return RolloutCollector(
        RolloutConfig(probe_probability=1.0), seed=0
    ).collect_episode(env, AdvanceStudent(), OracleTeacher(env.config))


class LocalDistillationTest(unittest.TestCase):
    def test_routes_only_actual_teacher_steps_to_local_sft(self) -> None:
        recovery_episode = collect(2, (1,), 6)
        correction_episode = collect(4, (1,), 8)
        dataset = extract_local_sft_examples((recovery_episode, correction_episode))

        self.assertEqual(len(dataset.corrective), 1)
        self.assertEqual(len(dataset.fallback), 4)
        self.assertEqual(
            len(dataset.corrective) + len(dataset.fallback),
            recovery_episode.teacher_costs.executed_teacher_steps
            + correction_episode.teacher_costs.executed_teacher_steps,
        )
        corrective = dataset.corrective[0]
        self.assertIs(corrective.kind, LocalSFTKind.CORRECTIVE)
        self.assertEqual(corrective.target_action, int(ChainAction.REPAIR))
        self.assertTrue(corrective.observation[1])

        self.assertEqual(
            [example.target_action for example in dataset.fallback],
            [
                int(ChainAction.REPAIR),
                int(ChainAction.ADVANCE),
                int(ChainAction.ADVANCE),
                int(ChainAction.ADVANCE),
            ],
        )
        self.assertTrue(dataset.fallback[0].observation[1])
        self.assertFalse(dataset.fallback[1].observation[1])
        self.assertAlmostEqual(sum(example.weight for example in dataset.fallback), 2.0)

    def test_rejects_corrupt_decision_slice_before_extraction(self) -> None:
        episode = collect(2, (1,), 6)
        corrupt_decision = replace(
            episode.decisions[0], transition_start=-1, transition_stop=1
        )
        corrupt = replace(
            episode, decisions=[corrupt_decision, *episode.decisions[1:]]
        )
        with self.assertRaisesRegex(ValueError, "contiguous|out of bounds"):
            validate_episode_for_distillation(corrupt)

    def test_rejects_episode_that_continues_after_terminal_transition(self) -> None:
        episode = collect(2, (1,), 6)
        transitions = list(episode.transitions)
        transitions[0] = replace(transitions[0], terminated=True)
        corrupt = replace(episode, transitions=transitions)

        with self.assertRaisesRegex(ValueError, "cannot continue after"):
            validate_episode_for_distillation(corrupt)

    def test_rejects_probe_and_candidate_set_disagreement(self) -> None:
        episode = collect(2, (1,), 6)
        first = episode.decisions[0]
        cases = {
            "unprobed decision retains Teacher candidates": replace(
                first, probed=False
            ),
            "probed decision omits a Teacher candidate": replace(
                first, candidates=first.candidates[:-1]
            ),
            "probed decision duplicates a candidate kind": replace(
                first, candidates=(*first.candidates, first.candidates[0])
            ),
        }

        for label, corrupt_decision in cases.items():
            with self.subTest(label=label):
                corrupt = replace(
                    episode,
                    decisions=[corrupt_decision, *episode.decisions[1:]],
                )
                with self.assertRaisesRegex(ValueError, "candidate set"):
                    validate_episode_for_distillation(corrupt)

    def test_rejects_teacher_ledger_disagreement(self) -> None:
        episode = collect(2, (1,), 6)
        cases = (
            ("probe_count", 1, "probe ledger"),
            ("query_count", 1, "query ledger"),
            ("generated_teacher_steps", 1, "generated-step ledger"),
            ("executed_teacher_steps", 1, "teacher-step ledger"),
            ("query_cost", 1.0, "query-cost ledger"),
            ("execution_cost", 1.0, "execution-cost ledger"),
        )

        for field, increment, message in cases:
            with self.subTest(field=field):
                corrupt_ledger = replace(
                    episode.teacher_costs,
                    **{field: getattr(episode.teacher_costs, field) + increment},
                )
                corrupt = replace(episode, teacher_costs=corrupt_ledger)
                with self.assertRaisesRegex(ValueError, message):
                    validate_episode_for_distillation(corrupt)

    def test_rejects_selected_candidate_preview_disagreement(self) -> None:
        episode = collect(2, (1,), 6)
        recovery = episode.decisions[1]
        selected_index = next(
            index
            for index, candidate in enumerate(recovery.candidates)
            if candidate.kind is recovery.selected_option
        )
        candidates = list(recovery.candidates)
        candidates[selected_index] = replace(
            candidates[selected_index], preview_steps=1
        )
        corrupt_decision = replace(recovery, candidates=tuple(candidates))
        corrupt = replace(
            episode,
            decisions=[episode.decisions[0], corrupt_decision],
        )

        with self.assertRaisesRegex(
            ValueError, "previewed candidate prefix|preview length"
        ):
            validate_episode_for_distillation(corrupt)

    def test_rejects_teacher_execution_cost_shifted_to_student_transition(self) -> None:
        episode = collect(2, (1,), 6)
        transitions = list(episode.transitions)
        shifted_cost = transitions[1].execution_cost
        self.assertGreater(shifted_cost, 0.0)
        transitions[0] = replace(transitions[0], execution_cost=shifted_cost)
        transitions[1] = replace(transitions[1], execution_cost=0.0)
        corrupt = replace(episode, transitions=transitions)

        self.assertAlmostEqual(
            sum(row.execution_cost for row in corrupt.transitions),
            corrupt.teacher_costs.execution_cost,
        )
        with self.assertRaisesRegex(
            ValueError, "Student transitions cannot carry Teacher execution cost"
        ):
            validate_episode_for_distillation(corrupt)

    def test_rejects_synchronized_infinite_candidate_transition_and_ledger_cost(self) -> None:
        episode = collect(2, (1,), 6)
        first = episode.decisions[0]
        corrupt_decision = replace(
            first,
            candidates=tuple(
                replace(candidate, query_cost=math.inf)
                for candidate in first.candidates
            ),
        )
        transitions = list(episode.transitions)
        transitions[0] = replace(transitions[0], query_cost=math.inf)
        corrupt = replace(
            episode,
            decisions=[corrupt_decision, *episode.decisions[1:]],
            transitions=transitions,
            teacher_costs=replace(episode.teacher_costs, query_cost=math.inf),
        )

        self.assertTrue(
            math.isinf(sum(row.query_cost for row in corrupt.transitions))
        )
        with self.assertRaisesRegex(ValueError, "rewards and costs must be finite"):
            validate_episode_for_distillation(corrupt)

    def test_rejects_student_proposal_and_candidate_action_disagreement(self) -> None:
        episode = collect(2, (1,), 6)
        first = episode.decisions[0]
        self.assertEqual(first.student_proposal.action, int(ChainAction.ADVANCE))
        corrupt_decision = replace(
            first,
            student_proposal=replace(
                first.student_proposal,
                action=int(ChainAction.REPAIR),
            ),
        )
        corrupt = replace(
            episode,
            decisions=[corrupt_decision, *episode.decisions[1:]],
        )

        with self.assertRaisesRegex(
            ValueError, "Student candidate must match the sampled proposal"
        ):
            validate_episode_for_distillation(corrupt)

    def test_actor_head_only_update_reduces_nll_without_value_drift(self) -> None:
        torch.manual_seed(13)
        model = ActorCritic(3, 2, hidden_size=8)
        dataset = LocalSFTDataset(
            corrective=(
                LocalSFTExample(
                    observation=(0.1, 1.0, 0.8),
                    target_action=int(ChainAction.REPAIR),
                    kind=LocalSFTKind.CORRECTIVE,
                    episode_index=0,
                    decision_id=0,
                    primitive_offset=0,
                    segment_length=1,
                    weight=1.0,
                ),
            ),
            fallback=(
                LocalSFTExample(
                    observation=(0.5, 0.0, 0.5),
                    target_action=int(ChainAction.ADVANCE),
                    kind=LocalSFTKind.FALLBACK,
                    episode_index=0,
                    decision_id=1,
                    primitive_offset=0,
                    segment_length=1,
                    weight=1.0,
                ),
            ),
        )
        encoder_before = copy.deepcopy(model.encoder.state_dict())
        value_head_before = copy.deepcopy(model.value_head.state_dict())
        observations = torch.tensor(
            [dataset.corrective[0].observation, dataset.fallback[0].observation]
        )
        with torch.no_grad():
            values_before = model(observations)[1].clone()

        metrics = local_sft_update(
            model,
            dataset,
            LocalSFTConfig(epochs=30, learning_rate=0.03),
        )
        self.assertLess(metrics["corrective_nll_after"], metrics["corrective_nll_before"])
        self.assertLess(metrics["fallback_nll_after"], metrics["fallback_nll_before"])
        self.assertEqual(metrics["local_sft_optimizer_steps"], 30.0)
        for name, parameter in model.encoder.state_dict().items():
            self.assertTrue(torch.equal(parameter, encoder_before[name]), name)
        for name, parameter in model.value_head.state_dict().items():
            self.assertTrue(torch.equal(parameter, value_head_before[name]), name)
        with torch.no_grad():
            values_after = model(observations)[1]
        self.assertTrue(torch.equal(values_before, values_after))

    def test_empty_dataset_skips_update_exactly(self) -> None:
        torch.manual_seed(14)
        model = ActorCritic(3, 2, hidden_size=8)
        before = copy.deepcopy(model.state_dict())
        metrics = local_sft_update(
            model,
            LocalSFTDataset(),
            LocalSFTConfig(epochs=5),
        )
        self.assertEqual(metrics["local_sft_optimizer_steps"], 0.0)
        for value in metrics.values():
            self.assertTrue(torch.isfinite(torch.tensor(value)))
        for name, parameter in model.state_dict().items():
            self.assertTrue(torch.equal(parameter, before[name]), name)


if __name__ == "__main__":
    unittest.main()
