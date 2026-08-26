import copy
import math
import unittest
from dataclasses import replace

import torch

from ar_opd.core import OptionKind
from ar_opd.models import ActorCritic
from ar_opd.opd import (
    ActionDistributionAnnotation,
    OPDAnnotationError,
    OPDConfig,
    OPDDataset,
    OPDExample,
    ToyOracleDistributionAnnotator,
    extract_student_only_opd,
    opd_forward_kl,
    opd_update,
    temperature_scale_distribution,
    validate_probability_distribution,
)
from ar_opd.rollout import RolloutCollector, RolloutConfig
from ar_opd.teacher import OracleTeacher
from ar_opd.toy_env import ChainAction, JammedChainConfig, JammedChainEnv


def collect_episode(probe_probability: float = 0.0):
    torch.manual_seed(5)
    model = ActorCritic(3, 2, hidden_size=8)
    env = JammedChainEnv(
        JammedChainConfig(goal_position=2, trap_positions=(1,), max_steps=5)
    )
    episode = RolloutCollector(
        RolloutConfig(probe_probability=probe_probability),
        seed=3,
        torch_generator=torch.Generator().manual_seed(7),
    ).collect_episode(env, model, OracleTeacher(env.config))
    return episode


def example(
    probabilities=(0.9, 0.1), *, proposal_action: int = 1, collection_id: int = 7
) -> OPDExample:
    return OPDExample(
        observation=(0.0, 0.0, 1.0),
        teacher_probabilities=probabilities,
        student_proposal_action=proposal_action,
        collection_id=collection_id,
        episode_index=0,
        decision_id=0,
    )


def dataset(*rows: OPDExample, collection_id: int = 7) -> OPDDataset:
    return OPDDataset(collection_id=collection_id, examples=rows)


class CountingAnnotator:
    def __init__(
        self,
        *,
        action_size: int = 2,
        query_cost: float = 0.0,
        fail_after: int | None = None,
    ) -> None:
        self.action_size = action_size
        self.query_cost = query_cost
        self.fail_after = fail_after
        self.call_count = 0

    def annotate(self, observation):
        self.call_count += 1
        if self.fail_after is not None and self.call_count > self.fail_after:
            raise RuntimeError("annotation backend failed")
        probabilities = tuple(1.0 / self.action_size for _ in range(self.action_size))
        return ActionDistributionAnnotation(probabilities, query_cost=self.query_cost)


class StudentOnlyOPDTest(unittest.TestCase):
    def test_probability_validation_rejects_invalid_targets(self) -> None:
        invalid = (
            (),
            (1.0,),
            (-0.1, 1.1),
            (0.2, 0.3),
            (math.nan, 0.0),
            (math.inf, 0.0),
        )
        for probabilities in invalid:
            with self.subTest(probabilities=probabilities):
                with self.assertRaises((TypeError, ValueError)):
                    validate_probability_distribution(probabilities)
        with self.assertRaises(TypeError):
            validate_probability_distribution((True, 0.0))

    def test_frozen_records_normalize_sequences_and_enforce_dataset_schema(self) -> None:
        annotation = ActionDistributionAnnotation([0.8, 0.2])
        row = example([0.8, 0.2])
        rows = [row]
        normalized = OPDDataset(collection_id=7, examples=rows)
        self.assertIsInstance(annotation.probabilities, tuple)
        self.assertIsInstance(row.teacher_probabilities, tuple)
        self.assertIsInstance(normalized.examples, tuple)

        wrong_collection = example(collection_id=8)
        with self.assertRaisesRegex(ValueError, "collection_id"):
            OPDDataset(collection_id=7, examples=(wrong_collection,))
        three_actions = example((0.6, 0.3, 0.1), proposal_action=2)
        with self.assertRaisesRegex(ValueError, "action dimension"):
            OPDDataset(collection_id=7, examples=(row, three_actions))

    def test_all_episodes_are_validated_before_the_first_annotation(self) -> None:
        valid = collect_episode(probe_probability=0.0)
        invalid = copy.deepcopy(valid)
        invalid.decisions[0] = replace(
            invalid.decisions[0],
            transition_stop=invalid.decisions[0].transition_stop + 1,
        )
        annotator = CountingAnnotator()
        with self.assertRaises(ValueError):
            extract_student_only_opd(
                (valid, invalid),
                annotator,
                expected_action_size=2,
                collection_id=7,
            )
        self.assertEqual(annotator.call_count, 0)

    def test_general_rollout_validator_mutants_are_rejected_without_queries(self) -> None:
        episode = collect_episode(probe_probability=0.0)
        cost_mutant = copy.deepcopy(episode)
        cost_mutant.transitions[0] = replace(
            cost_mutant.transitions[0], query_cost=0.25
        )
        candidate_mutant = copy.deepcopy(episode)
        decision = candidate_mutant.decisions[0]
        candidate = decision.candidates[0]
        wrong_action = 1 - candidate.actions[0]
        candidate_mutant.decisions[0] = replace(
            decision,
            candidates=(replace(candidate, actions=(wrong_action,)),),
        )

        for mutant in (cost_mutant, candidate_mutant):
            annotator = CountingAnnotator()
            with self.subTest(mutant=mutant.transitions[0].query_cost):
                with self.assertRaises(ValueError):
                    extract_student_only_opd(
                        (mutant,),
                        annotator,
                        expected_action_size=2,
                        collection_id=7,
                    )
                self.assertEqual(annotator.call_count, 0)

    def test_wrong_annotator_dimension_is_rejected_before_query(self) -> None:
        annotator = CountingAnnotator(action_size=3)
        with self.assertRaisesRegex(ValueError, "action_size"):
            extract_student_only_opd(
                (collect_episode(),),
                annotator,
                expected_action_size=2,
                collection_id=7,
            )
        self.assertEqual(annotator.call_count, 0)

    def test_annotation_failure_carries_successful_partial_ledger(self) -> None:
        annotator = CountingAnnotator(query_cost=0.25, fail_after=1)
        with self.assertRaises(OPDAnnotationError) as raised:
            extract_student_only_opd(
                (collect_episode(),),
                annotator,
                expected_action_size=2,
                collection_id=7,
            )
        ledger = raised.exception.partial_ledger
        self.assertEqual(ledger.query_count, 1)
        self.assertEqual(ledger.scored_actions, 2)
        self.assertAlmostEqual(ledger.query_cost, 0.25)

    def test_extraction_rejects_any_probed_or_hybrid_rollout(self) -> None:
        episode = collect_episode(probe_probability=1.0)
        self.assertTrue(any(decision.probed for decision in episode.decisions))
        with self.assertRaisesRegex(ValueError, "zero-Teacher|hybrid"):
            extract_student_only_opd(
                (episode,),
                ToyOracleDistributionAnnotator(),
                expected_action_size=2,
                collection_id=7,
            )

    def test_student_only_annotation_cost_is_isolated_from_ppo_rollout(self) -> None:
        episode = collect_episode(probe_probability=0.0)
        before = copy.deepcopy(episode)
        result = extract_student_only_opd(
            (episode,),
            ToyOracleDistributionAnnotator(
                preferred_probability=0.9, query_cost=0.125
            ),
            expected_action_size=2,
            collection_id=7,
        )

        self.assertEqual(len(result.dataset), len(episode.decisions))
        self.assertEqual(result.ledger.query_count, len(episode.decisions))
        self.assertEqual(result.ledger.scored_actions, 2 * len(episode.decisions))
        self.assertAlmostEqual(
            result.ledger.query_cost, 0.125 * len(episode.decisions)
        )
        self.assertEqual(episode, before)
        self.assertEqual(episode.teacher_costs.total_cost, 0.0)
        self.assertTrue(
            all(decision.selected_option is OptionKind.STUDENT for decision in episode.decisions)
        )
        self.assertTrue(
            all(row.query_cost == row.execution_cost == 0.0 for row in episode.transitions)
        )

    def test_toy_oracle_prefers_repair_only_when_jammed(self) -> None:
        annotator = ToyOracleDistributionAnnotator(preferred_probability=0.9)
        clear = annotator.annotate((0.25, 0.0, 0.8)).probabilities
        jammed = annotator.annotate((0.25, 1.0, 0.8)).probabilities
        self.assertEqual(max(range(2), key=clear.__getitem__), int(ChainAction.ADVANCE))
        self.assertEqual(max(range(2), key=jammed.__getitem__), int(ChainAction.REPAIR))
        self.assertTrue(all(0.0 < value < 1.0 for value in clear + jammed))

    def test_temperature_sharpens_and_softens_without_changing_argmax(self) -> None:
        base = (0.9, 0.1)
        cold = temperature_scale_distribution(base, 0.5)
        identity = temperature_scale_distribution(base, 1.0)
        hot = temperature_scale_distribution(base, 2.0)
        self.assertAlmostEqual(identity[0], base[0])
        self.assertGreater(cold[0], base[0])
        self.assertLess(hot[0], base[0])
        self.assertGreater(hot[0], 0.5)
        self.assertEqual(
            [max(range(2), key=rows.__getitem__) for rows in (cold, identity, hot)],
            [0, 0, 0],
        )
        with self.assertRaisesRegex(ValueError, "temperature"):
            temperature_scale_distribution(base, 0.0)

    def test_matching_student_distribution_has_zero_forward_kl(self) -> None:
        model = ActorCritic(3, 2, hidden_size=8)
        with torch.no_grad():
            model.actor_head.weight.zero_()
            model.actor_head.bias.copy_(torch.tensor([math.log(0.8), math.log(0.2)]))
        loss = opd_forward_kl(
            model,
            dataset(example((0.8, 0.2))),
            target_temperature=1.0,
        )
        self.assertAlmostEqual(float(loss.detach()), 0.0, places=6)

    def test_loss_is_forward_not_reverse_kl_and_supports_zero_target(self) -> None:
        model = ActorCritic(3, 2, hidden_size=8)
        with torch.no_grad():
            model.actor_head.weight.zero_()
            model.actor_head.bias.copy_(torch.tensor([math.log(0.6), math.log(0.4)]))
        teacher = torch.tensor([0.9, 0.1])
        student = torch.tensor([0.6, 0.4])
        loss = opd_forward_kl(model, dataset(example(tuple(teacher.tolist()))))
        forward = (teacher * (teacher.log() - student.log())).sum()
        reverse = (student * (student.log() - teacher.log())).sum()
        self.assertAlmostEqual(float(loss.detach()), float(forward), places=6)
        self.assertNotAlmostEqual(float(loss.detach()), float(reverse), places=4)

        zero_target = opd_forward_kl(model, dataset(example((1.0, 0.0))))
        self.assertTrue(bool(torch.isfinite(zero_target)))
        self.assertAlmostEqual(
            float(zero_target.detach()), -math.log(0.6), places=6
        )

    def test_gradient_update_increases_teacher_preferred_action(self) -> None:
        model = ActorCritic(3, 2, hidden_size=8)
        with torch.no_grad():
            model.actor_head.weight.zero_()
            model.actor_head.bias.zero_()
        observation = torch.tensor([[0.0, 0.0, 1.0]])
        before = torch.softmax(model(observation)[0], dim=-1)[0, 0].item()
        metrics = opd_update(
            model,
            dataset(example((0.95, 0.05))),
            OPDConfig(epochs=1, learning_rate=0.2),
            expected_collection_id=7,
        )
        after = torch.softmax(model(observation)[0], dim=-1)[0, 0].item()
        self.assertGreater(after, before)
        self.assertLess(metrics["opd_kl_after"], metrics["opd_kl_before"])
        self.assertEqual(metrics["opd_optimizer_steps"], 1.0)

    def test_actor_head_only_update_leaves_value_bitwise_unchanged(self) -> None:
        torch.manual_seed(13)
        model = ActorCritic(3, 2, hidden_size=8)
        observations = torch.tensor([[0.0, 0.0, 1.0], [0.5, 1.0, 0.5]])
        with torch.no_grad():
            values_before = model(observations)[1].clone()
        encoder_before = copy.deepcopy(model.encoder.state_dict())
        value_head_before = copy.deepcopy(model.value_head.state_dict())

        opd_update(
            model,
            dataset(
                example((0.95, 0.05)),
                OPDExample(
                    observation=(0.5, 1.0, 0.5),
                    teacher_probabilities=(0.05, 0.95),
                    student_proposal_action=0,
                    collection_id=7,
                    episode_index=0,
                    decision_id=1,
                ),
            ),
            OPDConfig(epochs=5, learning_rate=0.1),
            expected_collection_id=7,
        )
        with torch.no_grad():
            values_after = model(observations)[1]
        self.assertTrue(torch.equal(values_before, values_after))
        for name, value in model.encoder.state_dict().items():
            self.assertTrue(torch.equal(value, encoder_before[name]), name)
        for name, value in model.value_head.state_dict().items():
            self.assertTrue(torch.equal(value, value_head_before[name]), name)

    def test_student_proposal_action_does_not_enter_forward_kl(self) -> None:
        torch.manual_seed(17)
        model = ActorCritic(3, 2, hidden_size=8)
        first = opd_forward_kl(
            model, dataset(example(proposal_action=0))
        )
        second = opd_forward_kl(
            model, dataset(example(proposal_action=1))
        )
        self.assertTrue(torch.equal(first, second))

    def test_empty_dataset_skips_with_finite_metrics(self) -> None:
        model = ActorCritic(3, 2, hidden_size=8)
        before = copy.deepcopy(model.state_dict())
        metrics = opd_update(
            model,
            OPDDataset(collection_id=7),
            OPDConfig(epochs=3),
            expected_collection_id=7,
        )
        self.assertTrue(all(math.isfinite(value) for value in metrics.values()))
        self.assertEqual(metrics["opd_examples"], 0.0)
        self.assertEqual(metrics["opd_optimizer_steps"], 0.0)
        for name, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, before[name]), name)

    def test_update_rejects_a_stale_collection_before_mutation(self) -> None:
        model = ActorCritic(3, 2, hidden_size=8)
        before = copy.deepcopy(model.state_dict())
        with self.assertRaisesRegex(ValueError, "collection_id"):
            opd_update(
                model,
                dataset(example()),
                OPDConfig(epochs=1),
                expected_collection_id=8,
            )
        for name, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, before[name]), name)


if __name__ == "__main__":
    unittest.main()
