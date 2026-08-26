import copy
import unittest

import torch

from ar_opd.core import (
    ActionSource,
    DecisionRecord,
    EpisodeRollout,
    OptionCandidate,
    OptionKind,
    StudentProposal,
    Transition,
)
from ar_opd.models import ActorCritic
from ar_opd.ppo import PPOBatch, PPOConfig, build_batch, ppo_losses, ppo_update


def make_batch(teacher_actions: tuple[int, int], teacher_returns=(0.2, -0.4)) -> PPOBatch:
    return PPOBatch(
        observations=torch.tensor(
            [[0.1, 0.0, 1.0], [0.2, 1.0, 0.8], [0.2, 0.0, 0.7]],
            dtype=torch.float32,
        ),
        actions=torch.tensor([0, *teacher_actions], dtype=torch.long),
        old_log_probs=torch.tensor([-0.5, 0.0, 0.0], dtype=torch.float32),
        old_values=torch.zeros(3),
        advantages=torch.tensor([1.0, 25.0, -30.0]),
        returns=torch.tensor([1.0, *teacher_returns], dtype=torch.float32),
        actor_mask=torch.tensor([True, False, False]),
    )


def gradients(model: ActorCritic) -> dict[str, torch.Tensor | None]:
    return {
        name: None if parameter.grad is None else parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
    }


def one_decision_episode(
    rewards: tuple[float, ...],
    *,
    proposal_action: int = 0,
    actual_action: int = 1,
    terminated: bool = True,
    truncated: bool = False,
) -> EpisodeRollout:
    option = OptionKind.TEACHER_RECOVERY
    transitions = [
        Transition(
            decision_id=0,
            observation=(0.0, 0.0, 1.0),
            action=actual_action,
            next_observation=(0.5, 0.0, 0.5),
            env_reward=reward,
            query_cost=0.0,
            execution_cost=0.0,
            terminated=terminated and index == len(rewards) - 1,
            truncated=truncated and index == len(rewards) - 1,
            source=ActionSource.TEACHER,
            selected_option=option,
        )
        for index, reward in enumerate(rewards)
    ]
    candidate = OptionCandidate(
        kind=option,
        actions=tuple(actual_action for _ in rewards),
        preview_steps=len(rewards),
        estimated_task_value=0.0,
        query_cost=0.0,
        execution_cost=0.0,
        terminated=terminated,
        truncated=truncated,
    )
    decision = DecisionRecord(
        decision_id=0,
        observation=(0.0, 0.0, 1.0),
        student_proposal=StudentProposal(
            action=proposal_action, log_prob=-0.5, value=0.0
        ),
        probed=True,
        candidates=(candidate,),
        selected_option=option,
        transition_start=0,
        transition_stop=len(transitions),
    )
    return EpisodeRollout(transitions=transitions, decisions=[decision])


class MaskedPPOTest(unittest.TestCase):
    def test_teacher_actions_have_no_direct_actor_gradient(self) -> None:
        torch.manual_seed(3)
        first = ActorCritic(3, 2, hidden_size=8)
        second = copy.deepcopy(first)
        config = PPOConfig(value_coefficient=0.0, entropy_coefficient=0.0)

        first_loss = ppo_losses(first, make_batch((0, 1)), config)
        second_loss = ppo_losses(second, make_batch((1, 0)), config)
        first_loss.actor.backward()
        second_loss.actor.backward()

        self.assertEqual(first_loss.actor_rows, 1)
        self.assertTrue(torch.equal(first_loss.actor, second_loss.actor))
        for name, first_gradient in gradients(first).items():
            second_gradient = gradients(second)[name]
            if first_gradient is None:
                self.assertIsNone(second_gradient, name)
            else:
                self.assertTrue(torch.equal(first_gradient, second_gradient), name)

    def test_teacher_returns_train_the_critic(self) -> None:
        torch.manual_seed(4)
        model = ActorCritic(3, 2, hidden_size=8)
        config = PPOConfig()
        baseline = ppo_losses(model, make_batch((0, 1), (0.2, -0.4)), config)
        changed = ppo_losses(model, make_batch((0, 1), (5.0, -4.0)), config)
        self.assertFalse(torch.equal(baseline.value, changed.value))

    def test_teacher_only_update_does_not_move_actor_head_with_adam_momentum(self) -> None:
        torch.manual_seed(5)
        model = ActorCritic(3, 2, hidden_size=8)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        ppo_update(
            model,
            optimizer,
            make_batch((0, 1)),
            PPOConfig(value_coefficient=0.0, entropy_coefficient=0.0, epochs=1),
        )
        actor_before = copy.deepcopy(model.actor_head.state_dict())
        batch = make_batch((0, 1))
        teacher_only = PPOBatch(
            observations=batch.observations[1:],
            actions=batch.actions[1:],
            old_log_probs=batch.old_log_probs[1:],
            old_values=batch.old_values[1:],
            advantages=batch.advantages[1:],
            returns=batch.returns[1:],
            actor_mask=torch.zeros(2, dtype=torch.bool),
        )
        ppo_update(model, optimizer, teacher_only, PPOConfig(epochs=1))
        for name, parameter in model.actor_head.state_dict().items():
            self.assertTrue(torch.equal(parameter, actor_before[name]), name)

        all_before = copy.deepcopy(model.state_dict())
        ppo_update(
            model,
            optimizer,
            teacher_only,
            PPOConfig(value_coefficient=0.0, epochs=1),
        )
        for name, parameter in model.state_dict().items():
            self.assertTrue(torch.equal(parameter, all_before[name]), name)

    def test_smdp_return_discounts_recovery_steps_and_separates_episodes(self) -> None:
        model = ActorCritic(3, 2, hidden_size=8)
        config = PPOConfig(gamma=0.5, gae_lambda=1.0)
        recovery = one_decision_episode((-0.03, 1.0), proposal_action=0, actual_action=1)
        separate = one_decision_episode((2.0,), proposal_action=1, actual_action=0)
        batch = build_batch((recovery, separate), model, config)

        self.assertEqual(batch.actions.tolist(), [0, 1])
        self.assertAlmostEqual(float(batch.returns[0]), 0.47, places=6)
        self.assertAlmostEqual(float(batch.returns[1]), 2.0, places=6)

    def test_truncation_bootstraps_to_zero(self) -> None:
        torch.manual_seed(6)
        model = ActorCritic(3, 2, hidden_size=8)
        with torch.no_grad():
            model.value_head.weight.zero_()
            model.value_head.bias.fill_(999.0)
        episode = one_decision_episode(
            (0.25,), terminated=False, truncated=True
        )
        batch = build_batch((episode,), model, PPOConfig())
        self.assertAlmostEqual(float(batch.returns[0]), 0.25, places=6)
        self.assertEqual(batch.actions.tolist(), [0])


if __name__ == "__main__":
    unittest.main()
