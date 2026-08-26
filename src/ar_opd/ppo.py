"""Decision-boundary PPO with primitive Teacher steps excluded from actor actions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn
from torch.distributions import Categorical

from ar_opd.core import DecisionRecord, EpisodeRollout
from ar_opd.models import ActorCritic


@dataclass(frozen=True)
class PPOConfig:
    gamma: float = 0.97
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.01
    max_grad_norm: float = 1.0
    epochs: int = 4

    def __post_init__(self) -> None:
        if not 0.0 < self.gamma <= 1.0:
            raise ValueError("gamma must be in (0, 1]")
        if not 0.0 <= self.gae_lambda <= 1.0:
            raise ValueError("gae_lambda must be in [0, 1]")
        if self.clip_ratio < 0.0:
            raise ValueError("clip_ratio must be non-negative")
        if self.value_coefficient < 0.0 or self.entropy_coefficient < 0.0:
            raise ValueError("loss coefficients must be non-negative")
        if self.max_grad_norm <= 0.0:
            raise ValueError("max_grad_norm must be positive")
        if self.epochs < 1:
            raise ValueError("epochs must be positive")


@dataclass(frozen=True)
class PPOBatch:
    observations: torch.Tensor
    actions: torch.Tensor
    old_log_probs: torch.Tensor
    old_values: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    actor_mask: torch.Tensor


@dataclass(frozen=True)
class PPOLosses:
    total: torch.Tensor
    actor: torch.Tensor
    value: torch.Tensor
    entropy: torch.Tensor
    clip_fraction: torch.Tensor
    actor_rows: int


def _macro_reward(
    episode: EpisodeRollout,
    decision: DecisionRecord,
    gamma: float,
) -> float:
    rows = episode.transitions[decision.transition_start : decision.transition_stop]
    if len(rows) != decision.duration:
        raise ValueError("decision transition range is inconsistent")
    return sum((gamma**index) * row.net_reward for index, row in enumerate(rows))


def build_batch(
    episodes: Sequence[EpisodeRollout],
    model: ActorCritic,
    config: PPOConfig,
) -> PPOBatch:
    """Build an SMDP batch independently per episode.

    F primitive steps are folded into a duration-aware macro reward. The actor
    action is always the sampled Student proposal that influenced the gate,
    never an executed Teacher action.
    """

    if not episodes or not any(episode.decisions for episode in episodes):
        raise ValueError("cannot build a PPO batch without decisions")
    device = model.device
    observations: list[tuple[float, ...]] = []
    actions: list[int] = []
    old_log_probs: list[float] = []
    old_values: list[float] = []
    advantages: list[float] = []

    for episode in episodes:
        if not episode.decisions:
            continue
        episode_advantages = [0.0] * len(episode.decisions)
        last_advantage = 0.0
        for index in range(len(episode.decisions) - 1, -1, -1):
            decision = episode.decisions[index]
            rows = episode.transitions[
                decision.transition_start : decision.transition_stop
            ]
            if not rows:
                raise ValueError("a decision cannot have an empty transition range")
            final_row = rows[-1]
            ended = final_row.terminated or final_row.truncated
            if ended:
                next_value = 0.0
                continuation = 0.0
            elif index + 1 < len(episode.decisions):
                next_value = episode.decisions[index + 1].student_proposal.value
                continuation = 1.0
            else:
                next_value = model.value(final_row.next_observation)
                continuation = 0.0

            duration_discount = config.gamma**decision.duration
            delta = (
                _macro_reward(episode, decision, config.gamma)
                + duration_discount * next_value
                - decision.student_proposal.value
            )
            last_advantage = (
                delta
                + duration_discount
                * config.gae_lambda
                * continuation
                * last_advantage
            )
            episode_advantages[index] = last_advantage

        for decision, advantage in zip(
            episode.decisions, episode_advantages, strict=True
        ):
            observations.append(decision.observation)
            actions.append(decision.student_proposal.action)
            old_log_probs.append(decision.student_proposal.log_prob)
            old_values.append(decision.student_proposal.value)
            advantages.append(advantage)

    observations_tensor = torch.tensor(observations, dtype=torch.float32, device=device)
    old_values_tensor = torch.tensor(old_values, dtype=torch.float32, device=device)
    advantages_tensor = torch.tensor(advantages, dtype=torch.float32, device=device)
    return PPOBatch(
        observations=observations_tensor,
        actions=torch.tensor(actions, dtype=torch.long, device=device),
        old_log_probs=torch.tensor(old_log_probs, dtype=torch.float32, device=device),
        old_values=old_values_tensor,
        advantages=advantages_tensor,
        returns=advantages_tensor + old_values_tensor,
        actor_mask=torch.ones(len(actions), dtype=torch.bool, device=device),
    )


def ppo_losses(model: ActorCritic, batch: PPOBatch, config: PPOConfig) -> PPOLosses:
    logits, values = model(batch.observations)
    if bool(batch.actor_mask.any()):
        actor_logits = logits[batch.actor_mask]
        actor_actions = batch.actions[batch.actor_mask]
        old_log_probs = batch.old_log_probs[batch.actor_mask]
        actor_advantages = batch.advantages[batch.actor_mask]
        if actor_advantages.numel() > 1:
            standard_deviation = actor_advantages.std(unbiased=False)
            if float(standard_deviation) > 1e-8:
                actor_advantages = (
                    actor_advantages - actor_advantages.mean()
                ) / (standard_deviation + 1e-8)
        distribution = Categorical(logits=actor_logits)
        new_log_probs = distribution.log_prob(actor_actions)
        log_ratio = new_log_probs - old_log_probs
        if not bool(torch.isfinite(log_ratio).all()):
            raise FloatingPointError("non-finite PPO log probability ratio")
        ratio = log_ratio.exp()
        if not bool(torch.isfinite(ratio).all()):
            raise FloatingPointError("PPO probability ratio overflowed")
        unclipped = ratio * actor_advantages
        clipped = torch.clamp(
            ratio, 1.0 - config.clip_ratio, 1.0 + config.clip_ratio
        ) * actor_advantages
        actor_loss = -torch.minimum(unclipped, clipped).mean()
        entropy = distribution.entropy().mean()
        clip_fraction = ((ratio - 1.0).abs() > config.clip_ratio).float().mean()
        actor_rows = int(batch.actor_mask.sum().item())
    else:
        # Depending on values keeps the critic graph valid while leaving the
        # actor head gradient as None, so Adam momentum cannot move it.
        zero = values.sum() * 0.0
        actor_loss = zero
        entropy = zero
        clip_fraction = zero.detach()
        actor_rows = 0

    value_loss = nn.functional.mse_loss(values, batch.returns)
    total = (
        actor_loss
        + config.value_coefficient * value_loss
        - config.entropy_coefficient * entropy
    )
    return PPOLosses(
        total=total,
        actor=actor_loss,
        value=value_loss,
        entropy=entropy,
        clip_fraction=clip_fraction,
        actor_rows=actor_rows,
    )


def ppo_update(
    model: ActorCritic,
    optimizer: torch.optim.Optimizer,
    batch: PPOBatch,
    config: PPOConfig,
) -> dict[str, float]:
    totals: dict[str, float] = {
        "loss": 0.0,
        "actor_loss": 0.0,
        "value_loss": 0.0,
        "entropy": 0.0,
        "clip_fraction": 0.0,
    }
    skip_update = not bool(batch.actor_mask.any()) and config.value_coefficient == 0.0
    for _ in range(config.epochs):
        losses = ppo_losses(model, batch, config)
        if not bool(torch.isfinite(losses.total)):
            raise FloatingPointError("non-finite PPO loss")
        if not skip_update:
            optimizer.zero_grad(set_to_none=True)
            losses.total.backward()
            nn.utils.clip_grad_norm_(
                model.parameters(), config.max_grad_norm, error_if_nonfinite=True
            )
            optimizer.step()
        totals["loss"] += float(losses.total.detach())
        totals["actor_loss"] += float(losses.actor.detach())
        totals["value_loss"] += float(losses.value.detach())
        totals["entropy"] += float(losses.entropy.detach())
        totals["clip_fraction"] += float(losses.clip_fraction.detach())

    for key in totals:
        totals[key] /= config.epochs
    totals["actor_rows"] = float(batch.actor_mask.sum().item())
    totals["critic_rows"] = float(batch.actor_mask.numel())
    return totals
