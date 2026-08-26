"""Shared student actor and PPO value model."""

from __future__ import annotations

import torch
from torch import nn

from ar_opd.core import StudentProposal


class ActorCritic(nn.Module):
    def __init__(self, observation_size: int, action_size: int, hidden_size: int = 32) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(observation_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )
        self.actor_head = nn.Linear(hidden_size, action_size)
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.encoder(observations)
        return self.actor_head(features), self.value_head(features).squeeze(-1)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def act(
        self,
        observation: tuple[float, ...],
        *,
        deterministic: bool = False,
        generator: torch.Generator | None = None,
    ) -> StudentProposal:
        tensor = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            logits, value = self(tensor)
            log_probs = torch.log_softmax(logits, dim=-1)
            if deterministic:
                action = logits.argmax(dim=-1)
            else:
                probabilities = torch.softmax(logits, dim=-1)
                action = torch.multinomial(probabilities, 1, generator=generator).squeeze(-1)
            selected_log_prob = log_probs.gather(1, action.unsqueeze(-1)).squeeze(-1)
        return StudentProposal(
            action=int(action.item()),
            log_prob=float(selected_log_prob.item()),
            value=float(value.item()),
        )

    def value(self, observation: tuple[float, ...]) -> float:
        tensor = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            _, value = self(tensor)
        return float(value.item())
