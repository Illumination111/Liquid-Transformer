"""Multi-categorical PPO with GAE and a learned value baseline."""

from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    def __init__(self, state_dim: int, num_blocks: int, num_actions: int, hidden: int = 64):
        super().__init__()
        self.num_blocks, self.num_actions = num_blocks, num_actions
        self.features = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh()
        )
        self.actor = nn.Linear(hidden, num_blocks * num_actions)
        self.critic = nn.Linear(hidden, 1)

    def forward(self, state: torch.Tensor) -> tuple[Categorical, torch.Tensor]:
        features = self.features(state)
        logits = self.actor(features).reshape(*state.shape[:-1], self.num_blocks, self.num_actions)
        return Categorical(logits=logits), self.critic(features).squeeze(-1)

    def evaluate(self, states: torch.Tensor, actions: torch.Tensor):
        distribution, values = self(states)
        return distribution.log_prob(actions).sum(-1), distribution.entropy().sum(-1), values

    @torch.no_grad()
    def act(self, state: torch.Tensor, generator: torch.Generator):
        distribution, value = self(state)
        # A private RNG prevents candidate-training seeds from resetting policy sampling.
        actions = torch.multinomial(distribution.probs, 1, generator=generator).squeeze(-1)
        return actions, distribution.log_prob(actions).sum(-1), value


def compute_gae(rewards, values, dones, bootstrap=0.0, gamma=0.99, gae_lambda=0.95):
    advantages = torch.zeros_like(rewards)
    advantage = torch.zeros((), dtype=rewards.dtype)
    next_value = torch.as_tensor(bootstrap, dtype=rewards.dtype)
    for index in reversed(range(len(rewards))):
        continuation = 1.0 - dones[index]
        delta = rewards[index] + gamma * next_value * continuation - values[index]
        advantage = delta + gamma * gae_lambda * continuation * advantage
        advantages[index] = advantage
        next_value = values[index]
    return advantages, advantages + values


class PPO:
    def __init__(
        self,
        policy: ActorCritic,
        lr=3e-4,
        clip=0.2,
        entropy_coef=0.01,
        value_coef=0.5,
        update_epochs=4,
        gamma=0.99,
        gae_lambda=0.95,
    ):
        self.policy = policy
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
        self.clip, self.entropy_coef, self.value_coef = clip, entropy_coef, value_coef
        self.update_epochs, self.gamma, self.gae_lambda = update_epochs, gamma, gae_lambda

    def update(self, rollout: list[dict], bootstrap: float = 0.0) -> dict[str, float]:
        if not rollout:
            raise ValueError("PPO requires a nonempty rollout")
        states = torch.stack([row["state"] for row in rollout]).detach()
        actions = torch.stack([row["actions"] for row in rollout]).detach()
        old_log_probs = torch.stack([row["log_prob"] for row in rollout]).detach()
        values = torch.stack([row["value"] for row in rollout]).detach()
        rewards = torch.tensor([row["reward"] for row in rollout], dtype=torch.float32)
        dones = torch.tensor([row["done"] for row in rollout], dtype=torch.float32)
        advantages, returns = compute_gae(
            rewards, values, dones, bootstrap, self.gamma, self.gae_lambda
        )
        if len(advantages) > 1 and advantages.std(unbiased=False) > 1e-8:
            advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False)
        stats = {}
        for _ in range(self.update_epochs):
            log_probs, entropy, predictions = self.policy.evaluate(states, actions)
            ratios = (log_probs - old_log_probs).exp()
            clipped = ratios.clamp(1 - self.clip, 1 + self.clip)
            policy_loss = -torch.minimum(ratios * advantages, clipped * advantages).mean()
            value_loss = (predictions - returns).square().mean()
            loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy.mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite PPO loss")
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()
            stats = {
                "policy_loss": policy_loss.item(),
                "value_loss": value_loss.item(),
                "entropy": entropy.mean().item(),
                "mean_reward": rewards.mean().item(),
            }
        return stats
