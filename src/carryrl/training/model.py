"""Small PPO/CNN candidate; this is an experiment, not an architecture decision."""

import gymnasium as gym
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn

from carryrl.training.config import RunConfig


class SmallVisualEncoder(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 128):
        super().__init__(observation_space, features_dim)
        channels = observation_space.shape[0]
        self.conv = nn.Sequential(
            nn.Conv2d(channels, 16, kernel_size=5, stride=2), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2), nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2), nn.ReLU(), nn.Flatten(),
        )
        with torch.no_grad():
            flattened = self.conv(torch.zeros(1, *observation_space.shape)).shape[1]
        self.projection = nn.Sequential(nn.Linear(flattened, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.projection(self.conv(observations))


def build_model(config: RunConfig, env: gym.Env) -> PPO:
    if config.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but not available; no implicit CPU fallback")
    torch.set_num_threads(config.torch_threads)
    return PPO(
        "CnnPolicy", env, seed=config.seed, device=config.device,
        n_steps=config.n_steps, batch_size=config.batch_size, n_epochs=config.n_epochs,
        learning_rate=config.learning_rate, gamma=config.gamma, gae_lambda=config.gae_lambda,
        clip_range=config.clip_range, ent_coef=config.ent_coef,
        policy_kwargs={
            "features_extractor_class": SmallVisualEncoder,
            "features_extractor_kwargs": {"features_dim": config.embedding_size},
            "net_arch": {"pi": [64], "vf": [64]},
        },
        verbose=0,
    )
