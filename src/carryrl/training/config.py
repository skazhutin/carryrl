"""Strict, recorded experiment settings for the first engineering candidate."""

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RunConfig:
    seed: int = 0
    device: str = "cpu"
    total_timesteps: int = 4096
    n_steps: int = 256
    batch_size: int = 64
    n_epochs: int = 4
    learning_rate: float = 0.0003
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    image_size: int = 64
    embedding_size: int = 128
    max_episode_steps: int = 64
    eval_interval: int = 1024
    eval_seeds: tuple[int, ...] = tuple(range(1001, 1013))
    max_wall_seconds: float = 600.0
    torch_threads: int = 1

    def __post_init__(self) -> None:
        ints = (
            "seed", "total_timesteps", "n_steps", "batch_size", "n_epochs", "image_size",
            "embedding_size", "max_episode_steps", "eval_interval", "torch_threads",
        )
        for name in ints:
            value = getattr(self, name)
            if type(value) is not int or value < (0 if name == "seed" else 1):
                raise ValueError(f"{name} must be an integer >= {0 if name == 'seed' else 1}")
        if self.device not in {"cpu", "mps"}:
            raise ValueError("device must be explicit: cpu or mps")
        if self.seed >= 2**32:
            raise ValueError("seed must be smaller than 2**32")
        if self.image_size < 48:
            raise ValueError("image_size must be >= 48 for the visual environment")
        if self.n_steps < 2 or self.batch_size < 2:
            raise ValueError("PPO rollout and minibatch must contain at least two samples")
        if self.n_steps % self.batch_size:
            raise ValueError("n_steps must be a multiple of batch_size")
        if self.total_timesteps % self.n_steps or self.eval_interval % self.n_steps:
            raise ValueError("total_timesteps and eval_interval must be multiples of n_steps")
        for name in ("learning_rate", "gamma", "gae_lambda", "clip_range", "ent_coef",
                     "max_wall_seconds"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not 0 <= value < float("inf"):
                raise ValueError(f"{name} must be a finite nonnegative number")
        if self.learning_rate == 0 or self.max_wall_seconds == 0:
            raise ValueError("learning_rate and max_wall_seconds must be positive")
        if not 0 < self.gamma <= 1 or not 0 < self.gae_lambda <= 1 or not 0 < self.clip_range < 1:
            raise ValueError("gamma/gae_lambda must be in (0,1], clip_range in (0,1)")
        if not isinstance(self.eval_seeds, (tuple, list)) or not self.eval_seeds:
            raise ValueError("eval_seeds must be a nonempty list of integers")
        if any(type(seed) is not int or not 0 <= seed < 2**32 for seed in self.eval_seeds):
            raise ValueError("eval_seeds must contain integers in [0, 2**32)")
        if len(set(self.eval_seeds)) != len(self.eval_seeds):
            raise ValueError("eval_seeds must be unique")
        object.__setattr__(self, "eval_seeds", tuple(self.eval_seeds))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RunConfig":
        if not isinstance(data, dict):
            raise ValueError("configuration must be a JSON object")
        unknown = set(data) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f"unknown configuration fields: {sorted(unknown)}")
        return cls(**data)

    @classmethod
    def load(cls, path: str | Path) -> "RunConfig":
        with Path(path).open() as stream:
            return cls.from_dict(json.load(stream))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
