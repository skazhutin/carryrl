"""Strict settings for complete-episode generalist PPO experiments."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from carryrl.agent.model import AgentConfig
from carryrl.envs.atomic_suite import TASKS
from carryrl.rewards.composition import (
    TEXT_CONTROL_V2_VERSION,
    TEXT_CONTROL_VERSION,
    RewardCompositionConfig,
)

if TYPE_CHECKING:
    from carryrl.envs.presentation import PresentationConfig


@dataclass(frozen=True)
class TaskCurriculumConfig:
    """Pinned admission for the independently scheduled TRAIN stages."""

    admission_path: str
    admission_sha256: str

    def __post_init__(self):
        if not isinstance(self.admission_path, str) or not self.admission_path.strip():
            raise ValueError("curriculum admission_path must be nonempty")
        if (not isinstance(self.admission_sha256, str)
                or len(self.admission_sha256) != 64
                or any(c not in "0123456789abcdef" for c in self.admission_sha256)):
            raise ValueError("curriculum admission_sha256 must be a SHA256 digest")


@dataclass(frozen=True)
class SequenceRunConfig:
    seed: int = 0
    level_bank_seed: int = 0
    device: str = "cpu"
    agent: AgentConfig = field(default_factory=AgentConfig)
    tasks: tuple[str, ...] = TASKS
    total_rollouts: int = 2
    episodes_per_task: int = 1
    max_episode_steps: int = 80
    env_image_size: int = 96
    n_epochs: int = 2
    learning_rate: float = 0.0001
    weight_decay: float = 0.01
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.0001
    max_grad_norm: float = 0.5
    eval_episodes_per_task: int = 2
    eval_every_rollouts: int = 1
    checkpoint_every_rollouts: int = 1
    checkpoint_keep_raw: int = 0
    reset_optimizer_on_warm_start: bool = False
    minimum_free_disk_mib: int = 0
    torch_threads: int = 4
    max_wall_seconds: float = 300.0
    reward_bootstrap_episodes_per_task: int = 8
    reward_validation_episodes_per_task: int = 2
    reward_epochs: int = 20
    reward_confidence_threshold: float = 0.6
    motor_initialization: str = "legacy"
    ppo_minibatch_episodes: int = 0
    ppo_microbatch_episodes: int = 0
    ppo_task_balance: bool = False
    target_kl: float = 0.0
    eval_stochastic: bool = False
    eval_seed: int = 80001
    eval_all_levels: bool = False
    evaluate_initial: bool = False
    task_episode_multipliers: dict[str, int] = field(default_factory=dict)
    curriculum_difficulty: float | None = None
    evaluation_difficulty: float | None = None
    evaluation_canonical: bool = False
    evaluation_pixel_art: bool = False
    curriculum_warmup_rollouts: int = 0
    visual_config: PresentationConfig | None = None
    reward_model_mode: str = "legacy_hud"
    reward_checkpoint: str | None = None
    vision_cache_frames: int = 0
    rollout_memory_budget_mib: int = 0
    rollout_feature_storage: str = "memory"
    rollout_disk_budget_mib: int = 0
    episode_instructions: dict[str, tuple[str, ...]] = field(default_factory=dict)
    screen_text_language: str | None = None
    training_backend: str = "torch"
    reward_composition: RewardCompositionConfig = field(default_factory=RewardCompositionConfig)
    advantage_std_floor: float = 1e-8
    ppo_normalization: str = "minibatch"
    gradient_probe_every: int = 0
    text_reward_checkpoint: str | None = None
    text_reward_sha256: str | None = None
    control_schedule_rollouts: int = 0
    task_curriculum: TaskCurriculumConfig | dict | None = None

    def __post_init__(self) -> None:
        if isinstance(self.task_curriculum, dict):
            object.__setattr__(self, "task_curriculum",
                               TaskCurriculumConfig(**self.task_curriculum))
        if self.task_curriculum is not None and not isinstance(
            self.task_curriculum, TaskCurriculumConfig
        ):
            raise ValueError("task_curriculum must be TaskCurriculumConfig or null")
        if isinstance(self.reward_composition, dict):
            object.__setattr__(self, "reward_composition",
                               RewardCompositionConfig.from_dict(self.reward_composition))
        if not isinstance(self.reward_composition, RewardCompositionConfig):
            raise ValueError("reward_composition must be a RewardCompositionConfig or JSON object")
        if isinstance(self.agent, dict):
            object.__setattr__(self, "agent", AgentConfig.from_dict(self.agent))
        if not isinstance(self.agent, AgentConfig):
            raise ValueError("agent must be an AgentConfig or its JSON object")
        if self.visual_config is not None:
            from carryrl.envs.presentation import PresentationConfig

            if isinstance(self.visual_config, dict):
                object.__setattr__(self, "visual_config", PresentationConfig(**self.visual_config))
            if not isinstance(self.visual_config, PresentationConfig):
                raise ValueError("visual_config must be a PresentationConfig or its JSON object")
        ints = ("seed", "total_rollouts", "episodes_per_task", "max_episode_steps",
                "env_image_size",
                "n_epochs", "eval_episodes_per_task", "eval_every_rollouts",
                "checkpoint_every_rollouts", "torch_threads",
                "reward_bootstrap_episodes_per_task", "reward_validation_episodes_per_task",
                "reward_epochs")
        for name in ints:
            value = getattr(self, name)
            if type(value) is not int or value < (0 if name == "seed" else 1):
                raise ValueError(f"{name} must be a valid positive integer (seed permits zero)")
        if self.seed >= 2**32:
            raise ValueError("seed must be below 2**32")
        for name in ("ppo_minibatch_episodes", "ppo_microbatch_episodes",
                     "curriculum_warmup_rollouts", "eval_seed",
                     "level_bank_seed", "minimum_free_disk_mib",
                     "vision_cache_frames", "checkpoint_keep_raw", "rollout_memory_budget_mib",
                     "rollout_disk_budget_mib"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.rollout_feature_storage not in {"memory", "mmap"}:
            raise ValueError("rollout_feature_storage must be memory or mmap")
        if ((self.rollout_feature_storage == "mmap") != (self.rollout_disk_budget_mib > 0)):
            raise ValueError("mmap rollout storage requires a positive disk budget only")
        for name in ("gradient_probe_every", "control_schedule_rollouts"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.ppo_normalization not in {"minibatch", "rollout"}:
            raise ValueError("ppo_normalization must be minibatch or rollout")
        if self.ppo_normalization == "rollout" and not self.ppo_task_balance:
            raise ValueError("rollout normalization requires equal task balancing")
        if (type(self.advantage_std_floor) not in (int, float)
                or not 0 < self.advantage_std_floor < float("inf")):
            raise ValueError("advantage_std_floor must be positive and finite")
        if self.task_curriculum is not None:
            if not self.reward_composition.finite_horizon:
                raise ValueError("task curriculum requires classified finite-horizon outcomes")
            if self.curriculum_difficulty is not None or self.curriculum_warmup_rollouts:
                raise ValueError("task curriculum cannot combine with scalar curriculum")
            if self.evaluation_difficulty != 1.0:
                raise ValueError("task curriculum requires explicit full evaluation_difficulty=1")
        if (self.text_reward_checkpoint is None) != (self.text_reward_sha256 is None):
            raise ValueError("text reward requires both checkpoint and manifest SHA256")
        if self.text_reward_checkpoint is not None:
            if (not isinstance(self.text_reward_checkpoint, str)
                    or not self.text_reward_checkpoint.strip()
                    or not isinstance(self.text_reward_sha256, str)
                    or len(self.text_reward_sha256) != 64
                    or any(c not in "0123456789abcdef" for c in self.text_reward_sha256)):
                raise ValueError("invalid text reward checkpoint identity")
        if self.checkpoint_keep_raw == 1:
            raise ValueError("checkpoint_keep_raw must be zero (disabled) or at least two")
        if self.eval_seed >= 2**32:
            raise ValueError("eval_seed must be below 2**32")
        if self.level_bank_seed >= 2**32:
            raise ValueError("level_bank_seed must be below 2**32")
        if self.vision_cache_frames > 1024:
            raise ValueError("vision_cache_frames must be at most 1024")
        for name in ("ppo_task_balance", "eval_stochastic", "eval_all_levels", "evaluate_initial",
                     "reset_optimizer_on_warm_start", "evaluation_canonical",
                     "evaluation_pixel_art"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be a boolean")
        if self.evaluation_pixel_art:
            if not self.evaluation_canonical or not (
                self.visual_config is not None and self.visual_config.pixel_art
            ):
                raise ValueError("evaluation_pixel_art requires canonical evaluation and "
                                 "pixel-art training presentation")
            if self.visual_config.style_split is not None:
                raise ValueError("evaluation_pixel_art requires automatic held-out style_split")
        if self.motor_initialization not in {"legacy", "balanced"}:
            raise ValueError("motor_initialization must be legacy or balanced")
        if self.reward_model_mode not in {"legacy_hud", "screen_v2"}:
            raise ValueError("reward_model_mode must be legacy_hud or screen_v2")
        if self.reward_checkpoint is not None and (
            not isinstance(self.reward_checkpoint, str) or not self.reward_checkpoint.strip()
        ):
            raise ValueError("reward_checkpoint must be a nonempty path string or null")
        if not isinstance(self.task_episode_multipliers, dict) or any(
            task not in self.tasks or type(multiplier) is not int or multiplier < 1
            for task, multiplier in self.task_episode_multipliers.items()
        ):
            raise ValueError(
                "task_episode_multipliers must map configured tasks to positive integers"
            )
        object.__setattr__(self, "task_episode_multipliers", dict(self.task_episode_multipliers))
        if self.curriculum_difficulty is not None and (
            type(self.curriculum_difficulty) not in (int, float)
            or not 0 <= self.curriculum_difficulty <= 1
        ):
            raise ValueError("curriculum_difficulty must be in [0,1] or null")
        if self.curriculum_warmup_rollouts and self.curriculum_difficulty is None:
            raise ValueError("curriculum warmup requires an initial curriculum_difficulty")
        if self.evaluation_difficulty is not None and (
            type(self.evaluation_difficulty) not in (int, float)
            or not 0 <= self.evaluation_difficulty <= 1
        ):
            raise ValueError("evaluation_difficulty must be in [0,1] or null")
        if self.device not in {"cpu", "mps"}:
            raise ValueError("device must be explicitly cpu or mps")
        if self.training_backend not in {"torch", "mlx"}:
            raise ValueError("training_backend must be explicitly torch or mlx")
        if self.ppo_microbatch_episodes and self.training_backend != "mlx":
            raise ValueError("ppo_microbatch_episodes requires training_backend=mlx")
        if self.training_backend == "mlx" and self.device != "cpu":
            raise ValueError("MLX training requires device=cpu for frozen Torch encoders/rollouts")
        if not 1 <= self.max_episode_steps <= min(128, 2 * self.agent.segment_length):
            raise ValueError("complete episodes must fit at most two segments and 128 decisions")
        if not isinstance(self.tasks, (list, tuple)) or not self.tasks:
            raise ValueError("tasks must be a nonempty sequence")
        if len(set(self.tasks)) != len(self.tasks) or any(task not in TASKS for task in self.tasks):
            raise ValueError("tasks must be unique supported task names")
        object.__setattr__(self, "tasks", tuple(self.tasks))
        if not isinstance(self.episode_instructions, dict):
            raise ValueError("episode_instructions must map configured tasks to instruction lists")
        instructions = {}
        for task, variants in self.episode_instructions.items():
            if task not in self.tasks:
                raise ValueError("episode_instructions keys must be configured tasks")
            if not isinstance(variants, (list, tuple)) or not variants:
                raise ValueError("episode_instructions values must be nonempty lists or tuples")
            if any(not isinstance(text, str) or not text.strip() or len(text) > 8192
                   for text in variants):
                raise ValueError("episode instructions must be nonempty strings of at most 8192 "
                                 "characters")
            instructions[task] = tuple(variants)
        if self.screen_text_language is not None and (
            not isinstance(self.screen_text_language, str) or not self.screen_text_language
            or any(not part or part.strip() != part
                   for part in self.screen_text_language.split("+"))
        ):
            raise ValueError("screen_text_language must name explicit OCR languages or null")
        if getattr(self.agent, "text_encoder", "none") == "none":
            if instructions or self.screen_text_language is not None:
                raise ValueError("episode instructions and screen OCR require an enabled "
                                 "text_encoder")
        elif set(instructions) != set(self.tasks) and (
            instructions or self.screen_text_language is None
        ):
            raise ValueError("enabled text_encoder requires episode_instructions for every task")
        object.__setattr__(self, "episode_instructions", instructions)
        text_reward = self.reward_composition.version in (
            TEXT_CONTROL_VERSION, TEXT_CONTROL_V2_VERSION)
        if text_reward:
            if self.agent.text_encoder == "none" or set(instructions) != set(self.tasks):
                raise ValueError(
                    "text control reward requires explicit instructions for every task")
            if self.text_reward_checkpoint is None or self.control_schedule_rollouts < 1:
                raise ValueError("text control reward requires a pinned model and schedule horizon")
            if (self.reward_composition.version == TEXT_CONTROL_V2_VERSION
                    and self.control_schedule_rollouts != 64):
                raise ValueError("text reward v2 requires the fixed 64-cycle schedule declaration")
        elif self.text_reward_checkpoint is not None or self.control_schedule_rollouts:
            raise ValueError("text reward settings require text control composition")
        if self.env_image_size < 64:
            raise ValueError("env_image_size must be >= 64")
        if min(self.reward_bootstrap_episodes_per_task,
               self.reward_validation_episodes_per_task) < 2:
            raise ValueError("reward bootstrap/validation need at least two episodes per task")
        for name in ("learning_rate", "weight_decay", "gamma", "gae_lambda", "clip_range",
                     "value_coef",
                     "entropy_coef", "max_grad_norm", "max_wall_seconds",
                     "reward_confidence_threshold", "target_kl"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not 0 <= value < float("inf"):
                raise ValueError(f"{name} must be a finite nonnegative number")
        if min(self.learning_rate, self.max_grad_norm, self.max_wall_seconds) <= 0:
            raise ValueError("learning rate, gradient norm and wall budget must be positive")
        if not 0 < self.gamma <= 1 or not 0 < self.gae_lambda <= 1 or not 0 < self.clip_range < 1:
            raise ValueError("invalid discount, GAE or PPO clipping range")
        if not 0.5 <= self.reward_confidence_threshold <= 1:
            raise ValueError("reward confidence threshold must be in [0.5,1]")

        self.reward_composition.validate_context(
            gamma=self.gamma, horizon=self.max_episode_steps, tasks=self.tasks,
            score_confidence_threshold=self.reward_confidence_threshold)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> SequenceRunConfig:
        if not isinstance(value, dict):
            raise ValueError("sequence config must be an object")
        try:
            return cls(**value)
        except TypeError as error:
            raise ValueError(f"invalid sequence config: {error}") from error

    @classmethod
    def load(cls, path: Path | str) -> SequenceRunConfig:
        return cls.from_dict(json.loads(Path(path).read_text()))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
