"""Versioned composition of learned visual predictions and executed input costs.

The opt-in finite-episode potential telescopes to a policy-independent constant
for a fixed initial prediction. It does not establish optimality with uncertain
learned outcomes, nor does the trajectory cost bound guarantee expected returns.
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, fields, replace
from typing import Any

from carryrl.agent.actions import KEY_VOCAB, MOUSE_BUTTONS, PhysicalAction

LEGACY_REWARD_VERSION = "carryrl.reward.delta.v1"
OUTCOME_POTENTIAL_VERSION = "carryrl.reward.outcome_potential_cost.v1"
TEXT_CONTROL_VERSION = "carryrl.reward.outcome_potential_text.v1"
TEXT_CONTROL_V2_VERSION = "carryrl.reward.outcome_potential_text.v2"
MAX_INPUT_UNITS = len(KEY_VOCAB) + len(MOUSE_BUTTONS) + 3


def _number(value: Any, name: str, *, minimum: float, maximum: float | None = None):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite non-boolean number")
    if value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f"{name} outside supported range")
    return float(value)


@dataclass(frozen=True)
class OutcomePotentialConfig:
    success_reward: float = 1.0
    failure_penalty: float = 1.0
    progress_weight: float = 0.2
    time_cost: float = 0.001
    input_cost: float = 0.001
    event_confidence_threshold: float = 0.9
    goal_choice_utility: str | None = None

    def __post_init__(self):
        for name in ("success_reward", "failure_penalty", "progress_weight",
                     "time_cost", "input_cost"):
            object.__setattr__(self, name, _number(getattr(self, name), name, minimum=0))
        if self.success_reward <= 0:
            raise ValueError("success_reward must be positive")
        threshold = _number(self.event_confidence_threshold, "event_confidence_threshold",
                            minimum=0.5, maximum=1)
        if threshold <= 0.5:
            raise ValueError("event_confidence_threshold must exceed 0.5")
        object.__setattr__(self, "event_confidence_threshold", threshold)
        if self.goal_choice_utility not in (None, "completion_only"):
            raise ValueError("only explicit goal_choice_utility='completion_only' is supported")


@dataclass(frozen=True)
class TextControlConfig(OutcomePotentialConfig):
    control_schedule: str = "zero"

    def __post_init__(self):
        super().__post_init__()
        if self.control_schedule not in ("zero", "temporary", "permanent"):
            raise ValueError("control_schedule must be zero, temporary or permanent")


@dataclass(frozen=True)
class TextControlV2Config(TextControlConfig):
    unsupported_alpha: float = 0.01

    def __post_init__(self):
        super().__post_init__()
        object.__setattr__(self, "unsupported_alpha", _number(
            self.unsupported_alpha, "unsupported_alpha", minimum=0, maximum=0.01))


def control_schedule_coefficient(schedule: str, cycle: int, total_cycles: int) -> float:
    """Zero-based rollout index; choose once when resetting a complete episode."""
    if schedule not in ("zero", "temporary", "permanent"):
        raise ValueError("invalid control schedule")
    if type(cycle) is not int or cycle < 0 or type(total_cycles) is not int or total_cycles < 1:
        raise ValueError("schedule requires nonnegative cycle and positive total_cycles")
    if schedule == "zero":
        return 0.0
    if schedule == "permanent":
        return 1.0
    fraction = cycle / total_cycles
    return max(0.0, min(1.0, 2 - 4 * fraction))


def control_schedule_coefficient_v2(
    schedule: str, cycle: int, total_cycles: int | None = None,
) -> float:
    """Absolute one-based cycle schedule, independent of a run's planned length."""
    if schedule not in ("zero", "temporary", "permanent"):
        raise ValueError("invalid control schedule")
    if type(cycle) is not int or cycle < 1:
        raise ValueError("v2 schedule requires a positive one-based cycle")
    if total_cycles is not None and (type(total_cycles) is not int or total_cycles < 1):
        raise ValueError("total_cycles must be a positive integer")
    if schedule == "zero":
        return 0.0
    if schedule == "permanent":
        return 1.0
    return min(1.0, max(0.0, (64 - cycle) / 32))


@dataclass(frozen=True)
class RewardCompositionConfig:
    version: str = LEGACY_REWARD_VERSION
    parameters: (
        OutcomePotentialConfig | TextControlConfig | TextControlV2Config | dict | None
    ) = None

    def __post_init__(self):
        if self.version == LEGACY_REWARD_VERSION:
            if self.parameters is not None:
                raise ValueError("legacy reward composition does not accept parameters")
        elif self.version in (OUTCOME_POTENTIAL_VERSION, TEXT_CONTROL_VERSION,
                              TEXT_CONTROL_V2_VERSION):
            parameter_type = {TEXT_CONTROL_VERSION: TextControlConfig,
                              TEXT_CONTROL_V2_VERSION: TextControlV2Config}.get(
                                  self.version, OutcomePotentialConfig)
            parameters = self.parameters
            if parameters is None:
                parameters = parameter_type()
            elif isinstance(parameters, dict):
                unknown = set(parameters) - {field.name for field in fields(parameter_type)}
                if unknown:
                    raise ValueError(f"unknown reward composition parameters: {sorted(unknown)}")
                parameters = parameter_type(**parameters)
            if type(parameters) is not parameter_type:
                raise ValueError("parameters must be an OutcomePotentialConfig or dict")
            object.__setattr__(self, "parameters", parameters)
        else:
            raise ValueError(f"unsupported reward composition version: {self.version!r}")

    @property
    def finite_horizon(self) -> bool:
        return self.version in (OUTCOME_POTENTIAL_VERSION, TEXT_CONTROL_VERSION,
                                TEXT_CONTROL_V2_VERSION)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, record: Mapping) -> "RewardCompositionConfig":
        if not isinstance(record, Mapping):
            raise ValueError("reward composition config must be a mapping")
        unknown = set(record) - {"version", "parameters"}
        if unknown:
            raise ValueError(f"unknown reward composition fields: {sorted(unknown)}")
        return cls(**record)

    def validate_context(
        self, *, gamma: float, horizon: int, tasks: Sequence[str],
        score_confidence_threshold: float,
    ) -> None:
        gamma = _number(gamma, "gamma", minimum=0, maximum=1)
        if gamma <= 0:
            raise ValueError("gamma must be in (0,1]")
        if type(horizon) is not int or horizon < 1:
            raise ValueError("horizon must be a positive non-boolean integer")
        _number(score_confidence_threshold, "score_confidence_threshold", minimum=0, maximum=1)
        if not self.finite_horizon:
            return
        parameters = self.parameters
        if "goal_choice" in tasks and parameters.goal_choice_utility != "completion_only":
            raise ValueError("goal_choice requires explicit goal_choice_utility='completion_only'")
        discounted_win = parameters.success_reward * gamma ** (horizon - 1)
        discount_sum = float(horizon) if gamma == 1 else -math.expm1(
            horizon * math.log(gamma)) / (1 - gamma)
        maximum_cost = (parameters.time_cost + parameters.input_cost) * discount_sum
        if (self.version == TEXT_CONTROL_VERSION
                and parameters.control_schedule != "zero"):
            # Both positive and negative episode budgets can separate two trajectories.
            maximum_cost += 0.02 * parameters.success_reward
        elif (self.version == TEXT_CONTROL_V2_VERSION
              and parameters.control_schedule != "zero"):
            maximum_cost += (0.01 + parameters.unsupported_alpha) * parameters.success_reward
        if discounted_win <= maximum_cost:
            raise ValueError("discounted final success must exceed all maximum trajectory costs")


def reward_composition_identity(
    config: RewardCompositionConfig, *, gamma: float, horizon: int,
    score_confidence_threshold: float,
) -> dict:
    config.validate_context(gamma=gamma, horizon=horizon, tasks=(),
                            score_confidence_threshold=score_confidence_threshold)
    identity = {
        **config.to_dict(),
        "score_confidence_threshold": float(score_confidence_threshold),
        "finite_horizon": config.finite_horizon,
    }
    if config.finite_horizon:
        identity.update(gamma=float(gamma), horizon=horizon)
        identity["semantics"] = {
            "outcome": "confident_learned_event_at_boundary_only",
            "potential": "score_divided_by_5_bounded_0_1",
            "unknown": "carry_last_trusted_potential_charge_costs",
            "deadline": "terminated_or_truncated_zero_terminal_potential_no_bootstrap",
            "time_cost": "per_executed_command_including_neutral",
            "input_cost": "active_keys_buttons_nonzero_dx_dy_wheel_divided_by_max_input_units",
            "max_input_units": MAX_INPUT_UNITS,
            "terminal_uncertainty": "abort_rollout_before_update",
            "dominance": "individual_recognized_win_over_nonwin_with_same_initial_potential",
        }
    else:
        identity["semantics"] = {
            "outcome": "success_probability_delta_minus_failure_probability_delta",
            "potential": "unscaled_undiscounted_score_delta",
            "unknown": "zero_entire_transition_if_either_prediction_unknown_or_untrusted",
            "deadline": "legacy_bootstrap",
            "time_cost": 0.01, "input_cost": 0.0,
            "score_confidence": "caller_score_trusted_flag_default_true",
        }
    if config.version == TEXT_CONTROL_VERSION:
        identity["semantics"]["text_control"] = {
            "source": "frozen_learned_instruction_table_no_engine_metadata",
            "positive_per_class": "min(.001R,.01R/supported_alias_groups)",
            "negative_per_class": "min(.001R,.01R/unsupported_alias_groups)",
            "episode_sign_caps": ".01R_each_before_schedule",
            "deduplication": "episode_first_use_alias_class_including_vetoed_and_terminal",
            "command": "unsupported_veto_positive_at_most_one_positive_class",
            "terminal": "no_positive_penalties_allowed_preserve_visual_invalidity",
            "schedule": config.parameters.control_schedule,
        }
    elif config.version == TEXT_CONTROL_V2_VERSION:
        identity["semantics"]["text_control"] = {
            "source": "frozen_factorized_instruction_table_no_engine_metadata",
            "base_per_group": ".0005R_once_when_supported_and_deduplication_known",
            "mode_upgrade_per_group": ".0015R_once_after_clean_physical_mode_witness",
            "positive_caps": "actual_paid_.002R_per_group_.01R_per_episode",
            "mode_witness": ("held_same_candidate_2_steps_press_1_step_then_real_release_"
                             "relative_nonzero"),
            "negative": "one_recurring_unsupported_alpha_R_over_horizon_per_active_step",
            "unsupported_alpha": config.parameters.unsupported_alpha,
            "command": "unsupported_veto_one_positive_group_upgrade_priority_then_vocabulary",
            "terminal": ("no_positive_or_release_witness_penalties_allowed_"
                         "preserve_visual_invalidity"),
            "schedule": config.parameters.control_schedule,
            "schedule_clock": "one_based_absolute_cycles_1_32_full_33_63_decay_64_zero",
        }
    return identity


def legacy_transition(previous: Mapping, current: Mapping) -> float:
    """Preserve the historical arithmetic order and caller-controlled trust gate."""
    if any(not prediction.get("known", True)
           or not prediction.get("_score_trusted", True) for prediction in (previous, current)):
        return 0.0
    return float(current["score"] - previous["score"]
                 + current["success_probability"] - previous["success_probability"]
                 - current["failure_probability"] + previous["failure_probability"] - 0.01)


@dataclass(frozen=True)
class RewardStep:
    total: float
    outcome: float
    progress: float
    time_cost: float
    input_cost: float
    score_trusted: bool
    event_status: str
    valid_for_learning: bool = True
    invalid_reason: str | None = None
    control_bonus: float = 0.0
    unsupported_penalty: float = 0.0
    control_events: tuple[dict, ...] = ()

    def to_dict(self) -> dict:
        """Components are signed contributions to total."""
        return asdict(self)


def _probability(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1


def classify_learned_event(prediction: Mapping, threshold: float) -> str:
    """Classify learned event probabilities; independent verifier labels never enter."""
    if not prediction.get("known", True):
        return "unknown"
    success, failure = (prediction.get(name) for name in (
        "success_probability", "failure_probability"))
    if not _probability(success) or not _probability(failure):
        return "uncertain"
    low = round(1 - threshold, 15)
    if success >= threshold and failure >= threshold:
        return "conflict"
    if success >= threshold and failure <= low:
        return "success"
    if failure >= threshold and success <= low:
        return "failure"
    if success <= low and failure <= low:
        return "play"
    return "uncertain"


def _trusted_potential(prediction: Mapping, threshold: float) -> tuple[float, bool]:
    score, confidence = prediction.get("score"), prediction.get("score_confidence")
    trusted = (bool(prediction.get("known", True))
               and type(score) in (int, float) and math.isfinite(score) and 0 <= score <= 5
               and _probability(confidence) and confidence >= threshold)
    return (float(score) / 5, True) if trusted else (0.0, False)


class RewardComposer:
    """One stateful composer per complete episode, from its initial learned readout."""

    def __init__(
        self, config: RewardCompositionConfig, *, gamma: float, horizon: int,
        score_confidence_threshold: float, initial_prediction: Mapping,
        control_table=None, control_coefficient: float = 0.0,
    ):
        config.validate_context(gamma=gamma, horizon=horizon, tasks=(),
                                score_confidence_threshold=score_confidence_threshold)
        self.config = config
        self.gamma = float(gamma)
        self.horizon = horizon
        self.score_confidence_threshold = float(score_confidence_threshold)
        self._previous = dict(initial_prediction)
        self._potential, self.initial_score_trusted = _trusted_potential(
            initial_prediction, score_confidence_threshold)
        self.initial_event_status = (
            classify_learned_event(initial_prediction, config.parameters.event_confidence_threshold)
            if config.finite_horizon else "legacy")
        self._control_table = None
        self._control_coefficient = _number(control_coefficient, "control_coefficient",
                                            minimum=0, maximum=1)
        self._control_seen: set[str] = set()
        self._control_positive = self._control_negative = 0.0
        self._control_v2 = None
        if config.version == TEXT_CONTROL_VERSION:
            from carryrl.rewards.control_text import ControlPredictionTable
            if not isinstance(control_table, ControlPredictionTable):
                raise ValueError("text control reward requires a frozen learned prediction table")
            if config.parameters.control_schedule == "zero" and self._control_coefficient != 0:
                raise ValueError("zero schedule requires zero coefficient")
            if config.parameters.control_schedule == "permanent" and self._control_coefficient != 1:
                raise ValueError("permanent schedule requires coefficient one")
            self._control_table = control_table
        elif config.version == TEXT_CONTROL_V2_VERSION:
            from carryrl.rewards.control_text_factorized import FactorizedControlPredictionTable
            from carryrl.rewards.control_text_reward import TextControlLedger
            if type(control_table) is not FactorizedControlPredictionTable:
                raise ValueError(
                    "v2 text control reward requires a frozen factorized prediction table")
            if config.parameters.control_schedule == "zero" and self._control_coefficient != 0:
                raise ValueError("zero schedule requires zero coefficient")
            if config.parameters.control_schedule == "permanent" and self._control_coefficient != 1:
                raise ValueError("permanent schedule requires coefficient one")
            self._control_table = control_table
            self._control_v2 = TextControlLedger(
                control_table, reward=config.parameters.success_reward,
                coefficient=self._control_coefficient, horizon=horizon,
                unsupported_alpha=config.parameters.unsupported_alpha)
        elif control_table is not None or self._control_coefficient != 0:
            raise ValueError("text controls require the opt-in text reward version")
        self._steps = 0
        self._ended = False

    @property
    def finite_horizon(self) -> bool:
        return self.config.finite_horizon

    def step(
        self, current_prediction: Mapping, action: PhysicalAction, *,
        terminated: bool, truncated: bool,
    ) -> RewardStep:
        if self._ended or self._steps >= self.horizon:
            raise RuntimeError("reward composer episode has ended; create a new composer")
        if not isinstance(action, PhysicalAction):
            raise TypeError("reward composition requires an executed PhysicalAction")
        if type(terminated) is not bool or type(truncated) is not bool:
            raise ValueError("terminated and truncated must be booleans")
        boundary = terminated or truncated
        if self.finite_horizon and self._steps + 1 == self.horizon and not boundary:
            raise ValueError("finite reward horizon requires an explicit episode boundary")
        if not self.finite_horizon:
            previous = self._previous
            trusted = all(prediction.get("known", True)
                          and prediction.get("_score_trusted", True)
                          for prediction in (previous, current_prediction))
            result = RewardStep(
                total=legacy_transition(previous, current_prediction),
                outcome=float(current_prediction["success_probability"]
                              - previous["success_probability"]
                              - current_prediction["failure_probability"]
                              + previous["failure_probability"]) if trusted else 0.0,
                progress=float(current_prediction["score"] - previous["score"]) if trusted else 0.0,
                time_cost=-0.01 if trusted else 0.0, input_cost=0.0,
                score_trusted=bool(trusted), event_status="legacy",
            )
        else:
            parameters = self.config.parameters
            potential, trusted = _trusted_potential(
                current_prediction, self.score_confidence_threshold)
            if not trusted:
                potential = self._potential
            next_potential = 0.0 if boundary else potential
            progress = parameters.progress_weight * (
                self.gamma * next_potential - self._potential)
            status = classify_learned_event(
                current_prediction, parameters.event_confidence_threshold)
            outcome = 0.0
            invalid_reason = None
            if boundary:
                if status == "success":
                    outcome = parameters.success_reward
                elif status == "failure":
                    outcome = -parameters.failure_penalty
                elif status != "play":
                    invalid_reason = f"terminal_event_{status}"
                elif terminated:
                    invalid_reason = "terminated_without_learned_terminal_event"
            active_units = (len(action.keys) + len(action.mouse_buttons)
                            + int(action.dx != 0) + int(action.dy != 0) + int(action.wheel != 0))
            time_cost = -parameters.time_cost
            input_cost = -parameters.input_cost * active_units / MAX_INPUT_UNITS
            result = RewardStep(
                total=outcome + progress + time_cost + input_cost,
                outcome=outcome, progress=progress, time_cost=time_cost, input_cost=input_cost,
                score_trusted=trusted, event_status=status,
                valid_for_learning=invalid_reason is None, invalid_reason=invalid_reason,
            )
            self._potential = next_potential
        if self._control_table is not None:
            bonus, penalty, events = (
                self._control_v2.step(action, boundary) if self._control_v2 is not None
                else self._control_step(action, boundary))
            result = replace(result, total=result.total + bonus + penalty,
                             control_bonus=bonus, unsupported_penalty=penalty,
                             control_events=events)
        self._previous = dict(current_prediction)
        self._steps += 1
        self._ended = boundary
        return result

    def _state_identity(self):
        return {"config": self.config.to_dict(), "gamma": self.gamma,
                "horizon": self.horizon,
                "score_confidence_threshold": self.score_confidence_threshold,
                "control_coefficient": self._control_coefficient,
                "control_table_identity": self._control_table.identity}

    def state_dict(self) -> dict:
        """JSON-compatible v2 episode state; legacy replay semantics are unchanged."""
        if self._control_v2 is None:
            raise ValueError("state serialization is supported only for text reward v2")
        return {"version": "carryrl.reward_composer_state.v2", "identity": self._state_identity(),
                "previous_prediction": dict(self._previous), "potential": self._potential,
                "initial_score_trusted": self.initial_score_trusted,
                "initial_event_status": self.initial_event_status,
                "steps": self._steps, "ended": self._ended,
                "control": self._control_v2.state_dict()}

    def load_state_dict(self, record: Mapping) -> None:
        """Restore only a compatible frozen table, episode budget and configuration."""
        if self._control_v2 is None:
            raise ValueError("state serialization is supported only for text reward v2")
        expected = {"version", "identity", "previous_prediction", "potential",
                    "initial_score_trusted", "initial_event_status", "steps", "ended", "control"}
        if (not isinstance(record, Mapping) or set(record) != expected
                or record["version"] != "carryrl.reward_composer_state.v2"
                or record["identity"] != self._state_identity()):
            raise ValueError("incompatible v2 reward composer state identity")
        steps, ended = record["steps"], record["ended"]
        if (type(steps) is not int or not 0 <= steps <= self.horizon
                or type(ended) is not bool or (steps == self.horizon and not ended)
                or (steps == 0 and ended)
                or not isinstance(record["previous_prediction"], Mapping)
                or type(record["initial_score_trusted"]) is not bool
                or record["initial_event_status"] not in {
                    "unknown", "uncertain", "conflict", "success", "failure", "play"}):
            raise ValueError("invalid v2 reward composer episode state")
        potential = _number(record["potential"], "potential", minimum=0, maximum=1)
        if ended and potential != 0:
            raise ValueError("ended episode must have zero terminal potential")
        control = self._control_v2.validate_state(record["control"], steps=steps)
        # Validate everything before changing either half of the episode state.
        self._control_v2.load_state_dict(control, steps=steps)
        self._previous = dict(record["previous_prediction"])
        self._potential, self._steps, self._ended = potential, steps, ended
        self.initial_score_trusted = record["initial_score_trusted"]
        self.initial_event_status = record["initial_event_status"]

    def _control_step(self, action, boundary):
        from carryrl.rewards.control_text import executed_candidates
        table = {row.candidate: row for row in self._control_table.predictions}
        rows = [table[candidate] for candidate in executed_candidates(action)]
        unsupported = any(row.status == "unsupported" for row in rows)
        supported_groups = {row.group for row in table.values()
                            if row.status == "supported" and row.reward_eligible}
        unsupported_groups = {row.group for row in table.values()
                              if row.status == "unsupported" and row.reward_eligible}
        reward = self.config.parameters.success_reward
        cap = 0.01 * reward * self._control_coefficient
        positive_unit = min(0.001 * reward, 0.01 * reward / max(1, len(supported_groups)))
        negative_unit = min(0.001 * reward, 0.01 * reward / max(1, len(unsupported_groups)))
        bonus = penalty = 0.0
        events = []
        command_seen = set()
        for row in rows:
            if row.status == "unknown":
                events.append({"candidate": row.candidate, "event": "unknown", "amount": 0.0})
                continue
            if not row.reward_eligible:
                events.append({"candidate": row.candidate,
                               "event": "deduplication_unknown", "amount": 0.0})
                continue
            group = row.group
            if group in command_seen:
                events.append({"candidate": row.candidate, "group": group,
                               "event": "alias_duplicate", "amount": 0.0})
                continue
            command_seen.add(group)
            first = group not in self._control_seen
            self._control_seen.add(group)
            amount = 0.0
            event = "repeat"
            if first and row.status == "unsupported":
                amount = -min(negative_unit * self._control_coefficient,
                              max(0.0, cap - self._control_negative))
                penalty += amount
                self._control_negative -= amount
                event = "unsupported_first_use"
            elif first and row.status == "supported":
                if boundary:
                    event = "terminal_no_bonus"
                elif unsupported:
                    event = "unsupported_veto"
                elif bonus:
                    event = "command_positive_limit"
                else:
                    amount = min(positive_unit * self._control_coefficient,
                                 max(0.0, cap - self._control_positive))
                    bonus += amount
                    self._control_positive += amount
                    event = "supported_first_use"
            events.append({"candidate": row.candidate, "group": group,
                           "event": event, "amount": amount})
        return bonus, penalty, tuple(events)
