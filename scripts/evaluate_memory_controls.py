"""Frozen paired cue diagnostic on selected held-out geometries."""

import argparse
import copy
import hashlib
import inspect
import json
import math
import random
import time
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from carryrl.agent.actions import PhysicalAction
from carryrl.envs.atomic import AtomicEnv, _identity, atomic_level_banks, atomic_semantic_level_id
from carryrl.rewards.visual import verify_visible_outcome
from carryrl.training.run import atomic_json
from carryrl.training.sequence import (
    FrozenFrameCache,
    _episode_instruction,
    load_sequence_checkpoint,
    tensor_hash,
)
from carryrl.training.sequence_config import SequenceRunConfig

MODES = ("normal", "neutral_wait")


def direction(action):
    """Use the game's physical aliases and opposing-key cancellation."""
    keys = set(action.keys)
    horizontal = int(bool(keys & {"right", "d"})) - int(bool(keys & {"left", "a"}))
    return "right" if horizontal > 0 else "left" if horizontal < 0 else "neutral"


def _supports_nll(agent):
    evaluate = getattr(agent, "evaluate_episode", None)
    if evaluate is None:
        return False
    parameters = inspect.signature(evaluate).parameters
    return "executed_actions" in parameters or any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values())


def _run_episode(agent, cache, config, *, split, index, cue, mode, max_steps, first_choose_nll):
    seed = 1_000_000 + index
    instruction = _episode_instruction(config, "memory", seed)
    env = AtomicEnv("memory", split=split, image_size=config.env_image_size,
                    max_steps=max_steps, visual_config=config.visual_config, difficulty=1.0,
                    level_bank_seed=config.level_bank_seed)
    try:
        agent.reset_episode(instruction=instruction)
        _, origin = env.reset(seed=seed, options={"level_index": index})
        # At full difficulty _level aliases the cached bank. Never edit that dictionary.
        env._level = copy.deepcopy(env._level)
        env._level["cue"] = cue  # The same field controls visible cue and correct final door.
        env._frame = env._draw()
        frame = env.render()
        expected = "left" if cue == 0 else "right"
        runtime = atomic_semantic_level_id("memory", env._level)
        unlock = env._level["unlock"]
        prefix, executed, decisions = [], [], []
        first_choose = None
        wait_outputs = wait_commands = 0
        for step in range(max_steps):
            # Ground-truth phase stays in the evaluator; the actor receives RGB patches only.
            phase = "Wait" if env._ticks < unlock else "Choose"
            patches = cache.encode(frame)
            proposed = agent.act_from_patches(patches, deterministic=True).action
            if not isinstance(proposed, PhysicalAction):
                raise TypeError("actor must return a physical action")
            if first_choose is None:
                prefix.append(patches.detach().cpu())
            actual = PhysicalAction() if mode == "neutral_wait" and phase == "Wait" else proposed
            if phase == "Wait":
                wait_outputs += int(proposed != PhysicalAction())
                wait_commands += int(actual != PhysicalAction())
            if phase == "Choose" and first_choose is None:
                first_choose = {"step": step, "direction": direction(proposed),
                                "correct": direction(proposed) == expected,
                                "proposed_action": proposed.to_dict(), "correct_action_nll": None}
                if first_choose_nll:
                    # Replace only the final label. All prior inputs remain actually executed.
                    labels = [*executed, PhysicalAction(keys=(expected,))]
                    replay = agent.evaluate_episode(torch.stack(prefix), labels,
                                                    instruction=instruction,
                                                    executed_actions=[*executed, actual])
                    nll = -float(replay["log_prob"][-1])
                    if not math.isfinite(nll):
                        raise FloatingPointError("first-Choose correct-action NLL is nonfinite")
                    first_choose["correct_action_nll"] = nll
                prefix.clear()
            agent.observe_action(actual)
            decisions.append({"step": step, "phase": phase,
                              "proposed": proposed.to_dict(), "executed": actual.to_dict()})
            executed.append(actual)
            frame, _, terminated, truncated, _ = env.step(actual)
            if terminated or truncated:
                break
        else:
            raise RuntimeError("memory diagnostic exceeded its episode budget")
        outcome = verify_visible_outcome(frame)
        return {"mode": mode, "level_index": index, "seed": seed,
                "origin_bank_level_id": origin["level_id"],
                "origin_semantic_level_id": origin["semantic_level_id"],
                "origin_runtime_fingerprint": origin["runtime_fingerprint"],
                "counterfactual_runtime_fingerprint": runtime,
                "counterfactual_level_id": _identity(env._level),
                "cue": cue, "expected_direction": expected, "unlock_ticks": unlock,
                "cue_ticks": env._level["cue_ticks"], "instruction": instruction,
                "presentation": origin["presentation"], "first_choose": first_choose,
                "steps": step + 1, "terminated": bool(terminated), "truncated": bool(truncated),
                "nonneutral_wait_outputs": wait_outputs, "nonneutral_wait_commands": wait_commands,
                "visible_outcome": outcome, "decisions": decisions}
    finally:
        env.close()


def _summarize(records):
    pairs = [records[index:index + 2] for index in range(0, len(records), 2)]
    choices = [row["first_choose"] for row in records if row["first_choose"] is not None]
    nll = [choice["correct_action_nll"] for choice in choices
           if choice["correct_action_nll"] is not None]
    return {"episodes": len(records), "geometries": len(pairs),
            "first_choose_observed": len(choices),
            "first_choose_correct": sum(choice["correct"] for choice in choices),
            "paired_both_first_choose_correct": sum(all(
                row["first_choose"] is not None and row["first_choose"]["correct"]
                for row in pair) for pair in pairs),
            "cue_flip_changes_chosen_direction": sum(
                all(row["first_choose"] is not None for row in pair)
                and {row["first_choose"]["direction"] for row in pair} == {"left", "right"}
                for pair in pairs),
            "visible_wins": sum(row["visible_outcome"]["success"] for row in records),
            "visible_failures": sum(row["visible_outcome"]["failure"] for row in records),
            "timeouts": sum(row["truncated"] for row in records),
            "nonneutral_wait_outputs": sum(row["nonneutral_wait_outputs"] for row in records),
            "nonneutral_wait_commands": sum(row["nonneutral_wait_commands"] for row in records),
            "first_choose_correct_action_nll_mean": sum(nll) / len(nll) if nll else None,
            "first_choose_correct_action_nll_count": len(nll)}


def _selected_indices(config, split, levels):
    if split not in {"validation", "test"}:
        raise ValueError("memory diagnostic requires a held-out validation or test split")
    banks = atomic_level_banks("memory", config.level_bank_seed)
    train_ids = {atomic_semantic_level_id("memory", level) for level in banks["train"]}
    eligible = [index for index, level in enumerate(banks[split])
                if all(atomic_semantic_level_id("memory", {**level, "cue": cue}) not in train_ids
                       for cue in (0, 1))]
    # Historical validation selects the original prefix; never silently change its cohort.
    if split == "validation":
        count = len(banks[split]) if levels is None else levels
        if type(count) is not int or not 1 <= count <= len(banks[split]):
            raise ValueError(f"levels must select 1..{len(banks[split])} {split} geometries")
        indices = list(range(count))
        if any(index not in eligible for index in indices):
            raise ValueError("validation cue alternatives collide with training geometries")
        return banks[split], indices
    count = len(eligible) if levels is None else levels
    if type(count) is not int or not 1 <= count <= len(eligible):
        raise ValueError(f"levels must select 1..{len(eligible)} train-disjoint test geometries")
    return banks[split], eligible[:count]


def evaluate_memory_controls(agent, reward, config, *, split="validation", levels=None,
                             max_steps=80, first_choose_nll=True, on_progress=None):
    """Evaluate both cue values per geometry in each mode without updating any tensors."""
    bank, indices = _selected_indices(config, split, levels)
    levels = len(indices)
    if type(max_steps) is not int or not 1 <= max_steps <= 80:
        raise ValueError("max_steps must be in 1..80")
    if any(parameter.device.type != "cpu" for parameter in agent.parameters()):
        raise ValueError("memory diagnostic requires a CPU actor")
    agent_hash, reward_hash, bank_hash = tensor_hash(agent), tensor_hash(reward), _identity(bank)
    agent_mode, reward_mode = agent.training, reward.training
    rng, python_rng, numpy_rng = (torch.random.get_rng_state(), random.getstate(),
                                  np.random.get_state())
    nll_supported = _supports_nll(agent)
    started = time.monotonic()
    records = []
    cache = FrozenFrameCache(agent, capacity=512, max_bytes=256 * 1024**2)
    try:
        agent.eval()
        reward.eval()
        with torch.no_grad():
            for mode in MODES:
                for index in indices:
                    for cue in (0, 1):
                        row = _run_episode(agent, cache, config, split=split, index=index, cue=cue,
                                           mode=mode, max_steps=max_steps,
                                           first_choose_nll=first_choose_nll and nll_supported)
                        records.append(row)
                        if on_progress is not None:
                            on_progress({"completed": len(records), "total": levels * 4,
                                         "mode": mode, "level_index": index, "cue": cue})
    finally:
        agent.reset_episode()
        agent.train(agent_mode)
        reward.train(reward_mode)
        torch.random.set_rng_state(rng)
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
        if tensor_hash(agent) != agent_hash or tensor_hash(reward) != reward_hash:
            raise RuntimeError("memory diagnostic mutated policy or reward tensors")
        if _identity(bank) != bank_hash:
            raise RuntimeError("memory diagnostic mutated the cached held-out bank")
        cache.assert_unchanged()
    report = {"diagnostic": "paired_memory_cues", "split": split, "difficulty": 1.0,
            "created_utc": datetime.now(UTC).isoformat(),
            "elapsed_seconds": time.monotonic() - started,
            "level_bank_seed": config.level_bank_seed, "bank_sha256": bank_hash,
            "agent_sha256": agent_hash, "reward_sha256": reward_hash,
            "policy_and_reward_frozen": True, "max_steps": max_steps,
            "protocol": f"Each {split} geometry has both runtime cue/target alternatives; "
                        f"{levels * 2} matched episodes per mode at {levels} levels. "
                        "Deterministic RGB/history/prose "
                        "actor; neutral_wait executes neutral input whenever the visible phase is "
                        "Wait and supplies that actual input as previous action. Metadata is "
                        "evaluator-only. This paired diagnostic is not "
                        "the normal full held-out test.",
            "first_choose_nll": ("correct physical-action label with unchanged executed prefix"
                                 if first_choose_nll and nll_supported else
                                 "disabled" if not first_choose_nll else
                                 "unsupported: actor has no executed_actions replay API"),
            "summaries": {mode: _summarize([row for row in records if row["mode"] == mode])
                          for mode in MODES},
            "vision_cache": cache.statistics(), "records": records}
    if split == "test":
        _, eligible = _selected_indices(config, split, None)
        report["geometry_selection"] = {
            "bank_count": len(bank), "eligible_count": len(eligible),
            "selected_level_indices": indices,
            "excluded_level_indices": [index for index in range(len(bank))
                                       if index not in eligible],
            "exclusion_reason": "Either runtime cue alternative matches a training semantic ID",
            "scope": "Paired cue diagnostic over train-disjoint test geometries; "
                     "excluded geometries are not evaluated or replaced",
        }
    return report


def _checkpoint_hashes(checkpoint):
    return {str(path.relative_to(checkpoint)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(checkpoint.rglob("*")) if path.is_file()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--levels", type=int, default=None,
                        help="Geometry count; default all train-disjoint geometries in the split")
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--skip-first-choose-nll", action="store_true")
    args = parser.parse_args()
    config = replace(SequenceRunConfig.load(args.config), device="cpu")
    _selected_indices(config, args.split, args.levels)
    checkpoint = args.checkpoint.resolve(strict=True)
    before = _checkpoint_hashes(checkpoint)
    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(config.torch_threads)
        agent, _, reward, data = load_sequence_checkpoint(checkpoint, config, evaluation_only=True)
        report = evaluate_memory_controls(
            agent, reward, config, split=args.split, levels=args.levels, max_steps=args.max_steps,
            first_choose_nll=not args.skip_first_choose_nll,
            on_progress=lambda event: print(json.dumps(event), flush=True))
    finally:
        torch.set_num_threads(threads)
        if _checkpoint_hashes(checkpoint) != before:
            raise RuntimeError("memory diagnostic modified checkpoint files")
    report.update(checkpoint=str(checkpoint), checkpoint_sha256=before,
                  weights_revision=data["weights_revision"])
    atomic_json(args.output, report)
    print(json.dumps(report["summaries"], indent=2))


if __name__ == "__main__":
    main()
