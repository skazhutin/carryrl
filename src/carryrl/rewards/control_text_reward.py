"""Episode-local v2 accounting over a frozen learned factorized control table."""

import copy
import math

from carryrl.rewards.control_text import CANDIDATES, executed_candidates


class TextControlLedger:
    """No action intent or environment state enters physical mode witnesses."""

    def __init__(self, table, *, reward, coefficient, horizon, unsupported_alpha):
        self.table = table
        self.reward = reward
        self.coefficient = coefficient
        self.horizon = horizon
        self.unsupported_alpha = unsupported_alpha
        self.rows = {row.candidate: row for row in table.predictions}
        self.previous = set()
        self.runs = {}
        self.groups = {}
        self.positive = 0.0
        self.negative = 0.0

    def step(self, action, boundary):
        active = set(executed_candidates(action))
        unsupported = any(self.rows[name].status == "unsupported" for name in active)
        penalty = (-self.unsupported_alpha * self.reward / self.horizon * self.coefficient
                   if unsupported else 0.0)
        self.negative -= penalty
        # Each run belongs to an actual physical candidate, never to its alias group.
        next_runs = {}
        for name in active:
            if not name.startswith(("key:", "mouse:")):
                continue
            old = self.runs.get(name)
            next_runs[name] = {"length": 1 if old is None else old["length"] + 1,
                               "clean": not unsupported and (old is None or old["clean"])}
        proposals = []
        events = []
        for index, name in enumerate(CANDIDATES):
            released = name in self.previous and name not in active
            if name not in active and not released:
                continue
            row = self.rows[name]
            event = {"candidate": name, "event": "repeat", "amount": 0.0}
            if row.mode_decision.reason is not None:
                event["mode_reason"] = row.mode_decision.reason
            if row.status == "unknown":
                event["event"] = "unknown"
                event["reason"] = row.applicability_decision.reason
            elif row.status == "unsupported":
                event["event"] = "unsupported_active" if name in active else "release"
            elif not row.reward_eligible:
                event["event"] = "deduplication_unknown"
                reasons = sorted({pair.decision.reason for pair in self.table.pair_decisions
                                  if name in (pair.left, pair.right)
                                  and pair.decision.reason is not None})
                event["reason"] = reasons[0] if reasons else "inconsistent"
                event["pair_reasons"] = reasons
            else:
                group = row.group
                event["group"] = group
                paid = self.groups.get(group, {})
                base = name in active and not paid.get("base_awarded", False)
                reliable_mode = row.mode_decision.accepted and row.mode != "unknown"
                witness = False
                if reliable_mode and not boundary and not unsupported:
                    if row.mode == "relative":
                        witness = name in active and not name.startswith(("key:", "mouse:"))
                    elif row.mode == "held":
                        run = next_runs.get(name)
                        witness = bool(run and run["length"] >= 2 and run["clean"])
                    elif row.mode == "press":
                        run = self.runs.get(name)
                        witness = bool(released and run and run["length"] == 1 and run["clean"])
                upgrade = witness and not paid.get("upgrade_awarded", False)
                if boundary and (base or upgrade):
                    event["event"] = "terminal_no_bonus"
                elif unsupported and (base or upgrade):
                    event["event"] = "unsupported_veto"
                elif base or upgrade:
                    event["event"] = "command_positive_limit"
                    proposals.append((not upgrade, index, event, base, upgrade))
            events.append(event)
        bonus = 0.0
        if proposals and not unsupported and not boundary:
            _, _, event, base, upgrade = min(proposals, key=lambda item: item[:2])
            group = event["group"]
            paid = self.groups.setdefault(group, {"base_awarded": False,
                                                 "upgrade_awarded": False,
                                                 "base_paid": 0.0, "upgrade_paid": 0.0})
            group_remaining = 0.002 * self.reward - paid["base_paid"] - paid["upgrade_paid"]
            remaining = min(max(0.0, 0.01 * self.reward - self.positive),
                            max(0.0, group_remaining))
            for part, requested in (("base", base), ("upgrade", upgrade)):
                nominal = (0.0005 if part == "base" else 0.0015) * self.reward
                amount = min(nominal * self.coefficient, remaining) if requested else 0.0
                if amount > 0:
                    paid[f"{part}_awarded"] = True
                    paid[f"{part}_paid"] += amount
                    remaining -= amount
                    bonus += amount
                event[f"{part}_amount"] = amount
            event.update(event="supported_award" if bonus else "positive_cap_or_zero_schedule",
                         amount=bonus)
            self.positive += bonus
        if unsupported:
            # The penalty is charged once even when several unsupported candidates are active.
            events.append({"event": "unsupported_step_penalty", "amount": penalty})
        self.previous, self.runs = active, next_runs
        return bonus, penalty, tuple(events)

    def state_dict(self):
        return copy.deepcopy({"previous": sorted(self.previous), "runs": self.runs,
                              "groups": self.groups, "positive": self.positive,
                              "negative": self.negative})

    def validate_state(self, record, *, steps):
        if not isinstance(record, dict) or set(record) != {
            "previous", "runs", "groups", "positive", "negative"
        }:
            raise ValueError("invalid v2 control ledger fields")
        previous = record["previous"]
        if (not isinstance(previous, list) or any(not isinstance(name, str) for name in previous)
                or len(previous) != len(set(previous))
                or any(name not in CANDIDATES for name in previous)):
            raise ValueError("invalid previous physical candidates")
        if any(f"{axis}:-" in previous and f"{axis}:+" in previous
               for axis in ("dx", "dy", "wheel")):
            raise ValueError("conflicting previous relative signs")
        runs = record["runs"]
        expected_runs = {name for name in previous if name.startswith(("key:", "mouse:"))}
        if not isinstance(runs, dict) or set(runs) != expected_runs:
            raise ValueError("invalid physical candidate runs")
        for run in runs.values():
            if (not isinstance(run, dict) or set(run) != {"length", "clean"}
                    or type(run["length"]) is not int or not 1 <= run["length"] <= steps
                    or type(run["clean"]) is not bool):
                raise ValueError("invalid physical run witness")
        def amount(value, maximum):
            return (type(value) in (int, float) and math.isfinite(value)
                    and 0 <= value <= maximum + 1e-12)
        groups = record["groups"]
        eligible = {row.group for row in self.rows.values()
                    if row.status == "supported" and row.reward_eligible}
        if not isinstance(groups, dict) or not set(groups) <= eligible:
            raise ValueError("invalid paid alias groups")
        total = 0.0
        for paid in groups.values():
            if not isinstance(paid, dict) or set(paid) != {
                "base_awarded", "upgrade_awarded", "base_paid", "upgrade_paid"
            }:
                raise ValueError("invalid group payment ledger")
            for part, limit in (("base", 0.0005), ("upgrade", 0.0015)):
                value, awarded = paid[f"{part}_paid"], paid[f"{part}_awarded"]
                if (type(awarded) is not bool
                        or not amount(value, limit * self.reward * self.coefficient)
                        or awarded != (value > 0)):
                    raise ValueError("invalid actual paid reward part")
                total += value
        if (not amount(record["positive"], 0.01 * self.reward)
                or not math.isclose(total, record["positive"], rel_tol=0, abs_tol=1e-12)
                or not amount(record["negative"], steps * self.unsupported_alpha * self.reward
                              / self.horizon * self.coefficient)
                or (steps == 0 and (previous or runs or groups))):
            raise ValueError("invalid episode payment bounds")
        return copy.deepcopy(record)

    def load_state_dict(self, record, *, steps):
        record = self.validate_state(record, steps=steps)
        self.previous = set(record["previous"])
        self.runs, self.groups = record["runs"], record["groups"]
        self.positive, self.negative = record["positive"], record["negative"]
