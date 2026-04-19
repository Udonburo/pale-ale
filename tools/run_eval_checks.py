#!/usr/bin/env python3
"""Plan evaluation-factory checks by standing resource tier.

This scaffold is intentionally plan-only. It does not introduce analytical
methods, change Gate12A doctrine, or invoke model execution.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from enum import Enum
from typing import Sequence


class Tier(str, Enum):
    CPU_NIGHTLY = "cpu-nightly"
    L4_SMOKE = "l4-smoke"
    L4_WEEKLY = "l4-weekly"
    SUMMARIZE_EXISTING = "summarize-existing"


@dataclass(frozen=True)
class TierPlan:
    tier: Tier
    intent: str
    resource_posture: str
    planned_actions: tuple[str, ...]
    out_of_scope: tuple[str, ...]
    not_implemented_yet: tuple[str, ...]


TIER_VALUES = tuple(tier.value for tier in Tier)

L4_SMOKE_BOUNDARY = (
    "0.5B fixed family boundary set: transcript_v1, briefing_v1, archive_v1"
)

L4_WEEKLY_SURFACES = (
    "current 3B/4B dense-transformer family-set surfaces under the frozen Gate12A observable contract"
)

# 7B FP32 is not in the L4 mainline. Keep it out of l4-weekly until there is a
# separate resource posture and claim surface for it.
L4_WEEKLY_EXCLUDES_7B_FP32 = "7B FP32"

# Protocol-expanding, quantized, and sidecar candidates are not l4-weekly work.
L4_WEEKLY_EXCLUDED_CANDIDATES = (
    "protocol-expanding candidates",
    "quantized candidates",
    "sidecar candidates",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Print a dry-run evaluation-factory execution plan for one standing tier. "
            "This scaffold does not run expensive jobs or invoke GPU/model execution."
        )
    )
    parser.add_argument(
        "--tier",
        required=True,
        choices=TIER_VALUES,
        help="Standing evaluation tier to plan.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Print the execution plan without running jobs. This is the only supported behavior in this scaffold.",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def plan_cpu_nightly() -> TierPlan:
    return TierPlan(
        tier=Tier.CPU_NIGHTLY,
        intent="fast local checks; manifest / file / structure validation; no GPU expectation",
        resource_posture="CPU-only; cheap; suitable for local or nightly validation",
        planned_actions=(
            "validate expected repository file surfaces and manifest presence",
            "run lightweight structure checks that do not require model execution",
            "emit a compact status summary for later CI or scheduled-run wiring",
        ),
        out_of_scope=(
            "GPU invocation",
            "model execution",
            "Gate12A math or doctrine changes",
        ),
        not_implemented_yet=(
            "manifest traversal",
            "structured status artifact emission",
            "CI scheduling integration",
        ),
    )


def plan_l4_smoke() -> TierPlan:
    return TierPlan(
        tier=Tier.L4_SMOKE,
        intent="small standing smoke lane aligned with the 0.5B boundary set",
        resource_posture="single L4 posture; explicitly cheap and repeatable",
        planned_actions=(
            f"plan checks for {L4_SMOKE_BOUNDARY}",
            "confirm expected fixed-family inputs before any future dispatch",
            "reserve only a narrow smoke-lane path for later cheap model execution wiring",
        ),
        out_of_scope=(
            "full dense-transformer family-set expansion",
            "7B FP32",
            "new analytical methods",
        ),
        not_implemented_yet=(
            "0.5B smoke dispatch",
            "artifact completeness checks for the smoke lane",
            "result summarization handoff",
        ),
    )


def plan_l4_weekly() -> TierPlan:
    return TierPlan(
        tier=Tier.L4_WEEKLY,
        intent="mainline standing lane aligned with the current 3B/4B dense-transformer family-set surfaces",
        resource_posture="single L4 posture for planned weekly work; excludes 7B FP32",
        planned_actions=(
            f"plan checks for {L4_WEEKLY_SURFACES}",
            "confirm the frozen Gate12A observable surface before any future dispatch",
            "keep weekly planning to repeatable dense-transformer family-set surfaces only",
        ),
        out_of_scope=(
            L4_WEEKLY_EXCLUDES_7B_FP32,
            *L4_WEEKLY_EXCLUDED_CANDIDATES,
            "Gate12B promotion",
        ),
        not_implemented_yet=(
            "3B/4B weekly dispatch",
            "L4 runtime budget enforcement",
            "standing summary publication",
        ),
    )


def plan_summarize_existing() -> TierPlan:
    return TierPlan(
        tier=Tier.SUMMARIZE_EXISTING,
        intent="parse existing artifacts / manifests / summaries; no new model execution",
        resource_posture="CPU-only; read-only artifact parsing; no GPU invocation",
        planned_actions=(
            "discover existing artifact, manifest, and summary locations",
            "parse already-materialized outputs without spawning model jobs",
            "emit a compact rollup for later reporting surfaces",
        ),
        out_of_scope=(
            "new model execution",
            "GPU invocation",
            "new claim or release surface",
        ),
        not_implemented_yet=(
            "artifact discovery",
            "summary parsing",
            "rollup artifact writing",
        ),
    )


def dispatch(tier: Tier) -> TierPlan:
    if tier == Tier.CPU_NIGHTLY:
        return plan_cpu_nightly()
    if tier == Tier.L4_SMOKE:
        return plan_l4_smoke()
    if tier == Tier.L4_WEEKLY:
        return plan_l4_weekly()
    if tier == Tier.SUMMARIZE_EXISTING:
        return plan_summarize_existing()
    raise ValueError(f"unsupported tier: {tier}")


def render_plan(plan: TierPlan) -> str:
    lines = [
        f"tier: {plan.tier.value}",
        f"intent: {plan.intent}",
        f"expected resource posture: {plan.resource_posture}",
        "planned actions:",
    ]
    lines.extend(f"  - {action}" for action in plan.planned_actions)
    lines.append("out of scope:")
    lines.extend(f"  - {item}" for item in plan.out_of_scope)
    lines.append("not implemented yet:")
    lines.extend(f"  - {item}" for item in plan.not_implemented_yet)
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    plan = dispatch(Tier(args.tier))
    print(render_plan(plan))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
