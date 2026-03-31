#!/usr/bin/env python3
"""Replay the unchanged Gate12A machine surface from a Gate8 execution family."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = REPO_ROOT / "tools"
SCRIPT_PATH = Path(__file__).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay the existing Gate9/Gate12A machine surface from a single Gate8 execution "
            "family without changing the observable contract."
        )
    )
    parser.add_argument("--gate8-execution-dir", required=True)
    parser.add_argument(
        "--origin-label",
        help=(
            "Short stable label used in replay run ids. Defaults to the Gate8 execution "
            "directory name with the _candidate_execution suffix removed."
        ),
    )
    parser.add_argument("--out-root", default="runs")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--balanced-per-band", type=int, default=6)
    parser.add_argument("--reading-limit", type=int, default=0)
    return parser.parse_args()


def normalize_origin_label(gate8_execution_dir: Path, explicit_label: str | None) -> str:
    if explicit_label:
        return explicit_label
    name = gate8_execution_dir.name
    suffix = "_candidate_execution"
    if name.endswith(suffix):
        return name[: -len(suffix)]
    return name


def repo_path(relative: str) -> str:
    return str((REPO_ROOT / relative).resolve())


def build_run_dirs(
    out_root: Path,
    origin_label: str,
    balanced_per_band: int,
) -> dict[str, Path]:
    upstream_label = f"{origin_label}_gate9k"
    packet_name = (
        f"gate12a_triangle_reading_packet_balanced_recheck_from_gate12a_upstream_{upstream_label}"
        if balanced_per_band > 0
        else f"gate12a_triangle_reading_packet_recheck_from_gate12a_upstream_{upstream_label}"
    )
    return {
        "gate9a": out_root / f"gate9a_{origin_label}_failure_surface",
        "gate9g": out_root / f"gate9g_anchor_conditioned_triviality_recheck_from_gate9a_{origin_label}",
        "gate9h": out_root / f"gate9h_anchor_coverage_gap_redesign_recheck_from_gate9g_{origin_label}",
        "gate9i": out_root / f"gate9i_support_anchor_cleaner_cell_dominance_recheck_from_gate9h_{origin_label}",
        "gate9j": out_root / f"gate9j_distributed_underactivation_recheck_from_gate9i_{origin_label}",
        "gate9k": out_root / f"gate9k_trusted_tree_residual_chord_logging_recheck_from_gate9j_{origin_label}",
        "node_family": out_root / f"gate12a_node_local_object_family_recheck_from_{origin_label}",
        "relation_family": out_root / f"gate12a_relation_seed_family_recheck_from_gate9k_{origin_label}",
        "gate12a": out_root / f"gate12a_discrete_connection_recheck_from_gate12a_upstream_{upstream_label}",
        "calibration": out_root / f"gate12a_calibration_seed_audit_recheck_from_gate12a_upstream_{upstream_label}",
        "text_surface": out_root / f"gate12a_triangle_text_surface_audit_recheck_from_gate12a_upstream_{upstream_label}",
        "phenotype_prep": out_root / f"gate12a_triangle_phenotype_tag_prep_recheck_from_gate12a_upstream_{upstream_label}",
        "reading_queue": out_root / f"gate12a_triangle_reading_queue_recheck_from_gate12a_upstream_{upstream_label}",
        "reading_packet": out_root / packet_name,
    }


def build_commands(
    gate8_execution_dir: Path,
    out_root: Path,
    origin_label: str,
    top_k: int,
    balanced_per_band: int,
    reading_limit: int,
) -> List[Tuple[str, List[str]]]:
    run_dirs = build_run_dirs(out_root=out_root, origin_label=origin_label, balanced_per_band=balanced_per_band)
    python_exe = sys.executable

    commands: List[Tuple[str, List[str]]] = [
        (
            "gate9a",
            [
                python_exe,
                str((TOOLS_DIR / "run_gate9a_graph_gauge_consumer.py").resolve()),
                "--gate8-execution-dir",
                str(gate8_execution_dir),
                "--out-dir",
                str(run_dirs["gate9a"]),
            ],
        ),
        (
            "gate9g",
            [
                python_exe,
                str((TOOLS_DIR / "run_gate9g_anchor_conditioned_triviality_audit.py").resolve()),
                "--gate9a-dir",
                str(run_dirs["gate9a"]),
                "--out-dir",
                str(run_dirs["gate9g"]),
            ],
        ),
        (
            "gate9h",
            [
                python_exe,
                str((TOOLS_DIR / "run_gate9h_anchor_coverage_gap_redesign_audit.py").resolve()),
                "--gate9g-dir",
                str(run_dirs["gate9g"]),
                "--out-dir",
                str(run_dirs["gate9h"]),
            ],
        ),
        (
            "gate9i",
            [
                python_exe,
                str((TOOLS_DIR / "run_gate9i_support_anchor_cleaner_cell_dominance_audit.py").resolve()),
                "--gate9h-dir",
                str(run_dirs["gate9h"]),
                "--out-dir",
                str(run_dirs["gate9i"]),
            ],
        ),
        (
            "gate9j",
            [
                python_exe,
                str((TOOLS_DIR / "run_gate9j_distributed_underactivation_audit.py").resolve()),
                "--gate9i-dir",
                str(run_dirs["gate9i"]),
                "--out-dir",
                str(run_dirs["gate9j"]),
            ],
        ),
        (
            "gate9k",
            [
                python_exe,
                str((TOOLS_DIR / "run_gate9k_trusted_tree_residual_chord_logging.py").resolve()),
                "--gate9j-dir",
                str(run_dirs["gate9j"]),
                "--out-dir",
                str(run_dirs["gate9k"]),
            ],
        ),
        (
            "gate12a_node_family",
            [
                python_exe,
                str((TOOLS_DIR / "run_gate12a_node_local_object_export.py").resolve()),
                "--gate8-execution-dir",
                str(gate8_execution_dir),
                "--out-dir",
                str(run_dirs["node_family"]),
            ],
        ),
        (
            "gate12a_relation_family",
            [
                python_exe,
                str((TOOLS_DIR / "run_gate12a_relation_seed_export.py").resolve()),
                "--gate9k-dir",
                str(run_dirs["gate9k"]),
                "--out-dir",
                str(run_dirs["relation_family"]),
            ],
        ),
        (
            "gate12a",
            [
                python_exe,
                str((TOOLS_DIR / "run_gate12a_discrete_connection_audit.py").resolve()),
                "--node-artifact-dir",
                str(run_dirs["node_family"]),
                "--relation-seed-dir",
                str(run_dirs["relation_family"]),
                "--out-dir",
                str(run_dirs["gate12a"]),
            ],
        ),
        (
            "calibration",
            [
                python_exe,
                str((TOOLS_DIR / "run_gate12a_calibration_seed_audit.py").resolve()),
                "--gate12a-dir",
                str(run_dirs["gate12a"]),
                "--out-dir",
                str(run_dirs["calibration"]),
                "--top-k",
                str(top_k),
            ],
        ),
        (
            "text_surface",
            [
                python_exe,
                str((TOOLS_DIR / "run_gate12a_triangle_text_surface_audit.py").resolve()),
                "--gate12a-dir",
                str(run_dirs["gate12a"]),
                "--gate8-execution-dir",
                str(gate8_execution_dir),
                "--out-dir",
                str(run_dirs["text_surface"]),
                "--top-k",
                str(top_k),
            ],
        ),
        (
            "phenotype_prep",
            [
                python_exe,
                str((TOOLS_DIR / "run_gate12a_triangle_phenotype_tag_prep.py").resolve()),
                "--triangle-text-audit-dir",
                str(run_dirs["text_surface"]),
                "--out-dir",
                str(run_dirs["phenotype_prep"]),
            ],
        ),
        (
            "reading_queue",
            [
                python_exe,
                str((TOOLS_DIR / "run_gate12a_triangle_reading_queue.py").resolve()),
                "--phenotype-prep-dir",
                str(run_dirs["phenotype_prep"]),
                "--out-dir",
                str(run_dirs["reading_queue"]),
            ],
        ),
        (
            "reading_packet",
            [
                python_exe,
                str((TOOLS_DIR / "run_gate12a_triangle_reading_packet.py").resolve()),
                "--reading-queue-dir",
                str(run_dirs["reading_queue"]),
                "--triangle-text-audit-dir",
                str(run_dirs["text_surface"]),
                "--out-dir",
                str(run_dirs["reading_packet"]),
            ],
        ),
    ]
    if reading_limit > 0:
        commands[-1][1].extend(["--limit", str(reading_limit)])
    if balanced_per_band > 0:
        commands[-1][1].extend(["--balanced-per-band", str(balanced_per_band)])
    return commands


def run_subprocess(command: Sequence[str]) -> None:
    completed = subprocess.run(
        list(command),
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed rc={completed.returncode}: {' '.join(command)}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )


def main() -> int:
    args = parse_args()
    gate8_execution_dir = Path(args.gate8_execution_dir)
    out_root = Path(args.out_root)
    origin_label = normalize_origin_label(gate8_execution_dir, args.origin_label)
    commands = build_commands(
        gate8_execution_dir=gate8_execution_dir,
        out_root=out_root,
        origin_label=origin_label,
        top_k=args.top_k,
        balanced_per_band=args.balanced_per_band,
        reading_limit=args.reading_limit,
    )
    for _step, command in commands:
        run_subprocess(command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
