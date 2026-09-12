"""Rebuild the PDF and package the public review files; never publish or simulate."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import re
import subprocess
import sys
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo


ROOT = Path(__file__).resolve().parent
FILES = (
    ".gitignore",
    "README.md",
    "main.md",
    "build_pdf.py",
    "build_review_bundle.py",
    "pdf-requirements.txt",
    "pdf-style.typ",
    "figures/component-costs.png",
    "figures/component-costs.svg",
    "figures/reuse-costs.png",
    "figures/reuse-costs.svg",
    "repro/.gitignore",
    "repro/LICENSE",
    "repro/README.md",
    "repro/TABLES.md",
    "repro/reference.py",
    "repro/projection.py",
    "repro/ordered.py",
    "repro/test_projection.py",
    "repro/test_projection_structure.py",
    "repro/test_ordered.py",
    "repro/test_saved_results.py",
    "repro/run_smoke.py",
    "repro/render_results.py",
    "repro/requirements.txt",
    "repro/data/cases.json",
    "repro/data/saved_results.json",
    "repro/data/provenance.json",
    "output/pdf/binary-array-rqmc.pdf",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tools-dir", type=Path,
                        help="Optional directory containing the PDF dependencies.")
    args = parser.parse_args()
    command = [sys.executable, str(ROOT / "build_pdf.py")]
    if args.tools_dir:
        command.extend(["--tools-dir", str(args.tools_dir.resolve())])
    subprocess.run(command, cwd=ROOT, check=True)

    markdown = (ROOT / "main.md").read_text(encoding="utf-8")
    match = re.search(r"^Preprint - (\d+ \w+ \d{4})$", markdown, re.MULTILINE)
    if not match:
        raise ValueError("Expected an explicit manuscript date.")
    date = datetime.strptime(match.group(1), "%d %B %Y")
    zip_date = (date.year, date.month, date.day, 0, 0, 0)
    prefix = "binary-array-rqmc/"
    payload = {}
    for relative in sorted(FILES):
        source = ROOT / relative
        if not source.is_file() or not source.resolve().is_relative_to(ROOT):
            raise ValueError(f"Missing or external bundle source: {relative}")
        payload[prefix + relative] = source.read_bytes()

    destination = ROOT / "output/release/binary-array-rqmc-review.zip"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(destination, "w", compression=ZIP_DEFLATED) as archive:
        for name, data in payload.items():
            info = ZipInfo(name, date_time=zip_date)
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            info.compress_type = ZIP_DEFLATED
            archive.writestr(info, data)
    with ZipFile(destination) as archive:
        if archive.namelist() != list(payload) or archive.testzip() is not None:
            raise RuntimeError("Bundle membership or ZIP integrity check failed.")
        for name, data in payload.items():
            if archive.read(name) != data:
                raise RuntimeError(f"Archived bytes differ: {name}")
    print(json.dumps({"output": str(destination), "files": len(payload),
                      "bytes": destination.stat().st_size,
                      "source_byte_comparison": "PASS",
                      "published": False}, indent=2))


if __name__ == "__main__":
    main()
