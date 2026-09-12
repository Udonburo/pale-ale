"""Build the DOI-bound public deposit locally; no upload, publication, or experiment."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo


ROOT = Path(__file__).resolve().parent
CAPSULE_LAYOUT = (ROOT / "publication/zenodo-metadata.json").is_file()
METADATA_ROOT = (ROOT / "publication" if CAPSULE_LAYOUT else
                 ROOT.parent.parent / "publications/binary-array-rqmc/zenodo")
SOURCE_FILES = (
    ".gitignore", "README.md", "main.md", "LICENSES.txt", "CITATION.cff",
    "build_pdf.py", "build_publication.py", "pdf-requirements.txt", "pdf-style.typ",
    "figures/component-costs.png", "figures/component-costs.svg",
    "figures/reuse-costs.png", "figures/reuse-costs.svg",
    "repro/.gitignore", "repro/LICENSE", "repro/README.md", "repro/TABLES.md",
    "repro/reference.py", "repro/projection.py", "repro/ordered.py",
    "repro/test_projection.py", "repro/test_projection_structure.py",
    "repro/test_ordered.py", "repro/test_saved_results.py", "repro/run_smoke.py",
    "repro/render_results.py", "repro/requirements.txt", "repro/data/cases.json",
    "repro/data/saved_results.json", "repro/data/provenance.json",
    "output/pdf/binary-array-rqmc.pdf",
)
DEPOSIT_NOTES = (
    "README.txt", "LICENSES.txt", "REPOSITORY.md",
    "zenodo-description.md", "zenodo-metadata.json",
)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def inventory(files):
    return "".join(f"{digest(data)}  {name}\n" for name, data in sorted(files.items())).encode("utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tools-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    destination = (args.output_dir or
                   (ROOT / "output/release" if CAPSULE_LAYOUT else METADATA_ROOT)).resolve()
    metadata = json.loads((METADATA_ROOT / "zenodo-metadata.json").read_text(encoding="utf-8"))
    expected_doi = "10.5281/zenodo.22728405"
    if metadata["doi"] != expected_doi:
        raise ValueError("The deposit metadata must match this manuscript's reserved DOI.")
    manuscript = (ROOT / "main.md").read_text(encoding="utf-8")
    if f"https://doi.org/{expected_doi}" not in manuscript:
        raise ValueError("Manuscript does not cite its deposit identifier.")

    command = [sys.executable, str(ROOT / "build_pdf.py")]
    if args.tools_dir:
        command.extend(["--tools-dir", str(args.tools_dir.resolve())])
    subprocess.run(command, cwd=ROOT, check=True)

    source_payload = {}
    for relative in sorted(SOURCE_FILES):
        source = ROOT / relative
        if not source.is_file() or not source.resolve().is_relative_to(ROOT):
            raise ValueError(f"Missing or external source: {relative}")
        source_payload[relative] = source.read_bytes()
    notes = {name: (METADATA_ROOT / name).read_bytes() for name in DEPOSIT_NOTES}
    if notes["LICENSES.txt"] != source_payload["LICENSES.txt"]:
        raise ValueError("Manuscript and deposit license scopes differ.")
    contents = dict(source_payload)
    contents.update({"publication/" + name: data for name, data in notes.items()})
    contents["CHECKSUMS-SHA256.txt"] = inventory(contents)
    prefix = "binary-array-rqmc/"
    date = datetime.strptime(metadata["publication_date"], "%Y-%m-%d")
    zip_date = (date.year, date.month, date.day, 0, 0, 0)
    destination.mkdir(parents=True, exist_ok=True)
    capsule = destination / "reproducibility-capsule.zip"
    with ZipFile(capsule, "w", compression=ZIP_DEFLATED) as archive:
        for relative, data in sorted(contents.items()):
            info = ZipInfo(prefix + relative, date_time=zip_date)
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            info.compress_type = ZIP_DEFLATED
            archive.writestr(info, data)
    with ZipFile(capsule) as archive:
        expected_names = [prefix + name for name in sorted(contents)]
        if archive.namelist() != expected_names or archive.testzip() is not None:
            raise RuntimeError("Capsule membership or ZIP integrity failed.")
        for relative, data in contents.items():
            if archive.read(prefix + relative) != data:
                raise RuntimeError(f"Archived bytes differ: {relative}")

    top_level = dict(notes)
    top_level["binary-array-rqmc.pdf"] = source_payload["output/pdf/binary-array-rqmc.pdf"]
    top_level["reproducibility-capsule.zip"] = capsule.read_bytes()
    manifest = {
        "title": metadata["title"], "doi": expected_doi, "version": metadata["version"],
        "manuscript_date": metadata["publication_date"], "review_status": "not_peer_reviewed",
        "publication_license": "CC-BY-4.0", "code_license": "MPL-2.0",
        "package_containing_git_commit": None,
        "reproduction_scope": "Exact-law/ordering tests and saved-data figure/table regeneration; not all raw experiments.",
        "omissions": ["Full raw calibration/validation observations", "Bootstrap samples",
                      "Full timing-round history", "Historical baseline/calibration orchestration"],
        "capsule_members": len(contents),
        "files": {name: {"bytes": len(data), "sha256": digest(data)}
                  for name, data in sorted(top_level.items())},
        "capsule_source_files": {name: {"bytes": len(data), "sha256": digest(data)}
                                 for name, data in sorted(source_payload.items())},
    }
    top_level["release_manifest.json"] = (json.dumps(manifest, indent=2) + "\n").encode("utf-8")
    top_level["CHECKSUMS-SHA256.txt"] = inventory(top_level)
    for name, data in top_level.items():
        (destination / name).write_bytes(data)
    if any((destination / name).read_bytes() != data for name, data in top_level.items()):
        raise RuntimeError("Written deposit bytes differ from their inputs.")
    print(json.dumps({"destination": str(destination), "doi": expected_doi,
                      "capsule_members": len(contents), "deposit_files": sorted(top_level),
                      "source_byte_comparison": "PASS", "published": False}, indent=2))


if __name__ == "__main__":
    main()
