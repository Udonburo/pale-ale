"""Validate the platform-neutral publication catalog without network access."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CATALOG = Path(__file__).with_name("catalog.json")
SLUG = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
DOI = re.compile(r"^10\.5281/zenodo\.\d+$")
CHECKSUM = re.compile(r"^([0-9A-Fa-f]{64})\s+(.+)$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    data = json.loads(CATALOG.read_text(encoding="utf-8"))
    errors: list[str] = []
    seen_slugs: set[str] = set()
    seen_dois: set[str] = set()
    checksum_count = 0

    for item in data.get("publications", []):
        slug = item.get("slug", "")
        doi = item.get("doi", "")
        if not SLUG.fullmatch(slug):
            errors.append(f"invalid slug: {slug!r}")
        if slug in seen_slugs:
            errors.append(f"duplicate slug: {slug}")
        seen_slugs.add(slug)
        if not DOI.fullmatch(doi):
            errors.append(f"invalid DOI: {doi!r}")
        if doi in seen_dois:
            errors.append(f"duplicate DOI: {doi}")
        seen_dois.add(doi)

        expected_landing = f"publications/{slug}/README.md"
        if item.get("landing_path") != expected_landing:
            errors.append(f"{slug}: landing_path must be {expected_landing}")

        paths = [item.get("landing_path"), item.get("source_path")]
        zenodo = item.get("targets", {}).get("zenodo", {})
        paths.extend([zenodo.get("package_path")])
        for value in filter(None, paths):
            if not (ROOT / value).exists():
                errors.append(f"{slug}: missing repository path {value}")

        package_path = zenodo.get("package_path")
        if package_path and not (ROOT / package_path / "README.txt").is_file():
            errors.append(f"{slug}: Zenodo package has no README.txt")
        if package_path:
            package = ROOT / package_path
            inventory = package / "CHECKSUMS-SHA256.txt"
            if not inventory.is_file():
                errors.append(f"{slug}: Zenodo package has no checksum inventory")
            else:
                for line_number, line in enumerate(
                    inventory.read_text(encoding="utf-8").splitlines(), start=1
                ):
                    if not line.strip():
                        continue
                    match = CHECKSUM.fullmatch(line)
                    if not match:
                        errors.append(
                            f"{slug}: malformed checksum line {line_number}"
                        )
                        continue
                    expected, relative_text = match.groups()
                    relative = Path(relative_text.strip())
                    if relative.is_absolute() or ".." in relative.parts:
                        errors.append(
                            f"{slug}: unsafe checksum path {relative_text!r}"
                        )
                        continue
                    target = package / relative
                    if not target.is_file():
                        errors.append(f"{slug}: checksum target missing: {relative}")
                        continue
                    if sha256_file(target) != expected.lower():
                        errors.append(f"{slug}: checksum mismatch: {relative}")
                    checksum_count += 1

        release = item.get("targets", {}).get("github_release", {})
        tag = release.get("tag", "")
        expected_url = f"https://github.com/Udonburo/pale-ale/releases/tag/{tag}"
        if not tag or release.get("url") != expected_url:
            errors.append(f"{slug}: malformed GitHub Release target")

    other_records = data.get("other_public_records", [])
    for item in other_records:
        slug = item.get("slug", "")
        if not SLUG.fullmatch(slug):
            errors.append(f"invalid other-record slug: {slug!r}")
        if slug in seen_slugs:
            errors.append(f"duplicate slug: {slug}")
        seen_slugs.add(slug)

        expected_landing = f"publications/{slug}/README.md"
        if item.get("landing_path") != expected_landing:
            errors.append(f"{slug}: landing_path must be {expected_landing}")
        for value in filter(None, (item.get("landing_path"), item.get("source_path"))):
            if not (ROOT / value).exists():
                errors.append(f"{slug}: missing repository path {value}")

        doi = item.get("doi")
        if doi:
            if not DOI.fullmatch(doi):
                errors.append(f"invalid DOI: {doi!r}")
            if doi in seen_dois:
                errors.append(f"duplicate DOI: {doi}")
            seen_dois.add(doi)

        release = item.get("targets", {}).get("github_release", {})
        tag = release.get("tag", "")
        expected_url = f"https://github.com/Udonburo/pale-ale/releases/tag/{tag}"
        if not tag or release.get("url") != expected_url:
            errors.append(f"{slug}: malformed GitHub Release target")

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print(
        f"PASS: {len(data.get('publications', []))} papers/notes, "
        f"{len(other_records)} other public records, and "
        f"{checksum_count} package checksums validated"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
