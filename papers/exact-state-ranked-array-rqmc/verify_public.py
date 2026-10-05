"""Check public package bytes, original source identities and nested inputs."""
from pathlib import Path
from zipfile import ZipFile
import hashlib
import json

ROOT = Path(__file__).resolve().parent


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def safe_path(name):
    path = Path(name)
    if path.is_absolute() or '..' in path.parts or ':' in name:
        raise ValueError('Unsafe relative path: ' + name)
    return ROOT / path


def main():
    manifest = json.loads((ROOT / 'MANIFEST.json').read_text(encoding='utf-8'))
    for name, expected in manifest.items():
        assert sha256(safe_path(name).read_bytes()) == expected, name
    provenance = json.loads((ROOT / 'PUBLIC_RELEASE.json').read_text(encoding='utf-8'))
    for name, expected in provenance['unchanged_review_files'].items():
        assert sha256(safe_path(name).read_bytes()) == expected, name
    study = ROOT / 'review_checks/primal_dual'
    metadata = json.loads((study / 'confirmation/data/metadata.json').read_text())
    for name, expected in metadata['source'].items():
        assert sha256((study / 'confirmation/source' / name).read_bytes()) == expected, name
    with ZipFile(ROOT / 'archive/retained-dense-inputs.zip') as z:
        for info in z.infolist():
            safe_path(info.filename)
        for name, expected in provenance['unchanged_dense_files'].items():
            assert sha256(z.read('array-rqmc-paper1/' + name)) == expected, name
    timings = (study / 'confirmation/data/timings.jsonl').read_text().splitlines()
    assert len(timings) == 4080
    print(json.dumps(dict(status='PASS', manifest_files=len(manifest),
        unchanged_review_files=len(provenance['unchanged_review_files']),
        unchanged_dense_files=len(provenance['unchanged_dense_files']),
        measured_source_files=len(metadata['source']), original_timings=len(timings),
        paper_doi=provenance['paper_doi'], performance_timings_repeated=False), indent=2))


if __name__ == '__main__':
    main()
