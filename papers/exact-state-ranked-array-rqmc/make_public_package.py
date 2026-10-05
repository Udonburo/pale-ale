"""Build a deterministic ZIP from the reviewed public manifest."""
from pathlib import Path
from zipfile import ZipFile, ZipInfo, ZIP_DEFLATED
import argparse
import hashlib
import json

ROOT = Path(__file__).resolve().parent
STEM = 'array-rqmc-repro-20261006'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT / 'output' / (STEM + '.zip'))
    args = parser.parse_args()
    manifest = json.loads((ROOT / 'MANIFEST.json').read_text(encoding='utf-8'))
    contents = {}
    for name, expected in manifest.items():
        path = Path(name)
        if path.is_absolute() or '..' in path.parts or ':' in name:
            raise ValueError(name)
        content = (ROOT / path).read_bytes()
        assert hashlib.sha256(content).hexdigest() == expected, name
        contents[name] = content
    contents['MANIFEST.json'] = (ROOT / 'MANIFEST.json').read_bytes()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        raise FileExistsError('Choose a new destination; do not overwrite a distributed archive.')
    with ZipFile(args.out, 'w', ZIP_DEFLATED, compresslevel=9) as z:
        for name, content in sorted(contents.items()):
            entry = ZipInfo(STEM + '/' + name, date_time=(2026, 10, 6, 0, 0, 0))
            entry.compress_type = ZIP_DEFLATED
            z.writestr(entry, content, compresslevel=9)
    print(json.dumps(dict(archive=args.out.name, files=len(contents),
                         bytes=args.out.stat().st_size,
                         sha256=hashlib.sha256(args.out.read_bytes()).hexdigest()), indent=2))


if __name__ == '__main__':
    main()
