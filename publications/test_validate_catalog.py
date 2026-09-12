"""Check optional platform targets without weakening declared-release checks."""

from contextlib import redirect_stdout
from copy import deepcopy
from io import StringIO
import json
import unittest
from unittest.mock import patch

import validate_catalog


class PublicationTargetChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = json.loads(validate_catalog.CATALOG.read_text(encoding="utf-8"))

    def run_catalog(self, catalog):
        output = StringIO()
        with patch.object(validate_catalog, "CATALOG") as source:
            source.read_text.return_value = json.dumps(catalog)
            with redirect_stdout(output):
                code = validate_catalog.main()
        return code, output.getvalue()

    def test_published_catalog(self):
        code, output = self.run_catalog(self.catalog)
        self.assertEqual(code, 0, output)

    def test_paper_without_github_release(self):
        catalog = deepcopy(self.catalog)
        catalog["publications"][0]["targets"].pop("github_release")
        code, output = self.run_catalog(catalog)
        self.assertEqual(code, 0, output)

    def test_other_record_without_github_release(self):
        catalog = deepcopy(self.catalog)
        catalog["other_public_records"][0]["targets"].pop("github_release")
        code, output = self.run_catalog(catalog)
        self.assertEqual(code, 0, output)

    def test_explicit_empty_release_is_rejected(self):
        for section in ("publications", "other_public_records"):
            with self.subTest(section=section):
                catalog = deepcopy(self.catalog)
                catalog[section][0]["targets"]["github_release"] = {}
                code, output = self.run_catalog(catalog)
                self.assertEqual(code, 1)
                self.assertIn("malformed GitHub Release target", output)

    def test_mismatched_release_url_is_rejected(self):
        catalog = deepcopy(self.catalog)
        catalog["publications"][0]["targets"]["github_release"]["url"] = (
            "https://github.com/Udonburo/pale-ale/releases/tag/wrong-tag"
        )
        code, output = self.run_catalog(catalog)
        self.assertEqual(code, 1)
        self.assertIn("malformed GitHub Release target", output)

    def test_checksum_mismatch_is_still_rejected(self):
        with patch.object(validate_catalog, "sha256_file", return_value="0" * 64):
            code, output = self.run_catalog(self.catalog)
        self.assertEqual(code, 1)
        self.assertIn("checksum mismatch", output)


if __name__ == "__main__":
    unittest.main()
