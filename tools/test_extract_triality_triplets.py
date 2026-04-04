#!/usr/bin/env python3
"""Unit tests for attention-surface admission in extract_triality_triplets."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import extract_triality_triplets as extractor


class ResolveSplusSurfaceTests(unittest.TestCase):
    def test_attention_surface_stays_mainline_when_attentions_exist(self) -> None:
        uses_attention, splus_def_id = extractor.resolve_splus_surface(
            attentions=(object(),),
            allow_attentionless_splus_fallback=False,
        )
        self.assertTrue(uses_attention)
        self.assertEqual(splus_def_id, extractor.ATTN_WEIGHTED_SPLUS_DEF_ID)

    def test_attentionless_surface_fails_closed_without_explicit_opt_in(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "allow-attentionless-splus-fallback"):
            extractor.resolve_splus_surface(
                attentions=None,
                allow_attentionless_splus_fallback=False,
            )

    def test_attentionless_surface_requires_explicit_opt_in(self) -> None:
        uses_attention, splus_def_id = extractor.resolve_splus_surface(
            attentions=None,
            allow_attentionless_splus_fallback=True,
        )
        self.assertFalse(uses_attention)
        self.assertEqual(splus_def_id, extractor.ATTNLESS_PREFIX_MEAN_SPLUS_DEF_ID)


if __name__ == "__main__":
    unittest.main()
