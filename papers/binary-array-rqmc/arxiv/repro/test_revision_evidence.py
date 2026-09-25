"""Additional mathematical and original-row checks for the revised manuscript."""
from fractions import Fraction as F
import unittest
from stopped_witness import rank_laws, verify as verify_witness
from reanalyze_timing import verify as verify_timing


class RevisionEvidence(unittest.TestCase):
    def test_stopped_chain_complete_payoff_distribution(self):
        self.assertEqual(verify_witness()['variance_difference'], '45927/268435456')

    def test_rank_laws_have_identical_covariance(self):
        for law in rank_laws().values():
            for r in range(8):
                self.assertEqual(sum(v[r] for v in law), 0)
                for s in range(8):
                    expected = 1 if r == s else (-1 if r//2 == s//2 else 0)
                    self.assertEqual(F(sum(v[r]*v[s] for v in law), len(law)), expected)

    def test_original_paired_timing_rows(self):
        self.assertEqual(len(verify_timing()), 2)


if __name__ == '__main__':
    unittest.main()
