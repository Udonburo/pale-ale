"""Checks on the compact saved export, independent of the simulation tests."""
import math
import unittest
from render_results import load_data, validate_data, target_rows, crossover


class SavedResultsChecks(unittest.TestCase):
    def test_export_grain_and_cost_arithmetic(self):
        validate_data(load_data())

    def test_crossovers_by_integer_boundary(self):
        r=target_rows(load_data(),"case_08_distant")
        a,b=r["MAINTAINED"],r["CRN"]
        for key,expected in (("all_acquired_setup",2799),("incremental_setup",476)):
            k=crossover(a,b,key)
            self.assertEqual(k,expected)
            self.assertGreaterEqual(a[key]+(k-1)*a["wall"], b[key]+(k-1)*b["wall"])
            self.assertLess(a[key]+k*a["wall"], b[key]+k*b["wall"])

    def test_profiles_agree_with_exported_component_means(self):
        data=load_data()
        for batch,method in (("projection","LIBRARY"),("projection","DIRECT"),
                             ("ordering","DIRECT"),("ordering","MAINTAINED")):
            rows=[r for r in data["profiles"] if r["batch"]==batch and r["method"]==method]
            conditions={r["condition"] for r in rows}
            self.assertEqual(len(conditions),16)
            # Equal-sized strata: averaging all profiles must match averaging strata.
            total=sum(sum(r["times"].values()) for r in rows)/len(rows)
            by_condition=[sum(sum(r["times"].values()) for r in rows if r["condition"]==c)/
                          sum(r["condition"]==c for r in rows) for c in conditions]
            self.assertTrue(math.isclose(total,sum(by_condition)/16,rel_tol=1e-14))


if __name__=="__main__":
    unittest.main(verbosity=2)
