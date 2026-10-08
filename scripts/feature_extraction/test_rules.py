import unittest
from pathlib import Path

from match_annotations import Matcher, load_rules

HERE = Path(__file__).resolve().parent


class RuleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.matcher = Matcher(load_rules(HERE / "rules.json"))

    def test_is_row_counts_once(self):
        hits = self.matcher.is_hits("IS26 transposase")
        self.assertGreaterEqual(len(hits), 1)
        self.assertEqual(int(bool(hits)), 1)

    def test_is_boundaries(self):
        self.assertTrue(self.matcher.is_hits("Tn3 transposase"))
        self.assertFalse(self.matcher.is_hits("protein ISLAND regulator"))

    def test_amr_structured_fields(self):
        carb = self.matcher.amr_flags({"Type": "AMR", "Subclass": "BETA-LACTAM/CARBAPENEM", "Element name": "x"})
        self.assertEqual(carb["classified_carbapenemase"], 1)
        esbl = self.matcher.amr_flags({"Type": "AMR", "Subclass": "CEPHALOSPORIN", "Element name": "extended-spectrum class A beta-lactamase"})
        self.assertEqual(esbl["classified_ESBL"], 1)
        non_amr = self.matcher.amr_flags({"Type": "STRESS", "Subclass": "CARBAPENEM", "Element name": "extended-spectrum class A beta-lactamase"})
        self.assertEqual(sum(non_amr.values()), 0)

    def test_mob_fields(self):
        result = self.matcher.mob_features({"orit_type(s)": "-", "relaxase_type(s)": "MOBH", "mpf_type": "MPF_F", "predicted_mobility": " conjugative "})
        self.assertEqual(result, {"has_oriT": 0, "has_relaxase": 1, "has_MPF": 1, "mobility": 1})


if __name__ == "__main__":
    unittest.main()
