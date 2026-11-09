from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from geostroke.age_calibration import calibrate_2024, GROUPS


class AgeCalibrationTests(unittest.TestCase):
    def fixture(self, path, date="31.12.2024", duplicate=False):
        rows = []
        for county in range(1, 401):
            for labels in GROUPS.values():
                for label in labels:
                    rows.append([date, f"{county:05d}", "County", label, 1234, 600, 634])
        if duplicate:
            rows.append(rows[0])
        path.write_text("\n" * 6, encoding="latin-1")
        pd.DataFrame(rows).to_csv(path, sep=";", index=False, header=False, mode="a", encoding="latin-1")

    def test_full_precision_conserves_national_cases(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "age.csv"
            self.fixture(path)
            counties, calibration = calibrate_2024(path, 241251)
            self.assertEqual(len(counties), 400)
            self.assertAlmostEqual(counties.expected_strokes.sum(), 241251, places=7)
            self.assertAlmostEqual(calibration.estimated_cases.sum(), 241251)
            self.assertTrue(np.allclose(calibration.estimated_cases,
                calibration.population * calibration.rate_per_100000 / 100000))
            self.assertFalse(np.allclose(calibration.rate_per_100000, calibration.rate_per_100000.round()))

    def test_refuses_wrong_year(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "age.csv"
            self.fixture(path, date="31.12.2023")
            with self.assertRaisesRegex(ValueError, "31 December 2024"):
                calibrate_2024(path, 241251)

    def test_refuses_duplicate_county_age_cells(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "age.csv"
            self.fixture(path, duplicate=True)
            with self.assertRaisesRegex(ValueError, "duplicated"):
                calibrate_2024(path, 241251)


if __name__ == "__main__":
    unittest.main()
