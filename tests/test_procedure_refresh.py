"""Offline regression tests for count refresh and legacy route-cache safety."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from geostroke.procedure_inputs import (MANIFEST, RAW_NAMES, assert_current_inputs,
    file_hash, plan_input_refresh, preserve_cached_order, scan_sources, validate_sparse_pairs,
    threshold_polygon_indices, reuse_verified_geocodes)
from geostroke.str_models import gravity_alloc
from scripts.compute_str_travel_time_sparse_sensitivity_2024 import (
    compute_or_load_sparse_pairs, haversine_matrix)
import json


class ProcedureRefreshTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_numeric_prefixes_and_suppression(self):
        path = self.root / "1-2-2024-xml.xml"
        path.write_text("<report><Prozedur><OPS_301>8-020.8</OPS_301><Anzahl>30</Anzahl></Prozedur>"
            "<Prozedur><OPS_301>8-020.8a</OPS_301><Anzahl>188</Anzahl></Prozedur>"
            "<Prozedur><OPS_301>8-836.80</OPS_301><Anzahl>147</Anzahl></Prozedur>"
            "<Prozedur><OPS_301>8-836.80</OPS_301><Datenschutz/></Prozedur>"
            "<Prozedur><OPS_301>8-836.81</OPS_301><Anzahl>999</Anzahl></Prozedur>"
            "<Hauptdiagnose><ICD_10>I63.9</ICD_10><Fallzahl>400</Fallzahl></Hauptdiagnose></report>")
        rows, hashes = scan_sources(self.root)
        self.assertEqual(rows.iloc[0].lysis, 218)
        self.assertEqual(rows.iloc[0].thrombectomy, 147)
        self.assertEqual(rows.iloc[0].thrombectomy_suppressed_cells, 1)
        self.assertEqual(rows.iloc[0].i63, 400)
        self.assertEqual(hashes[path.name], file_hash(path))

    def test_malformed_xml_fails(self):
        (self.root / "1-2-2024-xml.xml").write_text("<broken>")
        with self.assertRaises(Exception):
            scan_sources(self.root)

    def test_invalid_counts_fail(self):
        (self.root / "1-2-2024-xml.xml").write_text("<r><Prozedur><OPS_301>8-020.8</OPS_301><Anzahl>-3</Anzahl></Prozedur></r>")
        with self.assertRaisesRegex(ValueError, "Invalid numeric"):
            scan_sources(self.root)

    def fixture_counts(self):
        base = pd.DataFrame({"file_name": ["b.xml", "a.xml"], "total_cases": [100, 43],
            "latitude": [50., 49.], "longitude": [7., 9.], "procedure_details": ["[]", "[]"],
            "procedures_found": [1, 1], "I63_total": [900, 200]})
        for therapy, name in RAW_NAMES.items():
            base.to_csv(self.root / name, index=False)
            for suffix in ["centers", "10plus"]:
                base.to_csv(self.root / f"{therapy}_{suffix}_geocoded.csv", index=False)
        return pd.DataFrame({"file_name": ["a.xml", "b.xml"], "lysis": [218, 100],
            "thrombectomy": [147, 100], "lysis_details": [[], []], "thrombectomy_details": [[], []]})

    def test_refresh_preserves_geography_linkage_and_order(self):
        sources = self.fixture_counts()
        frames, deltas = plan_input_refresh(self.root, sources)
        for therapy in RAW_NAMES:
            result = frames[f"{therapy}_10plus_geocoded.csv"]
            self.assertEqual(result.file_name.to_list(), ["b.xml", "a.xml"])
            self.assertEqual(result.latitude.to_list(), [50., 49.])
            self.assertEqual(result.I63_total.to_list(), [900, 200])
        self.assertEqual(len(deltas), 2)
        self.assertEqual(pd.read_csv(self.root / RAW_NAMES["lysis"]).total_cases.sum(), 143)

    def test_membership_change_fails_before_writes(self):
        sources = self.fixture_counts()
        sources.loc[0, "lysis"] = 5
        with self.assertRaisesRegex(ValueError, "cohort changed"):
            plan_input_refresh(self.root, sources)
        self.assertEqual(pd.read_csv(self.root / RAW_NAMES["lysis"]).total_cases.sum(), 143)

    def test_order_restored_by_filename(self):
        a = pd.DataFrame({"file_name": ["b", "a"], "x": [1, 2]})
        b = a.iloc[::-1].copy()
        self.assertEqual(preserve_cached_order(a, b).x.to_list(), [2, 1])

    def test_verified_geocodes_are_not_rewritten(self):
        self.fixture_counts()
        path = self.root / "lysis_centers_geocoded.csv"
        digest = file_hash(path)
        current = pd.read_csv(path).iloc[::-1].reset_index(drop=True)
        result = reuse_verified_geocodes(current, path)
        self.assertEqual(result.file_name.to_list(), ["b.xml", "a.xml"])
        self.assertEqual(file_hash(path), digest)
        current.loc[0, "total_cases"] += 1
        with self.assertRaisesRegex(ValueError, "changed total_cases"):
            reuse_verified_geocodes(current, path)

    def test_changed_xml_detected_without_count_change(self):
        xml = self.root / "1-2-2024-xml.xml"
        xml.write_text("<report/>")
        (self.root / MANIFEST).write_text(json.dumps({"sources": {xml.name: file_hash(xml)}, "inputs": {}}))
        assert_current_inputs(self.root, self.root)
        xml.write_text("<report> </report>")
        with self.assertRaisesRegex(RuntimeError, "XMLs changed"):
            assert_current_inputs(self.root, self.root)

    def test_changed_extraction_detected(self):
        cache = self.root / "input.csv"
        cache.write_text("count\n43\n")
        (self.root / MANIFEST).write_text(json.dumps({"sources": {}, "inputs": {cache.name: file_hash(cache)}}))
        assert_current_inputs(self.root)
        cache.write_text("count\n218\n")
        with self.assertRaisesRegex(RuntimeError, "Unverified procedure cache"):
            assert_current_inputs(self.root)

    def test_threshold_binding_not_csv_position(self):
        poly = self.root / "poly60_thrombectomy_2024.pkl"
        poly.write_bytes(b"test-cache")
        pd.DataFrame({"total_cases": [30, 147]}).to_csv(self.root / "thrombectomy_centers_geocoded.csv", index=False)
        binding = {"thrombectomy": {"hashes": {poly.name: file_hash(poly)}, "poly_volumes": [147, 30]}}
        (self.root / "threshold_cache_binding_2024.json").write_text(json.dumps(binding))
        self.assertEqual(threshold_polygon_indices(self.root, self.root, "thrombectomy", 100), [0])
        poly.write_bytes(b"changed-cache")
        with self.assertRaisesRegex(ValueError, "Isochrone cache changed"):
            threshold_polygon_indices(self.root, self.root, "thrombectomy", 100)

    def routes(self):
        nodes = pd.DataFrame({"node_id": ["001_0", "001_1"], "krs_code_str": ["00001", "00001"],
            "lat": [50., 50.2], "lon": [7., 7.2]})
        hospitals = pd.DataFrame({"facility_id": ["ivt_0000", "ivt_0001"],
            "latitude": [50.1, 50.3], "longitude": [7.1, 7.3]})
        distance = haversine_matrix(nodes, hospitals)
        pairs = pd.DataFrame([dict(node_id=nodes.iloc[i].node_id, krs_code_str="00001",
            node_idx=i, facility_idx=j, facility_id=hospitals.iloc[j].facility_id,
            duration_min=20., distance_km=30., haversine_km=distance[i,j]) for i in range(2) for j in range(2)])
        return nodes, hospitals, pairs

    def test_complete_cache_offline(self):
        nodes, hospitals, pairs = self.routes()
        path = self.root / "pairs.csv"
        pairs.to_csv(path, index=False)
        with patch("scripts.compute_str_travel_time_sparse_sensitivity_2024.CACHE_DIR", self.root), \
             patch("scripts.compute_str_travel_time_sparse_sensitivity_2024.sparse_path", return_value=path), \
             patch("scripts.compute_str_travel_time_sparse_sensitivity_2024.wait_for_ors") as wait:
            loaded = compute_or_load_sparse_pairs(nodes, hospitals, "ivt", 180, False, "unreachable")
            self.assertEqual(len(loaded), 4)
            wait.assert_not_called()

    def test_route_order_and_geometry_guard(self):
        nodes, hospitals, pairs = self.routes()
        validate_sparse_pairs(nodes, hospitals, pairs, 180)
        with self.assertRaisesRegex(ValueError, "identity/order"):
            validate_sparse_pairs(nodes, hospitals.iloc[::-1].reset_index(drop=True), pairs, 180)
        hospitals.loc[0, "latitude"] += .1
        with self.assertRaisesRegex(ValueError, "coordinates/order"):
            validate_sparse_pairs(nodes, hospitals, pairs, 180)

    def test_partial_duplicate_and_invalid_pairs(self):
        nodes, hospitals, pairs = self.routes()
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            validate_sparse_pairs(nodes, hospitals, pairs.iloc[:-1], 180)
        self.assertFalse(validate_sparse_pairs(nodes, hospitals, pairs.iloc[:-1], 180, False)["complete"])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            validate_sparse_pairs(nodes, hospitals, pd.concat([pairs, pairs.iloc[:1]]), 180)
        pairs.loc[0, "duration_min"] = -1
        with self.assertRaisesRegex(ValueError, "Invalid cached"):
            validate_sparse_pairs(nodes, hospitals, pairs, 180)

    def test_gravity_conserves_volume_including_fallback(self):
        hospitals = pd.DataFrame({"latitude": [50., 80.], "longitude": [7., 1.], "total_cases": [218, 147]})
        counties = pd.DataFrame({"centroid_lat": [50., 52.], "centroid_lon": [7., 8.], "total_pop": [1000, 500]})
        allocation = gravity_alloc(hospitals, counties, beta=2, max_km=100)
        self.assertAlmostEqual(allocation.sum(), 365)
        self.assertTrue((allocation >= 0).all())


if __name__ == "__main__":
    unittest.main()
