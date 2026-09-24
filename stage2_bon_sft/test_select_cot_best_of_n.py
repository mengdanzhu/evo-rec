#!/usr/bin/env python3
"""Unit tests for the best-of-N selection rule.

Run: ``python stage2_bon_sft/test_select_cot_best_of_n.py``
"""

import sys
import unittest
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
for path in (str(HERE), str(HERE.parent)):
    if path not in sys.path:
        sys.path.insert(0, path)

from select_cot_best_of_n import select_best_of_n  # noqa: E402


def expanded_frame(rows):
    """rows: {source_index: [cot_source, ...]} -> the aligned candidate frame."""
    records = [
        {"_source_index": index, "_cot_source": source,
         "reasoning_path": f"trace {index}/{source}",
         "history_item_sid": "['<a_1><b_1><c_1>']", "item_sid": "<a_2><b_2><c_2>"}
        for index, sources in rows.items() for source in sources
    ]
    return pd.DataFrame(records)


def delta_frame(values):
    """values: {(source_index, cot_source): delta}."""
    return pd.DataFrame([
        {"_source_index": index, "_cot_source": source, "delta": delta}
        for (index, source), delta in values.items()
    ])


class TestSelectBestOfN(unittest.TestCase):
    def test_takes_the_largest_not_the_first_positive(self):
        expanded = expanded_frame({0: [0, 1, 2, 3, 4]})
        deltas = delta_frame({(0, 0): -1.0, (0, 1): 0.2, (0, 2): 9.0,
                              (0, 3): 5.0, (0, 4): 7.0})
        selected, stats = select_best_of_n(expanded, deltas, 5)
        self.assertEqual(len(selected), 1)
        # first-accept would take source 1; best-of-N must take source 2.
        self.assertEqual(int(selected.iloc[0]["_cot_source"]), 2)
        self.assertAlmostEqual(float(selected.iloc[0]["_cot_delta"]), 9.0)
        self.assertEqual(stats["rows_accepted"], 1)
        self.assertEqual(stats["rows_rejected"], 0)
        self.assertAlmostEqual(stats["gain_over_first_accept"], 8.8)
        self.assertAlmostEqual(stats["frac_rows_differ_from_first_accept"], 1.0)
        self.assertAlmostEqual(stats["gain_over_random_pick"], 9.0 - 4.04)

    def test_does_not_keep_v6_when_a_resample_scores_higher(self):
        expanded = expanded_frame({0: [0, 1, 2]})
        deltas = delta_frame({(0, 0): 0.01, (0, 1): 8.0, (0, 2): 3.0})
        selected, _ = select_best_of_n(expanded, deltas, 3)
        self.assertEqual(int(selected.iloc[0]["_cot_source"]), 1)

    def test_ties_go_to_the_lowest_source(self):
        expanded = expanded_frame({0: [0, 1, 2]})
        deltas = delta_frame({(0, 0): 8.0, (0, 1): 8.0, (0, 2): 3.0})
        selected, _ = select_best_of_n(expanded, deltas, 3)
        self.assertEqual(int(selected.iloc[0]["_cot_source"]), 0)

    def test_row_with_no_positive_delta_is_rejected(self):
        expanded = expanded_frame({0: [0, 1], 1: [0, 1]})
        deltas = delta_frame({(0, 0): -1.0, (0, 1): 0.0,
                              (1, 0): -0.5, (1, 1): 3.0})
        selected, stats = select_best_of_n(expanded, deltas, 2)
        self.assertEqual(list(selected["_source_index"]), [1])
        self.assertEqual(stats["rows_total"], 2)
        self.assertEqual(stats["rows_accepted"], 1)
        self.assertEqual(stats["rows_rejected"], 1)
        self.assertAlmostEqual(stats["acceptance_rate"], 0.5)
        # delta == 0 is not an accept: the bar is strictly "better than no CoT".
        self.assertEqual(stats["candidates_positive"], 1)

    def test_rejection_is_row_level_not_candidate_level(self):
        # A row survives iff *some* candidate clears the bar, and the retained
        # trace is always the argmax, never merely the first positive one.
        rows = {index: [0, 1, 2] for index in range(60)}
        expanded = expanded_frame(rows)
        deltas = delta_frame({
            (index, source): ((index % 7) - 3.0) + source * 0.5
            for index in range(60) for source in range(3)
        })
        best, best_stats = select_best_of_n(expanded, deltas, 3)
        expected_accepted = sum(
            1 for index in range(60)
            if max(((index % 7) - 3.0) + source * 0.5 for source in range(3)) > 0
        )
        self.assertEqual(best_stats["rows_accepted"], expected_accepted)
        self.assertEqual(best_stats["rows_rejected"], 60 - expected_accepted)
        # delta grows with source, so the argmax is always the last candidate.
        self.assertEqual(best_stats["selected_from_source"][2],
                         best_stats["rows_accepted"])
        self.assertTrue((best["_cot_delta"].to_numpy() > 0).all())

    def test_one_row_per_source_index(self):
        expanded = expanded_frame({index: [0, 1, 2] for index in range(50)})
        deltas = delta_frame({
            (index, source): (1.0 + source if source == index % 3 else -1.0)
            for index in range(50) for source in range(3)
        })
        selected, stats = select_best_of_n(expanded, deltas, 3)
        self.assertEqual(len(selected), selected["_source_index"].nunique())
        self.assertEqual(len(selected), 50)
        self.assertTrue((selected["_cot_delta"] > 0).all())
        # index % 3 is the only positive candidate, so it must always win.
        self.assertEqual(stats["selected_from_source"], {0: 17, 1: 17, 2: 16})

    def test_missing_delta_raises(self):
        expanded = expanded_frame({0: [0, 1]})
        deltas = delta_frame({(0, 0): 1.0})
        with self.assertRaises(ValueError):
            select_best_of_n(expanded, deltas, 2)

    def test_gaps_in_candidates_are_handled(self):
        # Row 0 lost its v6 trace upstream (empty reasoning_path); the argmax is
        # taken over whatever candidates survived.
        expanded = expanded_frame({0: [1, 3]})
        deltas = delta_frame({(0, 1): 2.0, (0, 3): 1.5})
        selected, stats = select_best_of_n(expanded, deltas, 5)
        self.assertEqual(int(selected.iloc[0]["_cot_source"]), 1)
        self.assertEqual(stats["selected_from_source"], {0: 0, 1: 1, 2: 0, 3: 0, 4: 0})
        self.assertAlmostEqual(stats["n_candidates_mean"], 2.0)

    def test_min_delta_threshold_is_respected(self):
        expanded = expanded_frame({0: [0, 1], 1: [0, 1]})
        deltas = delta_frame({(0, 0): 0.5, (0, 1): 2.0,
                              (1, 0): 0.5, (1, 1): 0.9})
        selected, stats = select_best_of_n(expanded, deltas, 2, min_delta=1.0)
        # Row 1's best candidate (0.9) is below the bar -> rejected.
        self.assertEqual(list(selected["_source_index"]), [0])
        self.assertEqual(int(selected.iloc[0]["_cot_source"]), 1)
        self.assertEqual(stats["rows_rejected"], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
