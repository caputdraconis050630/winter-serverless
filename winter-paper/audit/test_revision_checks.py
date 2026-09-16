"""Focused checks for retained-data interpretation and document publication."""
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import build_paper
from build_nondrift_evidence import strict_al

ROOT = Path(__file__).resolve().parent.parent


class RevisionChecks(unittest.TestCase):
    def test_same_total_difference_has_two_normalizations(self):
        data = json.loads((ROOT / "audit/onboarding_metrics.json").read_text())
        for cohort in data.values():
            for rows in cohort["by_rho"].values():
                for row in rows.values():
                    self.assertAlmostEqual(row["delta_csr_pp"] * cohort["invocations"] / 100,
                                           row["delta_cold_per_fn"] * cohort["n_functions"], places=7)

    def test_matched_memory_is_unavailable_not_zero(self):
        rows = json.loads((ROOT / "audit/matched_memory.json").read_text())
        self.assertEqual(len(rows), 12)
        for row in rows:
            self.assertIsNone(row["interpolated_delta_csr_pp"])
            self.assertGreater(row["wm_target"], max(p[0] for p in row["frontier"]))

    def test_al_requires_all_31_ticks(self):
        self.assertIsNone(strict_al([1.] * 30)[0])
        self.assertEqual(strict_al([1.] * 31)[0], 0)
        self.assertIsNone(strict_al([2.] * 35 + [1.] * 20)[0])
        self.assertIsNone(strict_al([np.nan] + [1.] * 30)[0])

    def test_first_hour_share_uses_unsmoothed_counts(self):
        data = json.loads((ROOT / "audit/onboarding_metrics.json").read_text())
        for cohort in data.values():
            for rows in cohort["by_rho"].values():
                for arm in ("A5_proto", "gated_v3"):
                    row = rows[arm]
                    self.assertAlmostEqual(row["first_hour_share"],
                                           row["avoided_first_hour_per_fn"] / row["avoided_four_hours_per_fn"])

    def test_late_reference_failure_preserves_all_public_pdfs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ("main", "supplementary", "highlights"):
                (root / (name + ".pdf")).write_bytes(b"original")
                (root / (name + ".tex")).write_text("test source")

            def fake_run(command, **kwargs):
                stage = Path(kwargs["cwd"])
                if command[0] == "pdflatex":
                    name = Path(command[-1]).stem
                    (stage / (name + ".pdf")).write_bytes(b"new")
                    message = f"Output written on {name}.pdf (1 page, 100 bytes)."
                    if name == "supplementary":
                        message += "\nLaTeX Warning: There were undefined references."
                    (stage / (name + ".log")).write_text(message)
                return subprocess.CompletedProcess(command, 0, stdout="test")

            with patch.object(build_paper, "ROOT", root), patch.object(build_paper.subprocess, "run", fake_run):
                with self.assertRaisesRegex(RuntimeError, "Unresolved document references"):
                    build_paper.build(root / "build")
            for name in ("main", "supplementary", "highlights"):
                self.assertEqual((root / (name + ".pdf")).read_bytes(), b"original")


if __name__ == "__main__":
    unittest.main()
