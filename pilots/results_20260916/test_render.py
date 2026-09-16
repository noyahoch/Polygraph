"""Offline rendering contracts; no browser/network or scientific execution."""
import copy
import json
from pathlib import Path
import re
import tempfile
import unittest

from .render import render, render_site


class OfflineRenderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = {"schema_version": 1, "title": "Test result", "studies": {
            "layers": {"seeds": [], "primary": {}, "all3": {}},
            "topology": {"primary": {}, "baselines": [], "contrasts": []}},
            "insights": [], "analysis_notes": [], "csv_rows": [], "provenance": {}}

    def test_render_is_self_contained_and_preserves_json(self):
        output = self.root / "index.html"
        render(copy.deepcopy(self.data), output)
        text = output.read_text()
        payloads = re.findall(r'<script\b[^>]*type=["\']application/json["\'][^>]*>(.*?)</script>',
                              text, re.DOTALL)
        self.assertEqual(len(payloads), 1)
        self.assertEqual(json.loads(payloads[0]), self.data)
        self.assertNotRegex(text, r'<(?:script|link)\b[^>]*(?:src|href)=["\']https?://')
        self.assertNotIn("__RESULTS_JSON__", text)
        self.assertIn("Download CSV", text)

    def test_script_payload_cannot_close_embedding(self):
        self.data["insights"] = ['</script><script>alert("not executed")</script>', "<!--test-->", "\u2028\u2029"]
        output = self.root / "index.html"
        render(self.data, output)
        text = output.read_text()
        self.assertNotIn('</script><script>alert("not executed")</script>', text)
        payloads = re.findall(r'<script\b[^>]*type=["\']application/json["\'][^>]*>(.*?)</script>',
                              text, re.DOTALL)
        self.assertEqual(json.loads(payloads[0])["insights"], self.data["insights"])

    def test_nonfinite_metrics_are_not_serialized(self):
        self.data["studies"]["layers"]["primary"]["estimate"] = float("nan")
        with self.assertRaises(ValueError):
            render(self.data, self.root / "invalid.html")
        self.assertFalse((self.root / "invalid.html").exists())

    def test_site_separates_journey_and_full_explanation(self):
        self.data["studies"]["layers"]["seeds"] = [
            {"seed": seed, "status": "complete_late_diagnostic" if seed == 7 else "complete",
             "metrics": {"learned_stack": .89, "learned_last_only": .88}, "delta": .01,
             "eda": {"large_descriptive_section": [1, 2, 3]}} for seed in (7, 17, 27)]
        render_site(self.data, self.root / "index.html")
        summary = (self.root / "index.html").read_text()
        details = (self.root / "details.html").read_text()
        self.assertIn('href="details.html"', summary)
        self.assertIn('href="index.html"', details)
        self.assertIn("The journey", summary)
        self.assertNotIn("large_descriptive_section", summary)
        self.assertIn("large_descriptive_section", details)
        self.assertIn("Download CSV", details)

    def test_site_rejects_broken_navigation_filename(self):
        with self.assertRaisesRegex(ValueError, "index.html"):
            render_site(self.data, self.root / "different.html")
        self.assertFalse((self.root / "different.html").exists())


if __name__ == "__main__":
    unittest.main()
