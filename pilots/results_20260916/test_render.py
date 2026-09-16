"""Offline rendering contracts; no browser/network or scientific execution."""
import copy
import json
from pathlib import Path
import re
import tempfile
import unittest

from .render import render


class OfflineRenderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = {"schema_version": 1, "title": "Test result", "studies": {
            "layers": {"seeds": [], "primary": {}, "all3": {}},
            "topology": {"baselines": [], "contrasts": []}},
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


if __name__ == "__main__":
    unittest.main()
