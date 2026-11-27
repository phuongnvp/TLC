"""Check page results, exports, and stale-result handling without a browser."""

from contextlib import nullcontext
import importlib.util
import io
import os
from types import ModuleType, SimpleNamespace
import sys
import unittest
from unittest.mock import patch

import pandas as pd

from molecule_search import ROOT
from test_molecule_search import DatabaseFixture


class PageUI(ModuleType):
    def __init__(self):
        super().__init__("streamlit")
        self.session_state = {}
        self.events = []
        self.clicked = True
        self.target = 0.5
        self.zero_parts = False
        self.column_config = SimpleNamespace(NumberColumn=lambda **kwargs: kwargs)

    def cache_resource(self, **kwargs):
        return lambda function: function

    def multiselect(self, label, options, default, **kwargs):
        return default

    def number_input(self, label, *args, **kwargs):
        if label == "Target Rf":
            return self.target
        return 0.0 if self.zero_parts else kwargs["value"]

    def button(self, *args, **kwargs):
        return self.clicked

    def spinner(self, *args, **kwargs):
        return nullcontext()

    def progress(self, *args, **kwargs):
        return SimpleNamespace(progress=lambda *args, **kwargs: None, empty=lambda: None)

    def dataframe(self, table, **kwargs):
        self.events.append(("table", table.copy()))

    def download_button(self, label, data, *args, **kwargs):
        self.events.append(("download", data))

    def __getattr__(self, name):
        if name in {"title", "markdown", "caption", "info", "write", "warning", "error", "subheader"}:
            return lambda value, **kwargs: self.events.append((name, value))
        raise AttributeError(name)


class MoleculeSearchPageTests(DatabaseFixture, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.ui = PageUI()
        spec = importlib.util.spec_from_file_location("_test_molecule_search_page", ROOT / "molecule_search_page.py")
        self.page = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"streamlit": self.ui}):
            spec.loader.exec_module(self.page)

    def render(self, folder=None):
        with patch.dict(os.environ, {"TLC_EMBEDDINGS_DIR": str(folder or self.folder)}):
            self.page.render_molecule_search(self.ensemble)

    def test_search_and_csv_include_query_conditions(self):
        self.render()
        self.assertFalse([value for name, value in self.ui.events if name == "error"])
        table = next(value for name, value in self.ui.events if name == "table")
        self.assertEqual(table["COCONUT ID"].tolist(), ["ID3", "ID2"])
        exported = next(value for name, value in self.ui.events if name == "download")
        csv = pd.read_csv(io.BytesIO(exported))
        self.assertEqual(len(csv), 2)
        self.assertTrue((csv["Tolerance"] == 0.1).all())
        self.assertTrue((csv["Target Rf"] == 0.5).all())
        self.assertIn("70.00000000%", csv["Mobile phase"].iloc[0])

    def test_changed_query_hides_previous_results(self):
        self.render()
        self.ui.events.clear()
        self.ui.clicked = False
        self.ui.target = 0.0
        self.render()
        self.assertFalse(any(name in {"table", "download"} for name, _ in self.ui.events))

    def test_invalid_ratio_clears_previous_results(self):
        self.render()
        self.ui.events.clear()
        self.ui.zero_parts = True
        self.render()
        self.assertTrue(any(name == "error" for name, _ in self.ui.events))
        self.assertNotIn("molecule_search_result", self.ui.session_state)
        self.assertFalse(any(name == "table" for name, _ in self.ui.events))

    def test_missing_database_gives_setup_message(self):
        self.render(self.folder / "missing")
        self.assertTrue(any(name == "info" and "coconut_embeddings" in value for name, value in self.ui.events))


if __name__ == "__main__":
    unittest.main()
