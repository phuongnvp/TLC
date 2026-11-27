"""Exercise chunk alignment, filtering, corruption checks, and result ordering."""

import csv
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from torch import nn

from ensemble import WEIGHT, normalize_ratios
from molecule_search import EmbeddingDatabase, ROOT, _hash_file


class TinyModel(nn.Module):
    def __init__(self, dimension):
        super().__init__()
        self.solvent_encoder = nn.Sequential(
            nn.Linear(3, 2), nn.ReLU(), nn.Linear(2, 1), nn.ReLU(),
        )
        self.regressor = nn.Sequential(nn.Linear(dimension + 1, 1), nn.Sigmoid())
        with torch.no_grad():
            for parameter in self.parameters():
                parameter.zero_()
            self.solvent_encoder[0].weight[0, 0] = 1
            self.solvent_encoder[2].weight[0, 0] = 1
            self.regressor[0].weight[0, 0] = 1
            self.regressor[0].weight[0, -1] = 0.2
        self.eval()


class TinyEnsemble:
    def __init__(self):
        self.solvents = ["n-Hexane", "Ethyl acetate"]
        self.checkpoint_hashes = {"AttentiveFP": "a" * 64, "MPNN": "b" * 64}
        self.attentive = SimpleNamespace(solvents=self.solvents, model=TinyModel(4))
        self.mpnn = SimpleNamespace(solvents=self.solvents, model=TinyModel(2))
        self.features = {"CC": -2.0, "CCO": 0.2, "CCC": 0.0, "CCCC": 2.0}

    @torch.inference_mode()
    def predict(self, smiles, names, ratios):
        mixture = normalize_ratios(names, ratios, self.solvents)
        values = []
        for predictor, dimension in ((self.attentive, 4), (self.mpnn, 2)):
            molecular = torch.zeros(1, dimension)
            molecular[0, 0] = self.features[smiles]
            solvent = torch.tensor(
                [[mixture.get(s, 0.0) for s in self.solvents] + [0.0]], dtype=torch.float32,
            )
            combined = torch.cat((molecular, predictor.model.solvent_encoder(solvent)), dim=1)
            values.append(float(predictor.model.regressor(combined).item()))
        return {"AttentiveFP": values[0], "MPNN": values[1], "Rf": WEIGHT * values[0] + (1 - WEIGHT) * values[1]}


class DatabaseFixture:
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name)
        self.ensemble = TinyEnsemble()
        self.identity = {
            "schema_version": 1, "dtype": "float32", "dataset_rows": 5,
            "dimensions": [4, 2], "model_sha256": self.ensemble.checkpoint_hashes,
            "encoding_sha256": {name: _hash_file(ROOT / name) for name in ("ensemble.py", "graph_encoding.py")},
        }
        self.run_id = hashlib.sha256(json.dumps(self.identity, sort_keys=True).encode()).hexdigest()
        self.write_json(self.folder / "manifest.json", {"run_id": self.run_id, "identity": self.identity})
        self.summary = {"run_id": self.run_id, "complete": True, "chunks": 2, "dataset_rows": 5,
                        "processed": 5, "embedded": 4, "failed": 1}
        self.write_json(self.folder / "summary.json", self.summary)
        self.chunk(0, 0, 3, [(0, "ID0", "CC"), (2, "ID2", "CCO")], [(1, "bad", "!", "invalid_structure", "invalid")])
        self.chunk(1, 3, 5, [(3, "ID3", "CCC"), (4, "ID4", "CCCC")], [])

    @staticmethod
    def write_json(path, value):
        path.write_text(json.dumps(value), encoding="utf-8")

    def chunk(self, index, start, stop, rows, failures):
        folder = self.folder / f"chunk_{index:06d}"
        folder.mkdir()
        for name, dimension in (("attentivefp.npy", 4), ("mpnn.npy", 2)):
            array = np.zeros((len(rows), dimension), dtype=np.float32)
            array[:, 0] = [self.ensemble.features[row[2]] for row in rows]
            np.save(folder / name, array)
        for name, columns, values in (
            ("molecules.csv", ["source_row", "identifier", "canonical_smiles"], rows),
            ("failures.csv", ["source_row", "identifier", "smiles", "reason", "detail"], failures),
        ):
            with (folder / name).open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(columns)
                writer.writerows(values)
        info = {"index": index, "run_id": self.run_id, "source_start": start, "source_stop": stop,
                "embedded": len(rows), "failed": len(failures),
                "sha256": {name: _hash_file(folder / name) for name in ("attentivefp.npy", "mpnn.npy", "molecules.csv", "failures.csv")}}
        self.write_json(folder / "chunk.json", info)


class MoleculeSearchTests(DatabaseFixture, unittest.TestCase):
    def test_predictions_alignment_and_ranking(self):
        database = EmbeddingDatabase(self.folder, self.ensemble)
        progress = []
        result = database.search(self.ensemble, self.ensemble.solvents, "7:3", 0.5, progress=lambda a, b: progress.append((a, b)))
        self.assertEqual(result.searched, 4)
        self.assertEqual(result.table["COCONUT ID"].tolist(), ["ID3", "ID2"])
        self.assertEqual(progress[-1], (4, 4))
        for row in result.table.to_dict("records"):
            expected = self.ensemble.predict(row["SMILES"], self.ensemble.solvents, "7:3")["Rf"]
            self.assertAlmostEqual(row["Predicted Rf"], expected, places=7)
            self.assertLessEqual(row["Absolute Rf difference"], 0.1)

    def test_batch_size_and_ratio_normalization(self):
        database = EmbeddingDatabase(self.folder, self.ensemble)
        first = database.search(self.ensemble, self.ensemble.solvents, "7:3", 0.5, batch_size=1)
        second = database.search(self.ensemble, self.ensemble.solvents, "70:30", 0.5, batch_size=4)
        np.testing.assert_allclose(first.table["Predicted Rf"], second.table["Predicted Rf"])

    def test_exact_target_and_empty_results(self):
        database = EmbeddingDatabase(self.folder, self.ensemble)
        exact = self.ensemble.predict("CCC", self.ensemble.solvents, "7:3")["Rf"]
        result = database.search(self.ensemble, self.ensemble.solvents, "7:3", exact, tolerance=0)
        self.assertEqual(result.table["COCONUT ID"].tolist(), ["ID3"])
        empty = database.search(self.ensemble, self.ensemble.solvents, "7:3", 0, tolerance=0)
        self.assertTrue(empty.table.empty)
        self.assertIn("Rank", empty.table.columns)

    def test_invalid_query(self):
        database = EmbeddingDatabase(self.folder, self.ensemble)
        for target in (-1, 2, float("nan")):
            with self.assertRaises(ValueError):
                database.search(self.ensemble, self.ensemble.solvents, "7:3", target)
        with self.assertRaises(ValueError):
            database.search(self.ensemble, ["unsupported"], "1", 0.5)
        with self.assertRaises(ValueError):
            database.search(self.ensemble, self.ensemble.solvents, "0:0", 0.5)

    def test_corrupt_chunk_rejected(self):
        with (self.folder / "chunk_000000" / "mpnn.npy").open("ab") as stream:
            stream.write(b"corrupt")
        with self.assertRaisesRegex(ValueError, "Corrupt"):
            EmbeddingDatabase(self.folder, self.ensemble)

    def test_changed_files_rejected_after_initial_validation(self):
        database = EmbeddingDatabase(self.folder, self.ensemble)
        with (self.folder / "chunk_000000" / "failures.csv").open("a") as stream:
            stream.write("\n")
        with self.assertRaisesRegex(ValueError, "changed"):
            database.search(self.ensemble, self.ensemble.solvents, "7:3", 0.5)

    def test_incomplete_database_rejected(self):
        self.summary["complete"] = False
        self.write_json(self.folder / "summary.json", self.summary)
        with self.assertRaisesRegex(ValueError, "incomplete"):
            EmbeddingDatabase(self.folder, self.ensemble)

    def test_model_mismatch_rejected(self):
        self.ensemble.checkpoint_hashes = {"AttentiveFP": "different", "MPNN": "different"}
        with self.assertRaisesRegex(ValueError, "different model"):
            EmbeddingDatabase(self.folder, self.ensemble)


if __name__ == "__main__":
    unittest.main()
