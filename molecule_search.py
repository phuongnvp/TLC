"""Search persisted COCONUT embeddings without rebuilding molecular graphs."""

from dataclasses import dataclass
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
import time

import numpy as np
import pandas as pd
import torch

from ensemble import Ensemble, WEIGHT, normalize_ratios


ROOT = Path(__file__).resolve().parent
RESULT_COLUMNS = [
    "COCONUT ID", "SMILES", "Predicted Rf", "Absolute Rf difference",
    "AttentiveFP Rf", "MPNN Rf",
]
ARTIFACTS = ("attentivefp.npy", "mpnn.npy", "molecules.csv", "failures.csv")


def _hash_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _text_hash_matches(path, expected):
    """Allow Git's LF/CRLF conversion without accepting other source changes."""
    raw = path.read_bytes()
    lf = raw.replace(b"\r\n", b"\n")
    return any(
        hashlib.sha256(content).hexdigest() == expected
        for content in (raw, lf, lf.replace(b"\n", b"\r\n"))
    )


def _artifact_hash_matches(path, expected):
    if path.name in ("molecules.csv", "failures.csv"):
        return _text_hash_matches(path, expected)
    # Embedding arrays are binary: any byte change must still be rejected.
    return _hash_file(path) == expected


def _read_json(path):
    try:
        with path.open(encoding="utf-8") as stream:
            return json.load(stream)
    except (OSError, ValueError) as exc:
        raise ValueError(f"Cannot read embedding metadata: {path.name}: {exc}") from exc


def _close_array(array):
    mapping = getattr(array, "_mmap", None)
    if mapping is not None:
        mapping.close()


def _signature(path):
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns


def checkpoint_hashes(ensemble):
    known = getattr(ensemble, "checkpoint_hashes", None)
    if known is not None:
        return dict(known)
    # app.py assembles AttentiveFP in a temporary directory that no longer
    # exists after loading. Fall back to the original parts beside MPNN.
    mpnn = Path(ensemble.mpnn.path)
    attentive = Path(ensemble.attentive.path)
    if attentive.is_file():
        attentive_hash = _hash_file(attentive)
    else:
        digest = hashlib.sha256()
        for index in range(5):
            with (mpnn.parent / f"AttentiveFP.pt.part{index}").open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
        attentive_hash = digest.hexdigest()
    return {"AttentiveFP": attentive_hash, "MPNN": _hash_file(mpnn)}


@torch.inference_mode()
def _score_arrays(ensemble, arrays, mixture, batch_size):
    predictions = []
    for predictor, molecular in zip((ensemble.attentive, ensemble.mpnn), arrays):
        device = next(predictor.model.parameters()).device
        solvent = torch.tensor(
            [[mixture.get(name, 0.0) for name in predictor.solvents] + [0.0]],
            dtype=torch.float32, device=device,
        )
        solvent_embedding = predictor.model.solvent_encoder(solvent)
        values = []
        for start in range(0, len(molecular), batch_size):
            # Copy only the active batch: mmap arrays are read-only and must
            # not be retained by a tensor when the chunk's mapping closes.
            batch = torch.from_numpy(
                np.array(molecular[start:start + batch_size], dtype=np.float32, copy=True)
            ).to(device)
            if not bool(torch.isfinite(batch).all()):
                raise ValueError("Embedding data contains NaN or infinity.")
            combined = torch.cat(
                (batch, solvent_embedding.expand(len(batch), -1)), dim=1,
            )
            values.append(predictor.model.regressor(combined).flatten().cpu().numpy())
        prediction = np.concatenate(values) if values else np.empty(0, dtype=np.float32)
        if not np.isfinite(prediction).all():
            raise ValueError("Model produced a nonfinite Rf prediction.")
        predictions.append(prediction)
    attentive, mpnn = predictions
    # Match Ensemble.predict's Python-float weighting, including boundary tests.
    combined = WEIGHT * attentive.astype(np.float64) + (1 - WEIGHT) * mpnn.astype(np.float64)
    return attentive, mpnn, combined


@dataclass
class MoleculeMatches:
    table: pd.DataFrame
    searched: int
    elapsed_seconds: float
    mixture: dict
    target: float
    tolerance: float


class EmbeddingDatabase:
    """Immutable chunk index. Validate once; map one chunk per query at a time."""

    def __init__(self, folder, ensemble):
        self.folder = Path(folder).resolve()
        manifest_path = self.folder / "manifest.json"
        summary_path = self.folder / "summary.json"
        manifest = _read_json(manifest_path)
        summary = _read_json(summary_path)
        try:
            identity = manifest["identity"]
            run_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
            if manifest["run_id"] != run_id or summary["run_id"] != run_id:
                raise ValueError("Embedding manifest and summary do not identify the same run.")
            if identity["schema_version"] != 1 or identity["dtype"] != "float32":
                raise ValueError("Unsupported embedding format; expected schema 1, float32.")
            if summary["complete"] is not True:
                raise ValueError("Embedding preparation is incomplete. Finish it before searching.")
            if identity["model_sha256"] != checkpoint_hashes(ensemble):
                raise ValueError("Embeddings were generated with different model checkpoints.")
            if set(identity["encoding_sha256"]) != {"ensemble.py", "graph_encoding.py"}:
                raise ValueError("Unsupported embedding encoder metadata.")
            for name, expected in identity["encoding_sha256"].items():
                if not _text_hash_matches(ROOT / name, expected):
                    raise ValueError(f"Embeddings use a different version of {name}.")
            self.dimensions = identity["dimensions"]
            expected_dimensions = [
                predictor.model.regressor[0].in_features
                - predictor.model.solvent_encoder[-2].out_features
                for predictor in (ensemble.attentive, ensemble.mpnn)
            ]
            if self.dimensions != expected_dimensions:
                raise ValueError("Embedding dimensions do not match the loaded models.")
            if any(predictor.model.training for predictor in (ensemble.attentive, ensemble.mpnn)):
                raise ValueError("Models must be in evaluation mode before searching.")
            self._signatures = {
                manifest_path: _signature(manifest_path), summary_path: _signature(summary_path),
            }
            self.chunks = []
            processed = embedded = failed = 0
            folders = sorted(self.folder.glob("chunk_*"))
            if len(folders) != summary["chunks"]:
                raise ValueError("Embedding folder has missing or unexpected chunks.")
            for index, folder in enumerate(folders):
                if not folder.is_dir() or folder.name != f"chunk_{index:06d}":
                    raise ValueError("Embedding chunks must be contiguous and numbered from zero.")
                info = _read_json(folder / "chunk.json")
                if (info["run_id"] != run_id or info["index"] != index
                        or info["source_start"] != processed or info["source_stop"] <= processed
                        or info["embedded"] < 0 or info["failed"] < 0
                        or info["embedded"] + info["failed"] != info["source_stop"] - processed):
                    raise ValueError(f"Invalid row counts or run identity in {folder.name}.")
                if set(info["sha256"]) != set(ARTIFACTS):
                    raise ValueError(f"Unexpected artifacts in {folder.name}.")
                for name, expected in info["sha256"].items():
                    path = folder / name
                    before = _signature(path)
                    if not _artifact_hash_matches(path, expected) or _signature(path) != before:
                        raise ValueError(f"Corrupt or changed embedding file: {folder.name}/{name}")
                    self._signatures[path] = before
                self._signatures[folder / "chunk.json"] = _signature(folder / "chunk.json")
                for name, dim in zip(ARTIFACTS[:2], self.dimensions):
                    array = np.load(folder / name, mmap_mode="r", allow_pickle=False)
                    try:
                        if array.shape != (info["embedded"], dim) or array.dtype != np.float32:
                            raise ValueError(f"Invalid array shape/type in {folder.name}/{name}.")
                    finally:
                        _close_array(array)
                # Check alignment and rejected-row accounting once, before caching.
                valid_rows = self._metadata(folder, info)
                rejected_rows = pd.read_csv(folder / "failures.csv", keep_default_na=False)
                if list(rejected_rows.columns) != ["source_row", "identifier", "smiles", "reason", "detail"]:
                    raise ValueError(f"Unexpected failures.csv schema in {folder.name}.")
                source_rows = valid_rows["source_row"].tolist() + rejected_rows["source_row"].tolist()
                if (len(rejected_rows) != info["failed"]
                        or sorted(source_rows) != list(range(processed, info["source_stop"]))):
                    raise ValueError(f"Missing or duplicated source rows in {folder.name}.")
                self.chunks.append((folder, info))
                processed = info["source_stop"]
                embedded += info["embedded"]
                failed += info["failed"]
            if (processed != identity["dataset_rows"] or summary["dataset_rows"] != processed
                    or summary["processed"] != processed or summary["embedded"] != embedded
                    or summary["failed"] != failed or embedded == 0):
                raise ValueError("Embedding totals do not match the manifest and summary.")
            self.count = embedded
            self.failed = failed
            self.run_id = run_id
        except (KeyError, TypeError, IndexError) as exc:
            raise ValueError("Embedding metadata is malformed or incomplete.") from exc
        self.validate_predictions(ensemble)

    @staticmethod
    def _metadata(folder, info):
        table = pd.read_csv(
            folder / "molecules.csv", dtype={"identifier": str, "canonical_smiles": str},
            keep_default_na=False,
        )
        if (list(table.columns) != ["source_row", "identifier", "canonical_smiles"]
                or len(table) != info["embedded"]
                or not table["identifier"].str.strip().ne("").all()
                or not table["canonical_smiles"].str.strip().ne("").all()):
            raise ValueError(f"Molecule metadata is not aligned in {folder.name}.")
        return table

    def _assert_unchanged(self):
        for path, signature in self._signatures.items():
            if _signature(path) != signature:
                raise ValueError("Embedding files changed. Restart the app to validate them again.")

    def _arrays(self, folder):
        arrays = []
        try:
            for name in ARTIFACTS[:2]:
                arrays.append(np.load(folder / name, mmap_mode="r", allow_pickle=False))
            return arrays
        except Exception:
            for array in arrays:
                _close_array(array)
            raise

    def validate_predictions(self, ensemble, tolerance=1e-5):
        """Check producer/consumer package differences against direct CPU inference."""
        if any(next(p.model.parameters()).device.type != "cpu" for p in (ensemble.attentive, ensemble.mpnn)):
            raise ValueError("The database search uses the app's CPU models.")
        folder, info = next((folder, info) for folder, info in self.chunks if info["embedded"])
        metadata = self._metadata(folder, info).head(3)
        names = ["n-Hexane", "Ethyl acetate"]
        if not all(name in ensemble.solvents for name in names):
            names = [ensemble.solvents[0]]
        ratios = "7:3" if len(names) == 2 else "1"
        mixture = normalize_ratios(names, ratios, ensemble.solvents)
        arrays = self._arrays(folder)
        try:
            predictions = _score_arrays(ensemble, [a[:len(metadata)] for a in arrays], mixture, 3)
            for index, smiles in enumerate(metadata["canonical_smiles"]):
                direct = ensemble.predict(smiles, names, ratios)
                for values, name in zip(predictions, ("AttentiveFP", "MPNN", "Rf")):
                    value = float(values[index])
                    if not math.isfinite(direct[name]) or abs(value - direct[name]) > tolerance:
                        raise ValueError("Saved embeddings do not reproduce direct model predictions.")
        finally:
            for array in arrays:
                _close_array(array)

    def search(self, ensemble, names, ratios, target, tolerance=0.1, batch_size=1024, progress=None):
        if (not math.isfinite(target) or not 0 <= target <= 1
                or not math.isfinite(tolerance) or not 0 <= tolerance <= 1):
            raise ValueError("Target Rf and tolerance must be between zero and one.")
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("Batch size must be a positive integer.")
        mixture = normalize_ratios(names, ratios, ensemble.solvents)
        self._assert_unchanged()
        started = time.perf_counter()
        parts, searched = [], 0
        for folder, info in self.chunks:
            arrays = self._arrays(folder)
            try:
                attentive, mpnn, prediction = _score_arrays(ensemble, arrays, mixture, batch_size)
            finally:
                for array in arrays:
                    _close_array(array)
            error = np.abs(prediction - target)
            matches = np.flatnonzero(error <= tolerance)
            if len(matches):
                metadata = self._metadata(folder, info).iloc[matches]
                parts.append(pd.DataFrame({
                    "COCONUT ID": metadata["identifier"].to_numpy(),
                    "SMILES": metadata["canonical_smiles"].to_numpy(),
                    "Predicted Rf": prediction[matches],
                    "Absolute Rf difference": error[matches],
                    "AttentiveFP Rf": attentive[matches], "MPNN Rf": mpnn[matches],
                }))
            searched += info["embedded"]
            if progress:
                progress(searched, self.count)
        self._assert_unchanged()
        table = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=RESULT_COLUMNS)
        table = table.sort_values(
            ["Absolute Rf difference", "COCONUT ID"], kind="stable", ignore_index=True,
        )
        table.insert(0, "Rank", range(1, len(table) + 1))
        return MoleculeMatches(table, searched, time.perf_counter() - started, mixture, target, tolerance)


def _load_ensemble(models):
    with TemporaryDirectory() as temporary:
        attentive = models / "AttentiveFP.pt"
        if not attentive.exists():
            attentive = Path(temporary) / "AttentiveFP.pt"
            with attentive.open("wb") as destination:
                for index in range(5):
                    with (models / f"AttentiveFP.pt.part{index}").open("rb") as stream:
                        for block in iter(lambda: stream.read(1024 * 1024), b""):
                            destination.write(block)
        mpnn = models / "MPNN.pt"
        ensemble = Ensemble(attentive, mpnn)
        ensemble.checkpoint_hashes = {"AttentiveFP": _hash_file(attentive), "MPNN": _hash_file(mpnn)}
        return ensemble


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings", type=Path, default=ROOT / "coconut_embeddings")
    parser.add_argument("--models", type=Path, default=ROOT / "models")
    parser.add_argument("--solvents", nargs="+", default=["n-Hexane", "Ethyl acetate"])
    parser.add_argument("--ratios", default="7:3")
    parser.add_argument("--target", type=float, default=0.5)
    parser.add_argument("--tolerance", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--threads", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.threads:
        if args.threads < 1:
            parser.error("Threads must be positive.")
        torch.set_num_threads(args.threads)
    if args.output and args.output.exists():
        parser.error("Output already exists; choose another filename.")
    ensemble = _load_ensemble(args.models)
    print("Verifying embedding files and prediction consistency...", flush=True)
    started = time.perf_counter()
    database = EmbeddingDatabase(args.embeddings, ensemble)
    print(f"Verified {database.count:,} structures in {time.perf_counter() - started:.2f}s", flush=True)
    result = database.search(
        ensemble, args.solvents, args.ratios, args.target, args.tolerance, args.batch_size,
    )
    print(f"Scored {result.searched:,} structures in {result.elapsed_seconds:.2f}s; "
          f"found {len(result.table):,} matches.", flush=True)
    print(result.table.head(10).to_string(index=False), flush=True)
    if args.output:
        with args.output.open("x", encoding="utf-8", newline="") as stream:
            result.table.to_csv(stream, index=False)
        print(f"Saved {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
