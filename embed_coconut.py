"""Create resumable float32 COCONUT embeddings using the bundled TLC models.

Run beside ensemble.py and graph_encoding.py. Example for Colab:
python -u embed_coconut.py --csv /content/coconut.csv --device cuda \
    --output /content/drive/MyDrive/TLC/coconut_embeddings --batch-size 32

Rerun the same command with --resume after interruption. Only run one writer
per output folder. Temporary/incomplete chunks are ignored on resume; committed
chunks are verified using their checksums. This script does not modify the CSV,
model checkpoints, or application code. It does not train any models.
"""

import argparse
import csv
import gc
import hashlib
import itertools
import json
import math
import os
import shutil
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch_geometric
from rdkit import rdBase

from ensemble import (
    Batch, Ensemble, WEIGHT, canonicalize_smiles, normalize_ratios, smiles_to_graph,
)
from graph_encoding import encode_attentive, encode_mpnn


ROOT = Path(__file__).resolve().parent
SCHEMA_VERSION = 1
MODEL_HASHES = {
    "AttentiveFP": "b0ce3c93834d03578d094f7f7b7dc2429fb35892d6dd4d36478c8f27f909d5ba",
    "MPNN": "54d46ae50ec5027e2a07f9dd43cbfeb4481a9dc5f9436f104cd7f924d2dca4be",
}
MOLECULE_COLUMNS = ["source_row", "identifier", "canonical_smiles"]
FAILURE_COLUMNS = ["source_row", "identifier", "smiles", "reason", "detail"]


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    # A temporary sibling keeps replacement on the same filesystem.
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent,
        prefix=".json-", suffix=".tmp", delete=False,
    ) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def input_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["identifier", "canonical_smiles"]:
            raise ValueError(
                "Use the cleaned CSV with exactly these columns: "
                "identifier,canonical_smiles"
            )
        for source_row, row in enumerate(reader):
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"Malformed CSV record near physical line {reader.line_num}.")
            yield source_row, row["identifier"].strip(), row["canonical_smiles"].strip()


def load_models(models):
    with tempfile.TemporaryDirectory() as temporary:
        attentive = models / "AttentiveFP.pt"
        if not attentive.exists():
            attentive = Path(temporary) / "AttentiveFP.pt"
            with attentive.open("wb") as destination:
                for index in range(5):
                    with (models / f"AttentiveFP.pt.part{index}").open("rb") as source:
                        shutil.copyfileobj(source, destination)
        mpnn = models / "MPNN.pt"
        hashes = {"AttentiveFP": file_hash(attentive), "MPNN": file_hash(mpnn)}
        if hashes != MODEL_HASHES:
            raise ValueError("Model checksums differ from the app's bundled checkpoints.")
        return Ensemble(attentive, mpnn), hashes


def resolve_device(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA unavailable. Check the GPU runtime and CUDA-enabled PyTorch.")
    return torch.device(name)


@torch.inference_mode()
def encode_batch(ensemble, graphs, device):
    batch = None
    attentive = mpnn = None
    try:
        batch = Batch.from_data_list(graphs).to(device)
        attentive = encode_attentive(ensemble.attentive.model, batch).cpu().numpy().copy()
        mpnn = encode_mpnn(ensemble.mpnn.model, batch).cpu().numpy().copy()
        return attentive, mpnn
    finally:
        del batch, attentive, mpnn


@torch.inference_mode()
def preflight(ensemble, device, tolerance):
    """Compare batched embeddings on the chosen device with original CPU inference."""
    smiles = ["CCO", "c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O"]
    names = ["n-Hexane", "Ethyl acetate"]
    if not all(name in ensemble.solvents for name in names):
        if not ensemble.solvents:
            raise ValueError("The two models share no supported solvents.")
        names = [ensemble.solvents[0]]
    ratios = ["7:3", "3:7"] if len(names) == 2 else ["1"]
    references = {
        ratio: [ensemble.predict(smile, names, ratio) for smile in smiles]
        for ratio in ratios
    }
    for predictor in (ensemble.attentive, ensemble.mpnn):
        predictor.model.to(device).eval()
    graphs = [smiles_to_graph(canonicalize_smiles(smile)) for smile in smiles]
    embeddings = encode_batch(ensemble, graphs, device)
    errors = {"AttentiveFP": 0.0, "MPNN": 0.0, "Rf": 0.0}
    for ratio in ratios:
        predictions = []
        for predictor, molecular in zip((ensemble.attentive, ensemble.mpnn), embeddings):
            mixture = normalize_ratios(names, ratio, predictor.solvents)
            solvent = torch.tensor(
                [[mixture.get(name, 0.0) for name in predictor.solvents] + [0.0]],
                dtype=torch.float32, device=device,
            )
            encoded = predictor.model.solvent_encoder(solvent)
            combined = torch.cat(
                (torch.from_numpy(molecular).to(device), encoded.expand(len(smiles), -1)),
                dim=1,
            )
            predictions.append(predictor.model.regressor(combined).flatten().cpu().tolist())
        predictions.append([
            WEIGHT * a + (1 - WEIGHT) * b for a, b in zip(*predictions)
        ])
        for values, name in zip(predictions, errors):
            for index, value in enumerate(values):
                expected = references[ratio][index][name]
                if not math.isfinite(value) or not math.isfinite(expected):
                    raise RuntimeError("Nonfinite result during prediction consistency check.")
                errors[name] = max(errors[name], abs(value - expected))
    if max(errors.values()) > tolerance:
        raise RuntimeError(f"CPU/device prediction consistency failed: {errors}")
    print(f"Prediction consistency: PASS {errors}", flush=True)
    return [array.shape[1] for array in embeddings], errors


def failure(record, reason, detail):
    source_row, identifier, smiles = record
    return [source_row, identifier, smiles, reason, str(detail)[:1000]]


def process_batch(ensemble, entries, device, dimensions):
    """On GPU OOM, retry smaller batches. Other execution errors stop the run."""
    oom = False
    try:
        arrays = encode_batch(ensemble, [entry[2] for entry in entries], device)
    except torch.cuda.OutOfMemoryError:
        oom = True
    if oom:
        # Retry outside the except block so traceback-held tensors can be freed.
        gc.collect()
        torch.cuda.empty_cache()
        if len(entries) == 1:
            rejected = [failure(entries[0][0], "gpu_out_of_memory", "Failed even at batch size 1")]
            return [], [np.empty((0, dim), dtype=np.float32) for dim in dimensions], rejected
        middle = len(entries) // 2
        print(f"GPU memory limit: retrying batch {len(entries)} as smaller batches.", flush=True)
        left = process_batch(ensemble, entries[:middle], device, dimensions)
        right = process_batch(ensemble, entries[middle:], device, dimensions)
        return (
            left[0] + right[0],
            [np.concatenate((a, b)) for a, b in zip(left[1], right[1])],
            left[2] + right[2],
        )
    if any(array.shape != (len(entries), dim) for array, dim in zip(arrays, dimensions)):
        raise RuntimeError("Unexpected embedding dimensions.")
    good = np.isfinite(arrays[0]).all(axis=1) & np.isfinite(arrays[1]).all(axis=1)
    metadata, rejected = [], []
    for index, (record, canonical, _) in enumerate(entries):
        if good[index]:
            metadata.append([record[0], record[1], canonical])
        else:
            rejected.append(failure(record, "nonfinite_embedding", "NaN or infinity"))
    return metadata, [array[good].astype(np.float32, copy=False) for array in arrays], rejected


def process_chunk(ensemble, records, device, dimensions, batch_size):
    metadata, rejected, pending = [], [], []
    arrays = [[], []]

    def flush():
        good_rows, encoded, bad_rows = process_batch(ensemble, pending, device, dimensions)
        metadata.extend(good_rows)
        rejected.extend(bad_rows)
        for chunks, values in zip(arrays, encoded):
            chunks.append(values)
        pending.clear()

    for record in records:
        _, identifier, smiles = record
        try:
            if not identifier or not smiles:
                raise ValueError("Missing identifier or SMILES")
            with rdBase.BlockLogs():
                canonical = canonicalize_smiles(smiles)
                graph = smiles_to_graph(canonical)
        except (ValueError, RuntimeError, OverflowError) as exc:
            rejected.append(failure(record, "invalid_structure", exc))
            continue
        pending.append((record, canonical, graph))
        if len(pending) == batch_size:
            flush()
    if pending:
        flush()
    encoded = [
        np.concatenate(chunks) if chunks else np.empty((0, dim), dtype=np.float32)
        for chunks, dim in zip(arrays, dimensions)
    ]
    return metadata, encoded, sorted(rejected, key=lambda row: row[0])


def write_csv(path, header, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


def commit_chunk(output, index, run_id, records, result, batch_size):
    metadata, arrays, rejected = result
    final = output / f"chunk_{index:06d}"
    if final.exists():
        raise FileExistsError(f"Refusing to overwrite {final}")
    if len(metadata) + len(rejected) != len(records):
        raise RuntimeError("Source rows were lost or duplicated within a chunk.")
    with tempfile.TemporaryDirectory(prefix=".incomplete-", dir=output) as temporary:
        folder = Path(temporary)
        np.save(folder / "attentivefp.npy", arrays[0], allow_pickle=False)
        np.save(folder / "mpnn.npy", arrays[1], allow_pickle=False)
        write_csv(folder / "molecules.csv", MOLECULE_COLUMNS, metadata)
        write_csv(folder / "failures.csv", FAILURE_COLUMNS, rejected)
        files = {
            name: file_hash(folder / name)
            for name in ("attentivefp.npy", "mpnn.npy", "molecules.csv", "failures.csv")
        }
        info = {
            "run_id": run_id, "index": index, "created_utc": utc_now(),
            "source_start": records[0][0], "source_stop": records[-1][0] + 1,
            "embedded": len(metadata), "failed": len(rejected),
            "batch_size": batch_size, "sha256": files,
        }
        atomic_json(folder / "chunk.json", info)
        # Publish only after every artifact and its manifest have been written.
        os.rename(folder, final)
    return info


def verify_chunks(output, run_id, dimensions):
    infos, next_row = [], 0
    for folder in sorted(output.glob("chunk_*")):
        if not folder.is_dir() or folder.name != f"chunk_{len(infos):06d}":
            raise ValueError(f"Unexpected chunk or gap: {folder}")
        with (folder / "chunk.json").open(encoding="utf-8") as stream:
            info = json.load(stream)
        if (info["run_id"] != run_id or info["index"] != len(infos)
                or info["source_start"] != next_row
                or info["source_stop"] <= next_row
                or info["embedded"] < 0 or info["failed"] < 0
                or info["embedded"] + info["failed"] != info["source_stop"] - next_row):
            raise ValueError(f"Invalid resume metadata: {folder}")
        expected_files = {"attentivefp.npy", "mpnn.npy", "molecules.csv", "failures.csv"}
        if set(info["sha256"]) != expected_files:
            raise ValueError(f"Unexpected chunk artifacts: {folder}")
        for name, expected in info["sha256"].items():
            if file_hash(folder / name) != expected:
                raise ValueError(f"Corrupt/incomplete artifact: {folder / name}")
        for name, dim in zip(("attentivefp.npy", "mpnn.npy"), dimensions):
            array = np.load(folder / name, mmap_mode="r", allow_pickle=False)
            if array.shape != (info["embedded"], dim) or array.dtype != np.float32:
                raise ValueError(f"Unexpected array shape/type: {folder / name}")
            del array
        infos.append(info)
        next_row = info["source_stop"]
        if len(infos) % 20 == 0:
            print(f"Verified {len(infos)} existing chunks...", flush=True)
    return infos, next_row


def save_summary(output, run_id, infos, total):
    processed = infos[-1]["source_stop"] if infos else 0
    summary = {
        "run_id": run_id, "updated_utc": utc_now(), "dataset_rows": total,
        "processed": processed, "embedded": sum(info["embedded"] for info in infos),
        "failed": sum(info["failed"] for info in infos),
        "complete": processed == total, "chunks": len(infos),
    }
    atomic_json(output / "summary.json", summary)
    return summary


@torch.inference_mode()
def run(args):
    if min(args.batch_size, args.chunk_rows) < 1 or not 0 < args.atol < 1:
        raise ValueError("Batch/chunk sizes must be positive and tolerance between 0 and 1.")
    if args.threads is not None and args.threads < 1:
        raise ValueError("Threads must be positive.")
    csv.field_size_limit(10_000_000)
    if args.output.exists() and any(args.output.iterdir()) and not args.resume:
        raise FileExistsError("Output folder is not empty. Use --resume or a new output folder.")
    args.output.mkdir(parents=True, exist_ok=True)
    if args.resume and not (args.output / "manifest.json").exists():
        raise ValueError("No manifest.json to resume. Use a new output folder without --resume.")
    device = resolve_device(args.device)
    if args.threads:
        torch.set_num_threads(args.threads)
    torch.manual_seed(42)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"Device: {device}; batch: {args.batch_size}; chunk rows: {args.chunk_rows}", flush=True)

    print("Hashing and counting the cleaned CSV...", flush=True)
    original_stat = args.csv.stat()
    source_hash = file_hash(args.csv)
    total = sum(1 for _ in input_rows(args.csv))
    if not total:
        raise ValueError("Dataset is empty.")
    print(f"Dataset: {total:,} records", flush=True)
    ensemble, hashes = load_models(args.models)
    dimensions, errors = preflight(ensemble, device, args.atol)
    identity = {
        "schema_version": SCHEMA_VERSION, "dataset_sha256": source_hash,
        "dataset_rows": total, "model_sha256": hashes,
        "encoding_sha256": {
            name: file_hash(ROOT / name) for name in ("ensemble.py", "graph_encoding.py")
        },
        "preprocessor_sha256": file_hash(Path(__file__).resolve()),
        "versions": {
            "torch": str(torch.__version__).split("+")[0],
            "torch_geometric": str(torch_geometric.__version__),
            "rdkit": rdBase.rdkitVersion, "numpy": np.__version__,
        },
        "dimensions": dimensions, "dtype": "float32",
    }
    run_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    manifest_path = args.output / "manifest.json"
    if args.resume:
        with manifest_path.open(encoding="utf-8") as stream:
            manifest = json.load(stream)
        if manifest.get("run_id") != run_id or manifest.get("identity") != identity:
            raise ValueError(
                "Dataset, checkpoints, encoding code, or package versions changed. "
                "Restore the original environment/files or use a new output folder."
            )
        print("Verifying committed chunks before resuming...", flush=True)
    else:
        atomic_json(manifest_path, {
            "run_id": run_id, "identity": identity, "created_utc": utc_now(),
            "input_filename": args.csv.name, "preflight_max_abs_error": errors,
            "row_alignment": "Row i in molecules.csv matches row i in both NPY files.",
        })
    infos, next_row = verify_chunks(args.output, run_id, dimensions)
    if next_row > total:
        raise ValueError("Committed chunks exceed the dataset row count.")
    summary = save_summary(args.output, run_id, infos, total)
    print(f"Starting from source row {next_row:,}/{total:,}", flush=True)

    started, initial_row = time.perf_counter(), next_row
    records = iter(itertools.islice(input_rows(args.csv), next_row, None))
    while True:
        chunk = list(itertools.islice(records, args.chunk_rows))
        if not chunk:
            break
        result = process_chunk(ensemble, chunk, device, dimensions, args.batch_size)
        current_stat = args.csv.stat()
        if (current_stat.st_size, current_stat.st_mtime_ns) != (
            original_stat.st_size, original_stat.st_mtime_ns
        ):
            raise RuntimeError("Input CSV changed during processing; current chunk was not saved.")
        info = commit_chunk(args.output, len(infos), run_id, chunk, result, args.batch_size)
        infos.append(info)
        summary = save_summary(args.output, run_id, infos, total)
        elapsed = time.perf_counter() - started
        rate = (summary["processed"] - initial_row) / max(elapsed, 1e-9)
        remaining = (total - summary["processed"]) / rate / 60
        print(
            f"Chunk {info['index']:06d} saved | {summary['processed']:,}/{total:,} "
            f"({100 * summary['processed'] / total:.1f}%) | "
            f"embedded {summary['embedded']:,}, failed {summary['failed']:,} | "
            f"{rate:.1f} rows/sec | ETA {remaining:.1f} min",
            flush=True,
        )
        del result, chunk
    if not summary["complete"]:
        raise RuntimeError("Input ended before the recorded row count. Check the source CSV.")
    print(f"\nCOMPLETE: {summary['embedded']:,} embedded; {summary['failed']:,} failed.", flush=True)
    print(f"Output: {args.output.resolve()}", flush=True)
    print("Failures are recorded in failures.csv inside each committed chunk.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=ROOT / "coconut.csv")
    parser.add_argument("--models", type=Path, default=ROOT / "models")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--chunk-rows", type=int, default=5000)
    parser.add_argument("--threads", type=int)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    try:
        run(args)
    except KeyboardInterrupt:
        parser.exit(130, "\nInterrupted. Committed chunks are safe; rerun with --resume.\n")


if __name__ == "__main__":
    main()
