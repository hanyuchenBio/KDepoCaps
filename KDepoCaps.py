#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Minimal KPKTP protein prediction pipeline.

Workflow:
    protein FASTA -> ESM-C 300M embedding -> per-K Random Forest score -> output table

Required command-line inputs:
    -i    Protein FASTA file or directory
    -esm  ESM-C 300M .pth weight file
    -pm   Trained RF model bundle from train_final_rf_protein_minimal.py
    -o    Output prediction table

Output:
    First column: ID
    Remaining columns: one raw RF positive-class probability for each KL type.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import torch


FASTA_EXTENSIONS = {".fa", ".fasta", ".faa", ".fas", ".fsa"}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Predict Klebsiella KL types from protein FASTA using ESM-C 300M "
            "and RF models trained by train_final_rf_protein_minimal.py."
        )
    )
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="Protein FASTA file or directory containing protein FASTA files.",
    )
    parser.add_argument(
        "-esm",
        "--esm-model",
        required=True,
        help="Path to the ESM-C 300M .pth weight file.",
    )
    parser.add_argument(
        "-pm",
        "--prediction-model",
        required=True,
        help="Path to the trained RF model bundle produced by train_final_rf_protein_minimal.py.",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Output prediction table, e.g. prediction.csv.",
    )
    return parser.parse_args()


def find_fasta_files(input_path: Path) -> List[Path]:
    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    if input_path.is_file():
        if input_path.suffix.lower() not in FASTA_EXTENSIONS:
            raise ValueError(f"Unsupported FASTA extension: {input_path}")
        return [input_path]

    files = [
        p for p in sorted(input_path.rglob("*"))
        if p.is_file() and p.suffix.lower() in FASTA_EXTENSIONS
    ]
    if not files:
        raise FileNotFoundError(f"No protein FASTA files found in: {input_path}")
    return files


def read_fasta(path: Path) -> List[Tuple[str, str]]:
    records: List[Tuple[str, str]] = []
    seq_id = None
    seq_parts: List[str] = []

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if seq_id is not None:
                    records.append((seq_id, "".join(seq_parts)))
                seq_id = line[1:].split()[0]
                seq_parts = []
            else:
                seq_parts.append(line.upper())

    if seq_id is not None:
        records.append((seq_id, "".join(seq_parts)))

    return records


def clean_protein_sequence(seq: str) -> str:
    valid_aas = set("ACDEFGHIKLMNPQRSTVWYBXZUO")
    return "".join(aa if aa in valid_aas else "X" for aa in seq.upper())


def load_protein_records(input_path: Path) -> List[Tuple[str, str]]:
    records: List[Tuple[str, str]] = []
    seen: Dict[str, int] = {}

    for fasta_file in find_fasta_files(input_path):
        file_records = read_fasta(fasta_file)
        if not file_records:
            print(f"[WARN] No FASTA records found: {fasta_file}")
            continue

        for raw_id, raw_seq in file_records:
            seq = clean_protein_sequence(raw_seq)
            if not seq:
                print(f"[WARN] Empty protein sequence skipped: {fasta_file} | {raw_id}")
                continue

            seq_id = raw_id
            if seq_id in seen:
                seen[seq_id] += 1
                seq_id = f"{seq_id}__dup{seen[raw_id]}"
            else:
                seen[seq_id] = 1

            records.append((seq_id, seq))

    if not records:
        raise ValueError("No valid protein sequences were loaded.")

    print(f"[INFO] Protein sequences loaded: {len(records)}")
    return records


def load_esmc(weight_file: Path, device: torch.device):
    try:
        from esm.models.esmc import ESMC
        from esm.tokenization import get_esmc_model_tokenizers
    except Exception as exc:
        raise ImportError(
            "The EvolutionaryScale ESM package with esm.models.esmc is required."
        ) from exc

    print(f"[INFO] ESM-C weight: {weight_file}")
    print(f"[INFO] Device: {device}")

    model = ESMC(
        d_model=960,
        n_heads=15,
        n_layers=30,
        tokenizer=get_esmc_model_tokenizers(),
        use_flash_attn=False,
    ).eval()

    try:
        state_dict = torch.load(str(weight_file), map_location="cpu", weights_only=False)
    except TypeError:
        state_dict = torch.load(str(weight_file), map_location="cpu")

    model.load_state_dict(state_dict)

    if device.type == "cuda":
        model = model.to(device=device, dtype=torch.bfloat16)
    else:
        model = model.to(device=device)

    model.eval()
    return model


def embed_one_protein(model, sequence: str) -> np.ndarray:
    from esm.sdk.api import ESMProtein, LogitsConfig

    protein = ESMProtein(sequence=sequence)
    protein_tensor = model.encode(protein)

    with torch.no_grad():
        output = model.logits(
            protein_tensor,
            LogitsConfig(sequence=True, return_embeddings=True),
        )

    embeddings = output.embeddings
    if embeddings is None:
        raise RuntimeError("ESM-C returned no embeddings.")

    embeddings = embeddings.detach().float().cpu()
    if embeddings.ndim == 3:
        embeddings = embeddings.squeeze(0)
    if embeddings.ndim != 2:
        raise RuntimeError(
            f"Unexpected ESM-C embedding shape: {tuple(embeddings.shape)}"
        )

    seq_len = len(sequence)
    if embeddings.shape[0] == seq_len + 2:
        embeddings = embeddings[1:-1]
    elif embeddings.shape[0] == seq_len + 1:
        embeddings = embeddings[1:]
    elif embeddings.shape[0] > seq_len:
        embeddings = embeddings[:seq_len]

    if embeddings.shape[0] == 0:
        raise RuntimeError("Empty residue embedding after removing special tokens.")

    return embeddings.mean(dim=0).numpy().astype(float)


def embed_proteins(records: List[Tuple[str, str]], weight_file: Path) -> Tuple[List[str], np.ndarray]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_esmc(weight_file, device)

    ids: List[str] = []
    vectors: List[np.ndarray] = []

    for i, (seq_id, seq) in enumerate(records, start=1):
        vector = embed_one_protein(model, seq)
        ids.append(seq_id)
        vectors.append(vector)
        print(f"[INFO] Embedded: {i}/{len(records)} | {seq_id}")

        if device.type == "cuda":
            torch.cuda.empty_cache()

    matrix = np.vstack(vectors)
    print(f"[INFO] Protein embedding shape: {matrix.shape}")
    return ids, matrix


def natural_k_sort_key(k_name: str) -> Tuple[int, str]:
    match = re.fullmatch(r"K(\d+)", k_name, flags=re.IGNORECASE)
    if match:
        return int(match.group(1)), k_name
    return 10**9, k_name


def load_rf_models(model_file: Path) -> List[Tuple[str, object, List[str]]]:
    bundle = joblib.load(model_file)
    if not isinstance(bundle, dict) or not bundle:
        raise ValueError(f"Invalid or empty RF model bundle: {model_file}")

    loaded: List[Tuple[str, object, List[str]]] = []

    for model_key, payload in bundle.items():
        if not isinstance(payload, dict) or "model" not in payload:
            raise ValueError(f"Invalid model entry: {model_key}")

        match = re.fullmatch(r"(K\d+)_protein", str(model_key), flags=re.IGNORECASE)
        if not match:
            raise ValueError(
                f"Unexpected model key '{model_key}'. Expected format like K1_protein."
            )

        k_name = match.group(1).upper()
        feature_cols = [str(x) for x in payload.get("feature_cols", [])]
        loaded.append((k_name, payload["model"], feature_cols))

    loaded.sort(key=lambda x: natural_k_sort_key(x[0]))
    print(f"[INFO] RF models loaded: {len(loaded)}")
    return loaded


def positive_probability(model, x: pd.DataFrame) -> np.ndarray:
    if not hasattr(model, "predict_proba"):
        raise TypeError("RF model does not provide predict_proba().")

    proba = model.predict_proba(x)
    classes = list(model.classes_)
    if 1 not in classes:
        raise ValueError(f"RF model classes do not contain positive class 1: {classes}")

    return proba[:, classes.index(1)].astype(float)


def resolve_model_feature_names(model, feature_cols: List[str], k_name: str) -> List[str]:
    """Return the exact feature names used when the RF model was fitted."""
    model_feature_names = getattr(model, "feature_names_in_", None)
    if model_feature_names is not None:
        names = [str(x) for x in model_feature_names]
    elif feature_cols:
        names = [str(x) for x in feature_cols]
    else:
        expected_n = getattr(model, "n_features_in_", None)
        if expected_n is None:
            raise ValueError(
                f"{k_name}: cannot determine RF feature names or feature count."
            )
        names = [f"dim_{i}" for i in range(int(expected_n))]

    if len(names) != len(set(names)):
        raise ValueError(f"{k_name}: duplicated feature names were found in the RF model.")
    return names


def predict_all_ktypes(
    ids: List[str],
    embedding_matrix: np.ndarray,
    models: List[Tuple[str, object, List[str]]],
) -> pd.DataFrame:
    output: Dict[str, object] = {"ID": ids}
    dataframe_cache: Dict[Tuple[str, ...], pd.DataFrame] = {}

    for k_name, model, feature_cols in models:
        feature_names = resolve_model_feature_names(model, feature_cols, k_name)
        expected_n = len(feature_names)

        if embedding_matrix.shape[1] != expected_n:
            raise ValueError(
                f"{k_name}: model expects {expected_n} features, "
                f"but ESM-C generated {embedding_matrix.shape[1]}."
            )

        # The RF models were fitted with a pandas DataFrame, so prediction must
        # also use a DataFrame carrying the same feature names. This avoids
        # sklearn's 'X does not have valid feature names' warning and provides
        # an explicit feature-order contract.
        cache_key = tuple(feature_names)
        if cache_key not in dataframe_cache:
            dataframe_cache[cache_key] = pd.DataFrame(
                embedding_matrix,
                columns=feature_names,
            )
        x = dataframe_cache[cache_key]

        kl_name = re.sub(r"^K", "KL", k_name, flags=re.IGNORECASE)
        output[kl_name] = np.round(
            positive_probability(model, x),
            6,
        )

    return pd.DataFrame(output)


def write_output(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()

    if suffix == ".tsv":
        df.to_csv(output_path, sep="\t", index=False)
    elif suffix in {".xlsx", ".xls"}:
        df.to_excel(output_path, index=False)
    else:
        df.to_csv(output_path, index=False)


def main() -> None:
    args = parse_args()

    weight_file = Path(args.esm_model).expanduser().resolve()
    model_file = Path(args.prediction_model).expanduser().resolve()

    if not weight_file.is_file():
        raise FileNotFoundError(f"ESM-C weight file not found: {weight_file}")
    if not model_file.is_file():
        raise FileNotFoundError(f"RF model bundle not found: {model_file}")

    records = load_protein_records(Path(args.input))
    ids, embedding_matrix = embed_proteins(records, weight_file)
    models = load_rf_models(model_file)
    result = predict_all_ktypes(ids, embedding_matrix, models)

    output_path = Path(args.output)
    write_output(result, output_path)

    print(f"[DONE] Predicted proteins: {len(result)}")
    print(f"[DONE] KL-type models: {len(models)}")
    print(f"[DONE] Output: {output_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        raise SystemExit(f"[ERROR] {exc}")
