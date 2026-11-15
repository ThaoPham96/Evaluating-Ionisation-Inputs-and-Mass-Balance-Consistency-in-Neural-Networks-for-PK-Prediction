"""Build the frozen v6 graph and ChemBERTa features from reviewed structures."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import tempfile
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem, rdBase
from torch_geometric.data import Data

from pipknet import featurizers


ROOT = Path(__file__).resolve().parent
DATA_FINGERPRINT = "5413771691c6981d22b6c9ca8e21584dc93cd01244c5e3b143c489dfdc9c7942"
DATA_DIR = ROOT / "data" / "mass_balance" / DATA_FINGERPRINT[:16]
OUTPUT_DIR = ROOT / "outputs" / "mass_balance_features" / DATA_FINGERPRINT[:16]
MODEL_ID = "seyonec/PubChem10M_SMILES_BPE_450k"
MODEL_REVISION = "c18fccd09b3326bf2d4633412c256d7db872156d"
MODEL_API = f"https://huggingface.co/api/models/{MODEL_ID}/revision/{MODEL_REVISION}?blobs=true"
# Git blob IDs and LFS SHA-256 published by the Hugging Face model API.
MODEL_TEXT_BLOBS = {
    "config.json": "945d3f4694927f2144868e0f4f7e9254e2441307",
    "merges.txt": "763dc46f0758806cb81d0c25baca3b5ac21fccb7",
    "special_tokens_map.json": "e97d1993365bb21c88f390e8703e4c1af564821f",
    "tokenizer_config.json": "18d63e03c656615691d7e3b5e75c217d4088c357",
    "vocab.json": "06a706075a2bb0a243ba8e562722b775c76b1b3e",
}
MODEL_WEIGHTS = "pytorch_model.bin"
WEIGHTS_SHA256 = "f471de1ad73de11d47e1bbdfddb93aeace0b8aeab31ec03bcde0eabb17a97264"
N_STRUCTURES = 952
EMBEDDING_DIM = 768
NODE_DIM = 23
EDGE_DIM = 6


def sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def write_json(path: Path, data) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(data, stream, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def load_structures(data_dir: Path) -> tuple[list[str], list[str], dict]:
    manifest_path = data_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["fingerprint"] != DATA_FINGERPRINT or not manifest["data_frozen"]:
        raise ValueError("Expected the reviewed v6 data release")
    dataset = data_dir / "model_dataset.csv"
    dataset_sha256 = sha256(dataset)
    if dataset_sha256 != manifest["files"][dataset.name]:
        raise ValueError(f"Checksum mismatch for {dataset}")
    with dataset.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    structure_ids = [row["structure_id"] for row in rows]
    smiles = [row["canonical_smiles"] for row in rows]
    if len(structure_ids) != N_STRUCTURES or len(set(structure_ids)) != N_STRUCTURES:
        raise ValueError(f"Expected {N_STRUCTURES} unique frozen structures")
    for structure_id, smi in zip(structure_ids, smiles, strict=True):
        if not smi or hashlib.sha256(smi.encode()).hexdigest()[:16] != structure_id:
            raise ValueError(f"SMILES does not match frozen structure ID {structure_id}")
    return structure_ids, smiles, {
        "fingerprint": manifest["fingerprint"],
        "manifest.json": sha256(manifest_path),
        "model_dataset.csv": dataset_sha256,
        "ordered_structure_smiles_sha256": hashlib.sha256(
            json.dumps(list(zip(structure_ids, smiles, strict=True)),
                       ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
    }


def molecular_graph(structure_id: str, smiles: str) -> Data:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or not mol.GetNumAtoms():
        raise ValueError(f"RDKit could not parse frozen structure {structure_id}")
    x = torch.stack([featurizers.atom_to_feature_vector(atom) for atom in mol.GetAtoms()])
    if mol.GetNumBonds():
        edge_index, edge_attr = featurizers.build_edge_index_and_attr(mol)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, EDGE_DIM), dtype=torch.float32)
    n_edges = 2 * mol.GetNumBonds()
    if (
        x.shape != (mol.GetNumAtoms(), NODE_DIM)
        or edge_index.shape != (2, n_edges)
        or edge_attr.shape != (n_edges, EDGE_DIM)
        or x.dtype != torch.float32
        or edge_attr.dtype != torch.float32
        or edge_index.dtype != torch.long
        or not torch.isfinite(x).all()
        or not torch.isfinite(edge_attr).all()
        or (n_edges and (edge_index.min() < 0 or edge_index.max() >= x.size(0)))
    ):
        raise ValueError(f"Invalid graph tensor for {structure_id}")
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def prepare_model_files(offline: bool) -> tuple[Path, dict]:
    from huggingface_hub import snapshot_download

    snapshot = Path(snapshot_download(
        MODEL_ID,
        revision=MODEL_REVISION,
        endpoint="https://huggingface.co",
        allow_patterns=[*MODEL_TEXT_BLOBS, MODEL_WEIGHTS],
        local_files_only=offline,
        max_workers=1,
        token=False,
    ))
    if snapshot.name != MODEL_REVISION:
        raise ValueError(f"Model snapshot must be pinned to {MODEL_REVISION}")
    file_hashes = {}
    for name, expected_blob in MODEL_TEXT_BLOBS.items():
        data = (snapshot / name).read_bytes()
        git_blob = hashlib.sha1(f"blob {len(data)}\0".encode("ascii") + data).hexdigest()
        if git_blob != expected_blob:
            raise ValueError(f"Hugging Face Git blob mismatch for {name}")
        file_hashes[name] = {"sha256": hashlib.sha256(data).hexdigest(), "git_blob": git_blob}
    if sha256(snapshot / MODEL_WEIGHTS) != WEIGHTS_SHA256:
        raise ValueError(f"Hugging Face LFS SHA-256 mismatch for {MODEL_WEIGHTS}")
    file_hashes[MODEL_WEIGHTS] = {"sha256": WEIGHTS_SHA256}
    return snapshot, file_hashes


def tokenize_structures(tokenizer, structure_ids: list[str], smiles: list[str], limit: int):
    encoded = tokenizer(
        smiles, add_special_tokens=True, truncation=False, padding=False,
        return_attention_mask=True, return_special_tokens_mask=True,
    )
    lengths = [len(ids) for ids in encoded["input_ids"]]
    overflow = [
        (sid, length) for sid, length in zip(structure_ids, lengths, strict=True)
        if length > limit
    ]
    if overflow:
        raise ValueError(f"Token limit {limit} exceeded by {overflow}. No truncation applied.")
    reserved_ids = set(tokenizer.all_special_ids)
    for sid, ids, special_mask in zip(
        structure_ids, encoded["input_ids"], encoded["special_tokens_mask"], strict=True
    ):
        content_ids = [token for token, special in zip(ids, special_mask, strict=True) if not special]
        if not content_ids or reserved_ids.intersection(content_ids):
            raise ValueError(f"No usable SMILES tokens or reserved token in content for {sid}")
    return encoded, lengths


def pool_final_hidden(
    hidden: torch.Tensor, attention_mask: torch.Tensor, special_mask: torch.Tensor
) -> torch.Tensor:
    if hidden.ndim != 3 or attention_mask.shape != hidden.shape[:2] or special_mask.shape != hidden.shape[:2]:
        raise ValueError("Hidden state and token mask shapes do not agree")
    include = attention_mask.bool() & ~special_mask.bool()
    counts = include.sum(dim=1, keepdim=True)
    if (counts == 0).any():
        raise ValueError("Cannot pool a sequence with no content tokens")
    return hidden.masked_fill(~include.unsqueeze(-1), 0).sum(dim=1) / counts


def embed_structures(snapshot, structure_ids, smiles, device, batch_size):
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(snapshot, use_fast=True, local_files_only=True)
    config = AutoConfig.from_pretrained(snapshot, local_files_only=True)
    if (
        config.model_type != "roberta"
        or config.hidden_size != EMBEDDING_DIM
        or tokenizer.pad_token_id != config.pad_token_id
    ):
        raise ValueError("Expected the pinned 768-dimensional RoBERTa encoder and tokenizer")
    # RoBERTa's first non-padding position is pad_token_id + 1.
    position_limit = config.max_position_embeddings - config.pad_token_id - 1
    token_limit = min(tokenizer.model_max_length, position_limit)
    encoded, lengths = tokenize_structures(tokenizer, structure_ids, smiles, token_limit)
    print(f"Validated {len(lengths)} untruncated token sequences, {min(lengths)}-{max(lengths)} tokens "
          f"including special tokens, limit {token_limit}.", flush=True)

    model, load_info = AutoModel.from_pretrained(
        snapshot, local_files_only=True, use_safetensors=False, weights_only=True,
        add_pooling_layer=False, torch_dtype=torch.float32,
        attn_implementation="eager", output_loading_info=True,
    )
    # Only the pretrained transformer layers contribute to final token states.
    unused_pooler_keys = {"roberta.pooler.dense.bias", "roberta.pooler.dense.weight"}
    unexpected_encoder_keys = [
        key for key in load_info["unexpected_keys"] if not key.startswith("lm_head.")
        and key not in unused_pooler_keys
    ]
    if load_info["missing_keys"] or load_info["mismatched_keys"] or load_info["error_msgs"] or unexpected_encoder_keys:
        raise ValueError(f"Pretrained encoder load mismatch: {load_info}")
    model = model.to(device).eval().requires_grad_(False)
    embeddings = np.empty((len(structure_ids), EMBEDDING_DIM), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(structure_ids), batch_size):
            stop = min(start + batch_size, len(structure_ids))
            batch = tokenizer.pad(
                {key: value[start:stop] for key, value in encoded.items()},
                padding=True, return_tensors="pt",
            ).to(device)
            special_mask = batch.pop("special_tokens_mask")
            hidden = model(**batch).last_hidden_state
            pooled = pool_final_hidden(hidden, batch["attention_mask"], special_mask)
            if pooled.shape != (stop - start, EMBEDDING_DIM) or pooled.dtype != torch.float32:
                raise ValueError("Unexpected pooled embedding shape or dtype")
            embeddings[start:stop] = pooled.cpu().numpy()
    if not np.isfinite(embeddings).all():
        bad_ids = np.asarray(structure_ids)[~np.isfinite(embeddings).all(axis=1)]
        raise ValueError(f"Nonfinite ChemBERTa embeddings: {bad_ids.tolist()}")
    return embeddings, {
        "tokenizer_class": type(tokenizer).__name__,
        "tokenizer_use_fast": tokenizer.is_fast,
        "padding_side": tokenizer.padding_side,
        "special_token_ids": tokenizer.all_special_ids,
        "position_limit_including_special_tokens": token_limit,
        "observed_token_length_range_including_special_tokens": [min(lengths), max(lengths)],
        "longest_structure_ids": [
            sid for sid, length in zip(structure_ids, lengths, strict=True) if length == max(lengths)
        ],
        "truncation": False,
        "pooling": "float32 mean of final encoder hidden states at attended non-special token positions",
        "attention_implementation": "eager",
        "unused_checkpoint_keys": load_info["unexpected_keys"],
        "shape": list(embeddings.shape),
        "dtype": str(embeddings.dtype),
        "batch_size": batch_size,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--offline", action="store_true", help="Use the pinned Hugging Face snapshot cache")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.output.exists():
        parser.error(f"Frozen feature output already exists: {args.output}")

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable. Use --device cpu to run on the CPU.")
    if rdBase.rdkitVersion != "2025.09.6":
        parser.error("RDKit 2025.09.6 is required to match the frozen structure release.")

    structure_ids, smiles, data_source = load_structures(args.data_dir)
    graphs = [molecular_graph(sid, smi) for sid, smi in zip(structure_ids, smiles, strict=True)]
    snapshot, model_file_hashes = prepare_model_files(args.offline)
    embeddings, embedding_recipe = embed_structures(snapshot, structure_ids, smiles, device, args.batch_size)

    manifest = {
        "schema_version": 1,
        "structure_count": len(structure_ids),
        "order": "model_dataset.csv row order for structures.json, graphs.pt and chemberta.npy",
        "data": data_source,
        "source_hashes": {
            Path(__file__).name: sha256(Path(__file__)),
            "pipknet/featurizers.py": sha256(Path(featurizers.__file__)),
            "pipknet/models.py": sha256(ROOT / "pipknet" / "models.py"),
        },
        "graphs": {
            "representation": "list of PyTorch Geometric Data(x, edge_index, edge_attr)",
            "smiles": "frozen canonical_smiles, parsed directly without component filtering or normalization",
            "rdkit_parser": "Chem.MolFromSmiles(smiles) defaults: sanitize=True, removeHs=True",
            "atom_features": "pipknet.featurizers.atom_to_feature_vector (element, degree, formal charge, aromaticity)",
            "edge_features": "pipknet.featurizers.build_edge_index_and_attr (bond type, conjugation, ring; both directions)",
            "node_feature_dim": NODE_DIM,
            "edge_feature_dim": EDGE_DIM,
            "empty_edges": "int64 edge_index[2,0] and float32 edge_attr[0,6]",
            "bondless_structure_count": sum(graph.num_edges == 0 for graph in graphs),
            "explicit_ion_category_input": False,
            "dedicated_isotope_or_stereochemistry_channels": False,
        },
        "chemberta": {
            "model_id": MODEL_ID,
            "revision": MODEL_REVISION,
            "revision_metadata_url": MODEL_API,
            "source_files": model_file_hashes,
            "model_config": json.loads((snapshot / "config.json").read_text(encoding="utf-8")),
            "tokenizer_config": json.loads((snapshot / "tokenizer_config.json").read_text(encoding="utf-8")),
            "special_tokens_map": json.loads((snapshot / "special_tokens_map.json").read_text(encoding="utf-8")),
            **embedding_recipe,
        },
        "versions": {
            "python": platform.python_version(),
            "rdkit": rdBase.rdkitVersion,
            "torch_cuda": torch.version.cuda,
            **{name: version(name) for name in (
                "numpy", "torch", "torch_geometric", "transformers",
                "huggingface_hub", "safetensors", "tokenizers", "regex",
            )},
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".features-", dir=args.output.parent) as temporary:
        stage = Path(temporary) / "bundle"
        stage.mkdir()
        write_json(stage / "structures.json", structure_ids)
        torch.save(graphs, stage / "graphs.pt")
        np.save(stage / "chemberta.npy", embeddings, allow_pickle=False)
        manifest["files"] = {path.name: sha256(path) for path in stage.iterdir()}
        write_json(stage / "manifest.json", manifest)
        stage.rename(args.output)
    print(f"Wrote immutable features for {len(structure_ids)} structures to {args.output}", flush=True)


if __name__ == "__main__":
    main()
