"""Chunked storage with SHA-256 integrity — data-integrity test (H).

Split a file (here: the model file, or any payload) into fixed-size
chunks, hash every chunk, detect missing/corrupt chunks on verify,
and rebuild the original only when every chunk checks out.
"""
from __future__ import annotations

import hashlib
import json
import os

CHUNK_SIZE = 1 * 1024 * 1024  # 1 MiB


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def split_file(path: str, out_dir: str, chunk_size: int = CHUNK_SIZE) -> dict:
    """Split ``path`` into chunks under ``out_dir``; return the manifest."""
    os.makedirs(out_dir, exist_ok=True)
    chunks = []
    total = os.path.getsize(path)
    with open(path, "rb") as f:
        idx = 0
        while True:
            data = f.read(chunk_size)
            if not data:
                break
            name = f"chunk-{idx:05d}.bin"
            with open(os.path.join(out_dir, name), "wb") as cf:
                cf.write(data)
            chunks.append({"index": idx, "file": name,
                           "size": len(data),
                           "sha256": sha256_bytes(data)})
            idx += 1
    manifest = {"source": os.path.basename(path), "size": total,
                "chunk_size": chunk_size, "chunks": chunks,
                "source_sha256": sha256_file(path)}
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def verify_chunks(manifest: dict, chunk_dir: str) -> dict:
    """Return {ok, missing: [...], corrupt: [...]} — never silently pass."""
    missing, corrupt = [], []
    for c in manifest["chunks"]:
        p = os.path.join(chunk_dir, c["file"])
        if not os.path.exists(p):
            missing.append(c["index"])
            continue
        if sha256_file(p) != c["sha256"]:
            corrupt.append(c["index"])
    return {"ok": not missing and not corrupt,
            "missing": missing, "corrupt": corrupt,
            "checked": len(manifest["chunks"])}


def reconstruct(manifest: dict, chunk_dir: str, out_path: str) -> dict:
    """Rebuild only if verify() passes; returns {ok, sha256, ...}."""
    v = verify_chunks(manifest, chunk_dir)
    if not v["ok"]:
        return {"ok": False, **v}
    with open(out_path, "wb") as out:
        for c in manifest["chunks"]:
            with open(os.path.join(chunk_dir, c["file"]), "rb") as cf:
                out.write(cf.read())
    digest = sha256_file(out_path)
    return {"ok": digest == manifest["source_sha256"],
            "sha256": digest, "expected": manifest["source_sha256"], **v}
