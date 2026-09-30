"""Content-addressed cache keys and safetensors tensor cache (D-019, ARCHITECTURE §4).

Every cached artifact is named by `make_key(fields)`: the first 16 hex characters
of SHA-256 over canonical JSON of the fields that determine its content. A
different model revision, alpha, precision, layer, sentence-set hash or
`schema_version` therefore yields a new key, so stale data is never reused and
invalidation is a version bump rather than a delete (ROLLBACK.md).

`TensorCache` stores one `<key>.safetensors` + `<key>.json` sidecar per entry
under `<root>/<namespace>/`. The embed stage (M2) uses namespaces such as
`embeddings/<model_slug>/<regime>/<ckpt_slug>`.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path, PurePath
from typing import Any

import numpy as np

from rbbd.utils.logging import get_logger

log = get_logger(__name__)

KEY_LENGTH = 16
FLOAT_DECIMALS = 6


def canonical(obj: Any) -> Any:
    """Normalise `obj` into JSON-safe values with a single spelling per content.

    Floats are rounded to 6 decimals (so 0.1 + 0.2 and 0.3 hash equally), -0.0
    becomes 0.0, tuples become lists, paths become strings and NumPy scalars become
    Python scalars. Dict keys are stringified; key order is fixed later by
    `sort_keys` in `canonical_json`. NaN/Inf are rejected because they have no
    stable JSON form.
    """
    if isinstance(obj, Mapping):
        return {str(k): canonical(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [canonical(v) for v in obj]
    if isinstance(obj, np.generic):
        return canonical(obj.item())
    if isinstance(obj, bool) or obj is None or isinstance(obj, (int, str)):
        return obj
    if isinstance(obj, float):
        if not math.isfinite(obj):
            raise ValueError(f"non-finite float {obj!r} cannot be part of a cache key")
        rounded = round(obj, FLOAT_DECIMALS)
        return 0.0 if rounded == 0 else rounded
    if isinstance(obj, PurePath):
        return str(obj)
    raise TypeError(f"cannot canonicalise {type(obj).__name__} for a cache key")


def canonical_json(obj: Any) -> str:
    """Compact, key-sorted JSON of `canonical(obj)`."""
    return json.dumps(canonical(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def make_key(fields: Mapping[str, Any]) -> str:
    """16-hex content key for `fields` (ARCHITECTURE §4)."""
    return hashlib.sha256(canonical_json(fields).encode("utf-8")).hexdigest()[:KEY_LENGTH]


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """Full hex SHA-256 of a file's bytes."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def _to_numpy(value: Any) -> np.ndarray:
    # torch tensors (M2 onwards) are detached and moved to CPU; the cache itself
    # stays torch-free so CPU-only metric code can read it.
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.ascontiguousarray(np.asarray(value))


def _check_namespace(namespace: str) -> PurePath:
    path = PurePath(namespace)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"cache namespace must be a relative path without '..': {namespace!r}")
    return path


class TensorCache:
    """Safetensors cache rooted at the artifact store root (`$RBBD_ARTIFACTS`).

    Entries are immutable: `write` refuses to overwrite an existing key, because a
    key collision means identical content by construction.
    """

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def path_for(self, namespace: str, key: str) -> Path:
        """Path of the tensor file for (namespace, key); the sidecar shares the stem."""
        return self.root / _check_namespace(namespace) / f"{key}.safetensors"

    def lookup(self, namespace: str, key: str) -> Path | None:
        """Return the tensor path if a complete entry exists, logging `cache hit`."""
        path = self.path_for(namespace, key)
        sidecar = path.with_suffix(".json")
        if path.exists() and sidecar.exists():
            meta = json.loads(sidecar.read_text())
            if meta.get("key") == key:
                log.info("cache hit: %s/%s", namespace, key)
                return path
        log.info("cache miss: %s/%s", namespace, key)
        return None

    def write(
        self,
        namespace: str,
        key: str,
        tensors: Mapping[str, Any],
        sidecar: Mapping[str, Any],
    ) -> Path:
        """Atomically write `tensors` (name -> array [..]) plus a JSON sidecar.

        `sidecar` should carry the key fields and row metadata (for embeddings:
        `sentence_hashes` in row order). Arrays keep their dtype (fp16 for
        embeddings, D-019).
        """
        from safetensors.numpy import save_file

        path = self.path_for(namespace, key)
        if path.exists():
            raise FileExistsError(f"cache entry already exists: {namespace}/{key}")
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".safetensors.tmp")
        save_file({name: _to_numpy(t) for name, t in tensors.items()}, str(tmp))
        meta = {
            **canonical(dict(sidecar)),
            "key": key,
            "namespace": namespace,
            "created": datetime.now(timezone.utc).isoformat(),
            "sha256": sha256_file(tmp),
        }
        side_tmp = path.with_suffix(".json.tmp")
        side_tmp.write_text(json.dumps(meta, indent=2, sort_keys=True))
        # Tensor file first, sidecar last: `lookup` treats a missing sidecar as a miss,
        # so a crash between the two renames never yields a half-written hit.
        os.replace(tmp, path)
        os.replace(side_tmp, path.with_suffix(".json"))
        return path

    def read(self, namespace: str, key: str) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """Return (tensors, sidecar) for an existing entry; raises FileNotFoundError."""
        from safetensors.numpy import load_file

        path = self.path_for(namespace, key)
        meta = json.loads(path.with_suffix(".json").read_text())
        return load_file(str(path)), meta
