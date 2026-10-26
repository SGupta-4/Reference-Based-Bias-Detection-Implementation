"""In-memory full-weight interpolation for Tier 2 full fine-tuning (D-008).

    W(α) = (1−α)·W_h + α·W_u        computed in fp32 on CPU, then cast into the model

This form (not W_h + α(W_u − W_h)) returns the endpoints exactly: at α=1 it is
0·W_h + 1·W_u = W_u, and at α=0 it is W_h (DC-07). Endpoints are stored once,
privately (`ckpt_private/…`, D-041); merged α weights are never stored persistently.
`materialize` writes one α to ephemeral disk for vLLM (D-050) and refuses any path
under the persistent artifact root or /kaggle/working.

Callers: M4's `embed` stage (`apply_alpha` on one loaded model, swept over α, endpoints
held in RAM by `load_endpoint`) and M5's generation stage (`materialize` with
`LazyEndpoint`s, one tensor at a time; B-033).
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

PERSISTENT_ROOTS = (Path("/kaggle/working"),)


class EndpointMismatch(ValueError):
    """The two endpoints (or an endpoint and the model) disagree on keys or shapes."""


def load_endpoint(path: str | Path) -> dict[str, Any]:
    """{name: tensor fp32 on CPU} from a directory of safetensors shards (or one file)."""
    from safetensors.torch import load_file

    path = Path(path)
    files = sorted(path.glob("*.safetensors")) if path.is_dir() else [path]
    if not files:
        raise FileNotFoundError(f"no safetensors under {path}")
    state: dict[str, Any] = {}
    for f in files:
        state.update({k: v.float() for k, v in load_file(str(f)).items()})
    return state


class LazyEndpoint(Mapping):
    """Read-only {name: tensor fp32 on CPU} that reads one tensor at a time from the
    safetensors shards of an endpoint, so a merge holds one fp32 tensor per endpoint
    instead of two full fp32 copies (D-008 Revisit-if; B-033: two parallel Llama-1B merges
    were OOM-killed). Same keys and values as `load_endpoint(path)`."""

    def __init__(self, path: str | Path) -> None:
        from safetensors import safe_open

        path = Path(path)
        files = sorted(path.glob("*.safetensors")) if path.is_dir() else [path]
        if not files:
            raise FileNotFoundError(f"no safetensors under {path}")
        self._handles = [safe_open(str(f), framework="pt", device="cpu") for f in files]
        self._where = {k: h for h in self._handles for k in h.keys()}

    def __getitem__(self, name: str) -> Any:
        return self._where[name].get_tensor(name).float()

    def __iter__(self):
        return iter(self._where)

    def __len__(self) -> int:
        return len(self._where)

    def __contains__(self, name: object) -> bool:
        return name in self._where

    def shape(self, name: str) -> tuple[int, ...]:
        return tuple(self._where[name].get_slice(name).get_shape())


def _shape(endpoint: Mapping[str, Any], name: str) -> tuple[int, ...]:
    if isinstance(endpoint, LazyEndpoint):
        return endpoint.shape(name)
    return tuple(endpoint[name].shape)


def check_endpoints(w_h: Mapping[str, Any], w_u: Mapping[str, Any]) -> None:
    """Raise `EndpointMismatch` unless both endpoints have the same keys and shapes."""
    if set(w_h) != set(w_u):
        diff = sorted(set(w_h) ^ set(w_u))[:5]
        raise EndpointMismatch(f"endpoint keys differ, e.g. {diff}")
    bad = [k for k in w_h if _shape(w_h, k) != _shape(w_u, k)]
    if bad:
        raise EndpointMismatch(f"endpoint shapes differ for {bad[:5]}")


def interpolate(w_h: Any, w_u: Any, alpha: float) -> Any:
    """(1−α)·w_h + α·w_u in fp32; any shape. Exact endpoints at α ∈ {0, 1}."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    return (1.0 - alpha) * w_h.float() + alpha * w_u.float()


def apply_alpha(model: Any, w_h: Mapping[str, Any], w_u: Mapping[str, Any], alpha: float) -> None:
    """Overwrite `model`'s parameters and buffers in place with W(α), cast to their dtype.

    Every state-dict key must be in both endpoints, except a tied alias (e.g.
    `lm_head.weight` sharing storage with `model.embed_tokens.weight` in Llama-3.2-1B):
    `save_pretrained` stores tied tensors once, and writing the stored name updates the
    alias too. Runs under `torch.no_grad()`.
    """
    import torch

    state = model.state_dict()
    present = {k for k in state if k in w_h and k in w_u}
    stored_ptrs = {state[k].data_ptr() for k in present}
    missing = sorted(
        k for k in state if k not in present and state[k].data_ptr() not in stored_ptrs
    )
    if missing:
        raise EndpointMismatch(f"model keys missing from the endpoints, e.g. {missing[:5]}")
    with torch.no_grad():
        for name in sorted(present):
            target = state[name]
            if tuple(target.shape) != _shape(w_h, name):
                shapes = f"model {tuple(target.shape)} vs endpoint {_shape(w_h, name)}"
                raise EndpointMismatch(f"shape of {name}: {shapes}")
            target.copy_(interpolate(w_h[name], w_u[name], alpha).to(target.dtype))


def assert_ephemeral(path: str | Path, artifacts_root: str | Path) -> Path:
    """Return `path` resolved, or raise if it lies under persistent storage (D-008, D-050)."""
    resolved = Path(path).resolve()
    for root in (Path(artifacts_root), *PERSISTENT_ROOTS):
        if resolved.is_relative_to(Path(root).resolve()):
            raise ValueError(f"merged weights must not be written under {root}: {resolved}")
    return resolved


def materialize(
    model: Any,
    tokenizer: Any,
    w_h: Mapping[str, Any],
    w_u: Mapping[str, Any],
    alpha: float,
    out_dir: str | Path,
    artifacts_root: str | Path,
) -> Path:
    """Apply W(α) to `model` and save it (with the tokenizer) to an ephemeral `out_dir`.

    The caller deletes `out_dir` once vLLM has loaded it (D-050).
    """
    out = assert_ephemeral(out_dir, artifacts_root)
    apply_alpha(model, w_h, w_u, alpha)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out), safe_serialization=True)
    tokenizer.save_pretrained(str(out))
    return out
