"""Load a tokenizer and model at an explicit precision and placement (D-004, D-005, D-018).

Every embedding extraction for one model goes through `load_for_inference` with the
same `LoadSpec`, so the reference and every audited checkpoint share precision,
quantisation and placement (D-004); `LoadSpec.key_fields()` puts those facts into the
embedding cache key (D-019).

Tokenizers are right-padded with pad := eos when the model has no pad token (D-018),
which is what `embed.pooling` assumes.

`base_decoder` returns the inner transformer (the module holding `embed_tokens`,
`layers` and the final `norm`). Running it skips the LM head (E2), and its
`last_hidden_state` is the final layer after the final norm (D-017). For Gemma 3 the
`-it` checkpoints are multimodal; the text decoder is found the same way.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

DTYPES = ("fp16", "fp32")
PLACEMENTS = ("balanced", "cuda:0", "cuda:1", "cpu")


@dataclass(frozen=True)
class LoadSpec:
    """How to load one model for inference.

    precision: "fp16" | "fp32" (no bf16 on T4, D-002); placement: "balanced" (shard over
    all visible GPUs, 7-8B and Gemma fp32), "cuda:0"/"cuda:1" (one GPU), or "cpu" (tests).
    """

    model_id: str
    revision: str | None = None
    precision: str = "fp16"
    placement: str = "balanced"

    def __post_init__(self) -> None:
        if self.precision not in DTYPES:
            raise ValueError(f"precision must be one of {DTYPES}, got {self.precision!r}")
        if self.placement not in PLACEMENTS:
            raise ValueError(f"placement must be one of {PLACEMENTS}, got {self.placement!r}")

    def key_fields(self) -> dict[str, Any]:
        """Fields that must enter every cache key derived from this load (D-004, D-019)."""
        return {**asdict(self), "quant": None}


def torch_dtype(precision: str) -> Any:
    """torch dtype for a precision name."""
    import torch

    return {"fp16": torch.float16, "fp32": torch.float32}[precision]


def prepare_tokenizer(tok: Any) -> Any:
    """Right padding, pad := eos when missing (Llama, Mistral have no pad token; D-018)."""
    tok.padding_side = "right"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_tokenizer(model_id: str, revision: str | None = None) -> Any:
    """AutoTokenizer prepared for mask-aware pooling."""
    from transformers import AutoTokenizer

    return prepare_tokenizer(AutoTokenizer.from_pretrained(model_id, revision=revision))


def download(spec: LoadSpec) -> str:
    """Fetch the checkpoint's weights, config and tokenizer files into $HF_HOME; return the
    local snapshot path. Timed separately so DC-12 can exclude download time."""
    from huggingface_hub import snapshot_download

    return snapshot_download(
        spec.model_id,
        revision=spec.revision,
        allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "*.tiktoken"],
    )


def load_for_inference(spec: LoadSpec) -> tuple[Any, Any]:
    """(model in eval mode, tokenizer) for `spec`. Weights are downloaded to $HF_HOME."""
    from transformers import AutoModelForCausalLM

    kwargs: dict[str, Any] = {"dtype": torch_dtype(spec.precision), "revision": spec.revision}
    if spec.placement == "balanced":
        kwargs["device_map"] = "balanced"
    elif spec.placement.startswith("cuda"):
        kwargs["device_map"] = {"": spec.placement}
    model = AutoModelForCausalLM.from_pretrained(spec.model_id, **kwargs)
    model.eval()
    return model, load_tokenizer(spec.model_id, spec.revision)


def base_decoder(model: Any) -> Any:
    """The innermost module with `embed_tokens`, `layers` and `norm` (the text transformer).

    LlamaForCausalLM / MistralForCausalLM / Qwen2ForCausalLM -> `model.model`;
    Gemma3ForConditionalGeneration -> its text `language_model`.
    """
    for module in model.modules():
        if all(hasattr(module, a) for a in ("embed_tokens", "layers", "norm")):
            return module
    raise ValueError(f"no decoder with embed_tokens/layers/norm in {type(model).__name__}")


def num_layers(model: Any) -> int:
    """Number of decoder layers L; `last_hidden_state` is hidden_states[L] (D-017)."""
    return len(base_decoder(model).layers)


def input_device(model: Any) -> Any:
    """Device of the token embedding (where input ids must be placed under sharding)."""
    return base_decoder(model).embed_tokens.weight.device
