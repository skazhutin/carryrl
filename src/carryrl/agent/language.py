"""Pinned, frozen multilingual instruction embeddings with bounded CPU caching."""

import json
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoConfig, AutoModel, AutoTokenizer

TEXT_ID = "intfloat/multilingual-e5-small"
TEXT_REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"
TEXT_WIDTH = 384
INSTRUCTION_MAX_CHARS = 8192


def validate_instruction(instruction: str | None) -> str | None:
    if instruction is None:
        return None
    if not isinstance(instruction, str):
        raise ValueError("instruction must be a string or None")
    if len(instruction) > INSTRUCTION_MAX_CHARS:
        raise ValueError(f"instruction exceeds {INSTRUCTION_MAX_CHARS} characters")
    return instruction if instruction.strip() else None


class MultilingualInstructionEncoder(nn.Module):
    """E5 query-prefix, attention-mask mean pooling, and L2 normalization.

    Weights are always frozen. Cached embeddings are detached CPU tensors;
    trainable policy projections belong to the agent and are never cached.
    Token overflow is an error, including the query prefix and special tokens.
    Inject both a tokenizer and backbone for small, offline tests. Loading
    failures propagate rather than constructing randomly initialized weights.
    """

    def __init__(
        self, *, max_tokens: int = 512, local_files_only: bool = False,
        cache_dir: str | Path | None = None, tokenizer: Any = None,
        backbone: nn.Module | None = None, cache_size: int = 32,
    ):
        super().__init__()
        if type(max_tokens) is not int or not 2 <= max_tokens <= 512:
            raise ValueError("instruction_max_tokens must be an integer between 2 and 512")
        if type(cache_size) is not int or cache_size <= 0:
            raise ValueError("instruction cache_size must be a positive integer")
        if (tokenizer is None) != (backbone is None):
            raise ValueError("inject both tokenizer and backbone")
        self.max_tokens = max_tokens
        self.cache_size = cache_size
        self.tokenizer = tokenizer if tokenizer is not None else AutoTokenizer.from_pretrained(
            TEXT_ID, revision=TEXT_REVISION, local_files_only=local_files_only,
            cache_dir=cache_dir, trust_remote_code=False,
        )
        self.backbone = backbone if backbone is not None else AutoModel.from_pretrained(
            TEXT_ID, revision=TEXT_REVISION, local_files_only=local_files_only,
            cache_dir=cache_dir, use_safetensors=True, trust_remote_code=False,
        )
        if self.backbone.config.hidden_size != TEXT_WIDTH:
            raise ValueError(f"instruction encoder must have hidden_size={TEXT_WIDTH}")
        self.backbone.requires_grad_(False)
        self._cache: OrderedDict[str, torch.Tensor] = OrderedDict()
        self._token_cache: OrderedDict[str, torch.Tensor] = OrderedDict()
        self._strict_token_cache: OrderedDict[str, torch.Tensor] = OrderedDict()
        self.train(False)

    def train(self, mode: bool = True) -> "MultilingualInstructionEncoder":
        # A frozen encoder must never activate dropout during policy updates.
        super().train(False)
        return self

    def clear_cache(self) -> None:
        self._cache.clear()
        self._token_cache.clear()
        self._strict_token_cache.clear()

    def save_resources(self, directory: str | Path) -> None:
        """Save local tokenizer/config resources; weights stay in the agent checkpoint."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.tokenizer.save_pretrained(directory)
        self.backbone.config.save_pretrained(directory)
        (directory / "carryrl_language.json").write_text(json.dumps({
            "model_id": TEXT_ID, "revision": TEXT_REVISION,
        }) + "\n", encoding="utf-8")

    @classmethod
    def from_resources(
        cls, directory: str | Path, *, max_tokens: int = 512,
    ) -> "MultilingualInstructionEncoder":
        """Reconstruct a local checkpoint graph; strict state restoration is required.

        This explicit path initializes the graph from saved config. It never
        downloads or falls back after a pretrained loading failure. Forward
        calls remain disabled until the checkpoint state has been restored.
        """
        directory = Path(directory)
        identity = json.loads((directory / "carryrl_language.json").read_text(encoding="utf-8"))
        if identity != {"model_id": TEXT_ID, "revision": TEXT_REVISION}:
            raise ValueError("local instruction resources have a different model identity/revision")
        tokenizer = AutoTokenizer.from_pretrained(
            directory, local_files_only=True, trust_remote_code=False,
        )
        config = AutoConfig.from_pretrained(
            directory, local_files_only=True, trust_remote_code=False,
        )
        backbone = AutoModel.from_config(config, trust_remote_code=False)
        language = cls(max_tokens=max_tokens, tokenizer=tokenizer, backbone=backbone)
        language._awaiting_checkpoint_state = True
        language.register_load_state_dict_post_hook(language._checkpoint_restored)
        return language

    @staticmethod
    def _checkpoint_restored(module, incompatible_keys) -> None:
        module._awaiting_checkpoint_state = bool(
            incompatible_keys.missing_keys or incompatible_keys.unexpected_keys,
        )

    def _apply(self, fn, recurse=True):
        result = super()._apply(fn, recurse=recurse)
        self.clear_cache()
        return result

    def _load_from_state_dict(self, *args, **kwargs):
        self.clear_cache()
        super()._load_from_state_dict(*args, **kwargs)

    @torch.no_grad()
    def forward(self, instruction: str) -> torch.Tensor:
        if getattr(self, "_awaiting_checkpoint_state", False):
            raise RuntimeError(
                "local instruction graph requires strict checkpoint state restoration"
            )
        instruction = validate_instruction(instruction)
        if instruction is None:
            raise ValueError("instruction encoder requires nonempty text")
        cached = self._cache.get(instruction)
        if cached is not None:
            self._cache.move_to_end(instruction)
            return cached.clone()
        tokens = self.tokenizer(
            "query: " + instruction, return_tensors="pt", padding=False, truncation=False,
        )
        count = tokens["input_ids"].shape[-1]
        if count > self.max_tokens:
            raise ValueError(
                f"instruction requires {count} tokens including prefix/special tokens; "
                f"maximum is {self.max_tokens} (text is never truncated)"
            )
        device = next(self.backbone.parameters()).device
        tokens = {name: value.to(device) for name, value in tokens.items()}
        self.backbone.eval()
        hidden = self.backbone(**tokens, return_dict=True).last_hidden_state
        mask = tokens["attention_mask"].bool()
        if hidden.shape != (*mask.shape, TEXT_WIDTH) or not mask.any(dim=1).all():
            raise RuntimeError("instruction encoder output does not match its token attention mask")
        pooled = hidden.masked_fill(~mask[..., None], 0).sum(dim=1)
        pooled = pooled / mask.sum(dim=1, keepdim=True)
        feature = F.normalize(pooled, p=2, dim=1)[0].detach().cpu()
        if not torch.isfinite(feature).all():
            raise RuntimeError("instruction encoder produced non-finite features")
        self._cache[instruction] = feature.clone()
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return feature


    @torch.no_grad()
    def encode_tokens(self, text: str) -> torch.Tensor:
        """Return frozen contextual OCR tokens, without pooling or normalization.

        OCR uses the E5 query prefix and deterministic right truncation to the
        configured token budget (including prefix/special tokens). Unlike the
        strict instruction path, an overflowing screen observation is truncated.
        Only attended tokens are retained; the independent LRU stores CPU copies.
        """
        if getattr(self, "_awaiting_checkpoint_state", False):
            raise RuntimeError(
                "local instruction graph requires strict checkpoint state restoration"
            )
        if not isinstance(text, str) or not text.strip():
            raise ValueError("screen text encoder requires nonempty text")
        if len(text) > INSTRUCTION_MAX_CHARS:
            raise ValueError(f"screen text exceeds {INSTRUCTION_MAX_CHARS} characters")
        cached = self._token_cache.get(text)
        if cached is not None:
            self._token_cache.move_to_end(text)
            return cached.clone()
        original_side = getattr(self.tokenizer, "truncation_side", "right")
        self.tokenizer.truncation_side = "right"
        try:
            tokens = self.tokenizer(
                "query: " + text, return_tensors="pt", padding=False,
                truncation=True, max_length=self.max_tokens,
            )
        finally:
            self.tokenizer.truncation_side = original_side
        # Bound injected tokenizers too; production tokenizers already obey max_length.
        tokens = {name: value[:, :self.max_tokens] for name, value in tokens.items()}
        device = next(self.backbone.parameters()).device
        tokens = {name: value.to(device) for name, value in tokens.items()}
        self.backbone.eval()
        hidden = self.backbone(**tokens, return_dict=True).last_hidden_state
        mask = tokens["attention_mask"].bool()
        if (mask.ndim != 2 or mask.shape[0] != 1
                or hidden.shape != (*mask.shape, TEXT_WIDTH) or not mask.any()):
            raise RuntimeError("screen text encoder output does not match its token attention mask")
        features = hidden[0, mask[0]].detach().cpu()
        if not torch.isfinite(features).all():
            raise RuntimeError("screen text encoder produced non-finite features")
        self._token_cache[text] = features.clone()
        if len(self._token_cache) > self.cache_size:
            self._token_cache.popitem(last=False)
        return features

    @torch.no_grad()
    def encode_instruction_tokens(self, text: str) -> torch.Tensor:
        """Frozen contextual instruction tokens; overflow fails without truncation."""
        if getattr(self, "_awaiting_checkpoint_state", False):
            raise RuntimeError(
                "local instruction graph requires strict checkpoint state restoration"
            )
        text = validate_instruction(text)
        if text is None:
            raise ValueError("instruction token encoder requires nonempty text")
        cached = self._strict_token_cache.get(text)
        if cached is not None:
            self._strict_token_cache.move_to_end(text)
            return cached.clone()
        tokens = self.tokenizer(
            "query: " + text, return_tensors="pt", padding=False, truncation=False,
        )
        count = tokens["input_ids"].shape[-1]
        if count > self.max_tokens:
            raise ValueError(
                f"instruction requires {count} tokens including prefix/special tokens; "
                f"maximum is {self.max_tokens} (text is never truncated)"
            )
        device = next(self.backbone.parameters()).device
        tokens = {name: value.to(device) for name, value in tokens.items()}
        self.backbone.eval()
        hidden = self.backbone(**tokens, return_dict=True).last_hidden_state
        mask = tokens["attention_mask"].bool()
        if (mask.ndim != 2 or mask.shape[0] != 1
                or hidden.shape != (*mask.shape, TEXT_WIDTH) or not mask.any()):
            raise RuntimeError("instruction tokens do not match the attention mask")
        result = hidden[0, mask[0]].detach().cpu()
        if not torch.isfinite(result).all():
            raise RuntimeError("instruction token encoder produced non-finite features")
        self._strict_token_cache[text] = result.clone()
        if len(self._strict_token_cache) > self.cache_size:
            self._strict_token_cache.popitem(last=False)
        return result
