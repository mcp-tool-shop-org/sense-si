"""
SigLIP2 vision engine — discriminative image scoring.

Wraps SigLIP2 as a measurement instrument, not a conversational model.
Returns calibrated sigmoid scores: one image-text pair → one float.

No MCP dependency. Can be used standalone or from the MCP server.

Key design decisions:
  - Lazy model loading (first call triggers download/load)
  - Sigmoid scores are independent per query (not softmax)
  - All images converted to RGB (alpha stripped)
  - Forward passes serialized by a per-engine lock (safe for concurrent callers)
"""

import logging
import os
import sys
import threading
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from ai_eyes_mcp import __version__ as _package_version

logger = logging.getLogger("ai_eyes_mcp")

# ---------------------------------------------------------------------------
# Defaults (overridable via env vars)
# ---------------------------------------------------------------------------

DEFAULT_MODEL_ID = "google/siglip2-so400m-patch14-384"
DEFAULT_MODEL_REVISION = os.environ.get("AI_EYES_MODEL_REVISION", None)
DEFAULT_CACHE_DIR = os.environ.get("AI_EYES_MODEL_DIR", None)
DEFAULT_DEVICE = os.environ.get("AI_EYES_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_DTYPE = os.environ.get("AI_EYES_DTYPE", None)  # "float16" | "bfloat16" | None (full precision)
MAX_QUERY_LENGTH = 500  # CLIP-family tokenizers truncate beyond ~77 tokens; 500 chars is generous
try:
    DEFAULT_THRESHOLD = float(os.environ.get("AI_EYES_DEFAULT_THRESHOLD", "0.02"))
except (ValueError, TypeError):
    DEFAULT_THRESHOLD = 0.02
    logger.warning(
        "Invalid AI_EYES_DEFAULT_THRESHOLD (%r), using %s",
        os.environ.get('AI_EYES_DEFAULT_THRESHOLD'),
        DEFAULT_THRESHOLD,
    )


class SigLIPEngine:
    """SigLIP2 vision scoring engine.

    Lazy-loads the model on first inference call. All scoring methods
    return raw sigmoid floats — the caller decides what threshold means.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        cache_dir: str | Path | None = DEFAULT_CACHE_DIR,
        device: str = DEFAULT_DEVICE,
        revision: str | None = DEFAULT_MODEL_REVISION,
        dtype: str | None = DEFAULT_DTYPE,
    ):
        self.model_id = model_id
        self.cache_dir = cache_dir
        self.device = device
        self.revision = revision
        self.dtype = dtype
        self._model = None
        self._processor = None
        # Serialize GPU forward passes. The engine is documented as standalone-
        # usable and is exercised concurrently by the test suite; a future
        # threaded / HTTP MCP transport would call it in parallel. The lock is
        # effectively free — inference is GPU-bound and serial on one device.
        self._forward_lock = threading.Lock()
        # Optional eager load: surface a broken model/cache at construction
        # (server start) instead of on the first tool call.
        if os.environ.get("AI_EYES_EAGER_LOAD", "").strip().lower() in ("1", "true", "yes", "on"):
            self._ensure_loaded()

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def _ensure_loaded(self):
        """Load model and processor on first use.

        Uses local variables during the load sequence so that
        ``self._model`` and ``self._processor`` are only set after the
        entire chain (download, load, eval, device transfer) succeeds.
        On any failure both attributes stay ``None`` and the next call
        retries cleanly.
        """
        if self._model is not None:
            return

        from transformers import AutoModel, AutoProcessor

        logger.info("Loading %s ...", self.model_id)

        kwargs = {}
        if self.cache_dir:
            kwargs["cache_dir"] = self.cache_dir
        if self.revision:
            kwargs["revision"] = self.revision

        try:
            processor = AutoProcessor.from_pretrained(self.model_id, **kwargs)
            model = AutoModel.from_pretrained(self.model_id, **kwargs)
            model = model.eval().to(self.device)

            # Apply dtype conversion if requested
            if self.dtype == "float16":
                model = model.half()
                logger.info("Applied float16 (half precision)")
            elif self.dtype == "bfloat16":
                model = model.bfloat16()
                logger.info("Applied bfloat16 precision")
            elif self.dtype is not None:
                logger.warning("Unknown AI_EYES_DTYPE '%s', keeping full precision", self.dtype)
        except Exception as exc:
            # Ensure no half-state: both stay None so next call retries.
            self._model = None
            self._processor = None
            logger.error(
                "Failed to load model '%s': %s\n"
                "  Hints:\n"
                "  - Check network connectivity (model may need downloading)\n"
                "  - Check HuggingFace cache dir for corruption (%s)\n"
                "  - Check GPU memory (device=%s) — try AI_EYES_DEVICE=cpu as fallback",
                self.model_id,
                exc,
                self.cache_dir or '~/.cache/huggingface',
                self.device,
            )
            raise

        # Commit only after full success
        self._processor = processor
        self._model = model

        param_count = sum(p.numel() for p in self._model.parameters())
        logger.info("Loaded on %s, %.0fM params", self.device, param_count / 1e6)

    def _load_image(self, image_path: str) -> Image.Image:
        """Load an image from path, convert to RGB."""
        path = Path(image_path)
        if path.exists() and not path.is_file():
            raise FileNotFoundError(f"Path is not a file: {image_path}")
        if not path.is_file():
            raise FileNotFoundError(f"Image not found: {image_path}")
        try:
            return Image.open(path).convert("RGB")
        except Image.DecompressionBombError:
            raise ValueError(
                f"Image too large (possible decompression bomb): {image_path}"
            )
        except (Image.UnidentifiedImageError, OSError) as exc:
            raise ValueError(
                f"Cannot open image (corrupt or unsupported format): {image_path}"
            ) from exc

    @staticmethod
    def _validate_query(query: str) -> None:
        """Raise ValueError for empty or excessively long queries."""
        if not query or not query.strip():
            raise ValueError("Query must not be empty")
        if len(query) > MAX_QUERY_LENGTH:
            raise ValueError(
                f"Query too long ({len(query)} chars, max {MAX_QUERY_LENGTH})"
            )

    def _warn_if_truncated(self, query: str) -> None:
        """Log a warning when a query tokenizes past the text encoder's limit.

        SigLIP's tokenizer silently truncates long prompts, so the score would
        reflect only the first N tokens without the caller knowing. Observability
        only — it must never raise (a truncation *check* breaking a scoring call
        would be worse than the truncation it reports).
        """
        try:
            tok = getattr(self._processor, "tokenizer", None)
            if tok is None:
                return
            # The real limit is the text encoder's positional capacity
            # (max_position_embeddings, e.g. 64 for SigLIP) — NOT the tokenizer's
            # model_max_length, which SigLIP leaves as a huge sentinel.
            cfg = getattr(self._model, "config", None)
            max_len = None
            for obj in (getattr(cfg, "text_config", None), cfg):
                if obj is not None:
                    max_len = getattr(obj, "max_position_embeddings", None)
                    if max_len:
                        break
            if not max_len:
                return
            n = len(tok(query, truncation=False, add_special_tokens=True)["input_ids"])
            if n > max_len:
                logger.warning(
                    "Query tokenizes to %d tokens but the text encoder holds %d — "
                    "the query was truncated; the score reflects only the first %d "
                    "tokens. Shorten the query.",
                    n, max_len, max_len,
                )
        except Exception:  # noqa: BLE001 — observability must never break scoring
            pass

    def score(self, image_path: str, query: str) -> float:
        """Score a single image against a single text query.

        Returns a sigmoid probability (0-1). Independent per query —
        not relative to other queries. Higher = stronger match.
        """
        self._validate_query(query)
        self._ensure_loaded()
        self._warn_if_truncated(query)

        image = self._load_image(image_path)
        inputs = self._processor(
            text=[query],
            images=image,
            padding="max_length",
            truncation=True,  # >64-token queries would otherwise crash the forward pass
            return_tensors="pt",
        ).to(self.device)

        with self._forward_lock, torch.no_grad():
            outputs = self._model(**inputs)
            prob = torch.sigmoid(outputs.logits_per_image[0, 0]).item()

        return prob

    def score_multi(self, image_path: str, queries: list[str]) -> dict[str, float]:
        """Score one image against multiple text queries.

        Returns a dict mapping each query to its independent sigmoid score.
        Scores are NOT softmax — each query is evaluated independently.

        Duplicate queries are deduplicated before scoring so every unique
        query appears exactly once in the result dict.
        """
        if not queries:
            raise ValueError("At least one query is required")
        for q in queries:
            self._validate_query(q)
        self._ensure_loaded()

        # Deduplicate while preserving order
        seen: set[str] = set()
        unique_queries: list[str] = []
        for q in queries:
            if q not in seen:
                seen.add(q)
                unique_queries.append(q)

        for q in unique_queries:
            self._warn_if_truncated(q)

        image = self._load_image(image_path)
        inputs = self._processor(
            text=unique_queries,
            images=image,
            padding="max_length",
            truncation=True,  # >64-token queries would otherwise crash the forward pass
            return_tensors="pt",
        ).to(self.device)

        with self._forward_lock, torch.no_grad():
            outputs = self._model(**inputs)
            probs = torch.sigmoid(outputs.logits_per_image[0]).cpu().numpy()

        return {q: float(p) for q, p in zip(unique_queries, probs)}

    def _encode_text(self, query: str) -> dict:
        """Encode a text query once and return the tokenized tensors.

        The returned dict contains the text-side input tensors on the
        correct device, ready to be merged with per-image inputs.
        """
        # Validate here so image_score_batch (which calls this directly rather
        # than going through score/score_multi) enforces the same query contract
        # as every other text-scoring path.
        self._validate_query(query)
        self._ensure_loaded()
        self._warn_if_truncated(query)
        text_inputs = self._processor(
            text=[query],
            padding="max_length",
            truncation=True,  # >64-token queries would otherwise crash the forward pass
            return_tensors="pt",
        ).to(self.device)
        return text_inputs

    def _score_with_text_inputs(self, image_path: str, text_inputs: dict) -> float:
        """Score a single image using pre-encoded text tensors.

        Skips text encoding — only processes the image and runs the
        forward pass with the pre-computed text input tensors.
        """
        self._ensure_loaded()
        image = self._load_image(image_path)
        image_inputs = self._processor(
            images=image,
            return_tensors="pt",
        ).to(self.device)

        # Merge text and image tensors for the forward pass
        combined = {**text_inputs, **image_inputs}

        with self._forward_lock, torch.no_grad():
            outputs = self._model(**combined)
            prob = torch.sigmoid(outputs.logits_per_image[0, 0]).item()

        return prob

    def score_batch(self, image_paths: list[str], query: str) -> list[float]:
        """Score multiple images against a single text query.

        Encodes the text once, scores each image independently.
        Returns list of sigmoid scores in same order as input paths.
        Logs progress to stderr every 25 images for large batches.
        """
        if not image_paths:
            raise ValueError("At least one image path is required")
        self._validate_query(query)
        self._ensure_loaded()

        # Encode text ONCE, reuse for every image
        text_inputs = self._encode_text(query)

        total = len(image_paths)
        scores = []
        for i, path in enumerate(image_paths):
            scores.append(self._score_with_text_inputs(path, text_inputs))
            # Progress reporting every 25 images (skip the very last — caller sees the result)
            if total > 1 and (i + 1) % 25 == 0 and (i + 1) < total:
                logger.info("Batch progress: %d/%d", i + 1, total)
        return scores

    def embed_image(self, image_path: str) -> np.ndarray:
        """Extract the image embedding vector.

        Returns a 1D numpy array (normalized). Use for cosine similarity
        comparisons between images.
        """
        self._ensure_loaded()

        image = self._load_image(image_path)
        inputs = self._processor(images=image, return_tensors="pt").to(self.device)

        with self._forward_lock, torch.no_grad():
            emb = self._model.get_image_features(**inputs)
            # transformers 4.x returns a bare tensor; 5.x returns an output
            # object (BaseModelOutputWithPooling) whose .pooler_output is the
            # pooled image embedding. Handle both, version-agnostically.
            if not torch.is_tensor(emb):
                emb = emb.pooler_output
            # clamp_min avoids 0/0 -> NaN for a degenerate (zero-norm) embedding.
            emb = emb / emb.norm(dim=-1, keepdim=True).clamp_min(1e-12)

        return emb.cpu().numpy().squeeze()

    def compare(self, image_a: str, image_b: str) -> float:
        """Compute cosine similarity between two images.

        Returns a float in [-1, 1]. Higher = more similar.
        """
        emb_a = self.embed_image(image_a)
        emb_b = self.embed_image(image_b)
        return float(np.dot(emb_a, emb_b))

    def status(self) -> dict:
        """Return engine status info."""
        import transformers

        info = {
            "ai_eyes_version": _package_version,
            "model_id": self.model_id,
            "device": self.device,
            "dtype": self.dtype or "float32",
            "loaded": self.loaded,
            "cache_dir": self.cache_dir or "default",
            "python_version": sys.version,
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
        }
        if self.loaded:
            param_count = sum(p.numel() for p in self._model.parameters())
            info["parameters"] = f"{param_count/1e6:.0f}M"
            if self.device == "cuda" and torch.cuda.is_available():
                vram_mb = torch.cuda.memory_allocated() / 1024 / 1024
                info["vram_mb"] = round(vram_mb)
        return info
