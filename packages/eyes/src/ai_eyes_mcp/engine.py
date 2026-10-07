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
import re
import sys
import threading
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from ai_eyes_mcp import __version__ as _package_version

logger = logging.getLogger("ai_eyes_mcp")


def configure_logging() -> None:
    """Idempotent stderr logging for both the MCP server and standalone import.

    Honours AI_EYES_LOG_LEVEL. Must not stack handlers if server.py and the
    engine are both imported (W1-COORD-006).
    """
    level_name = os.environ.get("AI_EYES_LOG_LEVEL", "WARNING").strip().upper()
    logger.setLevel(getattr(logging, level_name, logging.WARNING))
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(name)s [%(levelname)s] %(message)s")
        )
        logger.addHandler(handler)
        logger.propagate = False


configure_logging()

# ---------------------------------------------------------------------------
# Defaults (overridable via env vars)
# ---------------------------------------------------------------------------

DEFAULT_MODEL_ID = os.environ.get(
    "AI_EYES_MODEL_ID", "google/siglip2-so400m-patch14-384"
)
# Blessed snapshot — the SHA every score in this swarm was produced against.
# Not "main": a branch name floats. See W1-COORD-008.
PINNED_MODEL_REVISION = "e8e487298228002f3d8a82e0cd5c8ea9c567f57f"
DEFAULT_MODEL_REVISION = PINNED_MODEL_REVISION
_SHA40 = re.compile(r"^[0-9a-fA-F]{40}$")


class Score(float):
    """A sigmoid that still behaves as a float, plus honesty fields.

    Wave 6 design (W5-ENGINE-003): ``score()`` cannot stay a bare float —
    that withholds truncated and revision, which the thesis forbids. A
    parallel ``score_detailed()`` would leave the advertised ``score()``
    dishonest. A dict return is a break we are not version-bumping this
    wave (W5-CITOOL-001 is held). Subclassing float keeps
    ``isinstance(x, float)``, comparisons, and ``round(x, n)`` working
    for existing standalone callers while exposing ``.truncated`` and
    ``.revision``.
    """

    def __new__(cls, value, truncated=False, revision=""):
        return float.__new__(cls, value)

    def __init__(self, value, truncated=False, revision=""):
        self.truncated = bool(truncated)
        self.revision = str(revision)
DEFAULT_CACHE_DIR = os.environ.get("AI_EYES_MODEL_DIR", None)
DEFAULT_DEVICE = os.environ.get("AI_EYES_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_DTYPE = os.environ.get("AI_EYES_DTYPE", None)  # "float16" | "bfloat16" | None (full precision)
MAX_QUERY_LENGTH = 500  # character cap; the text encoder holds 64 tokens (see query_truncated)
# In-memory image-embedding cache. A caller's baseline pairs are the same
# images on every image_compare / image_rank call, and re-embedding them cost
# 36.5 ms each. Bounded so a long-running server scanning directories cannot
# grow without limit; 1152 float32 per entry, so the default cap is ~295 KB.
# In-memory ONLY — no disk, no sidecar, no index file.
try:
    EMBED_CACHE_MAX = max(0, int(os.environ.get("AI_EYES_EMBED_CACHE", "64")))
except (ValueError, TypeError):
    EMBED_CACHE_MAX = 64
    logger.warning(
        "Invalid AI_EYES_EMBED_CACHE (%r), using %s",
        os.environ.get("AI_EYES_EMBED_CACHE"),
        EMBED_CACHE_MAX,
    )
_THRESHOLD_FALLBACK = 0.02
try:
    DEFAULT_THRESHOLD = float(os.environ.get("AI_EYES_DEFAULT_THRESHOLD", "0.02"))
except (ValueError, TypeError):
    DEFAULT_THRESHOLD = _THRESHOLD_FALLBACK
    logger.warning(
        "Invalid AI_EYES_DEFAULT_THRESHOLD (%r), using %s",
        os.environ.get("AI_EYES_DEFAULT_THRESHOLD"),
        DEFAULT_THRESHOLD,
    )
else:
    # NaN-safe: a NaN fails the chained compare. Out-of-range is a loud
    # fallback, not a hard refuse — the tools would reject it at call time
    # anyway, and unlike a floating revision this cannot silently corrupt
    # a measurement (W1-ENGINE-005).
    if not (0.0 <= DEFAULT_THRESHOLD <= 1.0):
        logger.warning(
            "AI_EYES_DEFAULT_THRESHOLD=%r is outside [0.0, 1.0]; using %s",
            os.environ.get("AI_EYES_DEFAULT_THRESHOLD"),
            _THRESHOLD_FALLBACK,
        )
        DEFAULT_THRESHOLD = _THRESHOLD_FALLBACK


def assert_cold_status(s: dict) -> None:
    """CI/verify gate for a freshly constructed, unloaded engine.

    Must be able to fail: loaded lying, a missing/wrong revision, or a
    missing model_id. Truthy-only checks on model_id/device cannot (W1-CITOOL-002).
    """
    if s.get("loaded") is not False:
        raise AssertionError(
            f"cold engine must report loaded=False (status does not load the "
            f"model); got loaded={s.get('loaded')!r}"
        )
    if s.get("revision") != PINNED_MODEL_REVISION:
        raise AssertionError(
            f"status revision {s.get('revision')!r} does not match the pin "
            f"{PINNED_MODEL_REVISION}"
        )
    if s.get("model_id") != DEFAULT_MODEL_ID:
        raise AssertionError(
            f"status model_id {s.get('model_id')!r} != {DEFAULT_MODEL_ID!r}"
        )


def validate_model_revision(revision: str | None) -> str:
    """Return a 40-character hex SHA, or raise.

    ``None`` means the blessed pin. Empty string, ``main``, a tag, a
    branch, or any non-SHA is a load failure — not a silent fallback.
    """
    if revision is None:
        return PINNED_MODEL_REVISION
    if not isinstance(revision, str):
        revision = str(revision)
    stripped = revision.strip()
    if not _SHA40.fullmatch(stripped):
        raise ValueError(
            f"AI_EYES_MODEL_REVISION={revision!r} is not a 40-character hex "
            "commit SHA. Branch names and tags (including 'main') are refused "
            "because they resolve to different weights over time. Pass a "
            f"commit SHA. The blessed default is {PINNED_MODEL_REVISION}."
        )
    return stripped.lower()


def display_round(value: float, ndigits: int = 4) -> float:
    """Round ``value`` for a payload without turning a non-zero into 0.0.

    Decide on full precision elsewhere; this is display only. A tiny
    sigmoid that would round to 0 at ``ndigits`` keeps significant digits
    so the payload does not look like a calibrated zero.
    """
    rounded = round(value, ndigits)
    if value != 0 and rounded == 0:
        return float(format(abs(value), ".5g")) * (1.0 if value > 0 else -1.0)
    return rounded


def round_preserving_gt(value: float, other: float, ndigits: int = 4) -> float:
    """Round ``value`` so ``(value > other)`` is unchanged after rounding."""
    present = value > other
    for d in range(ndigits, 16):
        r = display_round(value, d)
        if (r > other) == present:
            return r
    return value


def round_pair_preserving_order(a: float, b: float, ndigits: int = 4) -> tuple[float, float]:
    """Round a pair so ``(a > b)`` equals ``(ra > rb)``."""
    present = a > b
    for d in range(ndigits, 16):
        ra, rb = display_round(a, d), display_round(b, d)
        if (ra > rb) == present:
            return ra, rb
    return a, b


def confidence_from_magnitude(magnitude: float) -> str:
    if magnitude >= 0.3:
        return "high"
    if magnitude >= 0.1:
        return "moderate"
    return "low — target and best alternative are close; treat as inconclusive"


def round_margin(margin: float, ndigits: int = 4) -> float:
    """Round a margin so its confidence band matches the unrounded band."""
    band = confidence_from_magnitude(abs(margin))
    for d in range(ndigits, 16):
        r = display_round(margin, d)
        if confidence_from_magnitude(abs(r)) == band:
            return r
    return margin


class _TokenIdConfigWarningFilter(logging.Filter):
    """Drops the benign SigLIP ``bos_token_id`` / ``eos_token_id`` config
    warnings — they fire on every model load and would clutter stderr for a
    hand-off. Targeted: only these two messages; all other transformers warnings
    pass through untouched.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        return not ("bos_token_id must be" in msg or "eos_token_id must be" in msg)


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
        revision: str | None = None,
        dtype: str | None = DEFAULT_DTYPE,
    ):
        self.model_id = model_id
        self.cache_dir = cache_dir
        self.device = device
        if revision is None:
            env_rev = os.environ.get("AI_EYES_MODEL_REVISION")
            revision = PINNED_MODEL_REVISION if env_rev is None else env_rev
        self.revision = validate_model_revision(revision)
        self._resolved_revision = self.revision
        self.dtype = dtype
        self._model = None
        self._processor = None
        self._vram_mb: int | None = None
        # Serialize GPU forward passes. FastMCP runs sync tools on a worker
        # threadpool against one engine; inference is GPU-bound and serial
        # on one device. Do NOT reuse this lock for model load — a 10-20s
        # load would stall every in-flight caller (W1-ENGINE-001).
        self._forward_lock = threading.Lock()
        self._load_lock = threading.Lock()
        # Path+mtime+size -> embedding. Guarded by its own lock: a cache hit
        # must not queue behind an in-flight forward pass.
        self._embed_cache: "OrderedDict[tuple, np.ndarray]" = OrderedDict()
        self._embed_cache_lock = threading.Lock()
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

        with self._load_lock:
            self._ensure_loaded_locked()

    def _ensure_loaded_locked(self):
        """Load the model. Caller holds ``_load_lock``. Re-checks after acquire."""
        if self._model is not None:
            return

        from transformers import AutoModel, AutoProcessor

        # Suppress the benign SigLIP bos/eos_token_id config warnings on the
        # "transformers" logger (targeted — only those two messages, so real
        # transformers warnings still surface).
        _tf_log = logging.getLogger("transformers")
        if not any(isinstance(f, _TokenIdConfigWarningFilter) for f in _tf_log.filters):
            _flt = _TokenIdConfigWarningFilter()
            _tf_log.addFilter(_flt)
            for _h in list(_tf_log.handlers):
                _h.addFilter(_flt)

        logger.info("Loading %s ...", self.model_id)

        kwargs = {
            "revision": self.revision,  # unconditional — omitting this is the float
        }
        if self.cache_dir:
            kwargs["cache_dir"] = self.cache_dir

        cuda = self.device == "cuda" and torch.cuda.is_available()
        vram_before = torch.cuda.memory_allocated() if cuda else None

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
        cfg = getattr(model, "config", None)
        resolved = getattr(cfg, "_commit_hash", None) if cfg is not None else None
        self._resolved_revision = resolved or self.revision
        if vram_before is not None:
            delta = torch.cuda.memory_allocated() - vram_before
            self._vram_mb = round(delta / 1024 / 1024)

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

    def query_truncated(self, query: str) -> bool:
        """Return True if ``query`` tokenizes past the text encoder's capacity.

        The encoder holds ``max_position_embeddings`` tokens (64 for SigLIP2),
        not CLIP's 77 and not ``MAX_QUERY_LENGTH`` characters. Must never raise.
        """
        return self._warn_if_truncated(query)

    def _warn_if_truncated(self, query: str) -> bool:
        """Log a warning when a query tokenizes past the text encoder's limit.

        Returns True if the score will reflect a truncated prefix. Observability
        must never raise (a truncation *check* breaking a scoring call would be
        worse than the truncation it reports).
        """
        try:
            tok = getattr(self._processor, "tokenizer", None)
            if tok is None:
                return False
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
                return False
            n = len(tok(query, truncation=False, add_special_tokens=True)["input_ids"])
            if n > max_len:
                logger.warning(
                    "Query tokenizes to %d tokens but the text encoder holds %d — "
                    "the query was truncated; the score reflects only the first %d "
                    "tokens. Shorten the query.",
                    n, max_len, max_len,
                )
                return True
            return False
        except Exception:  # noqa: BLE001 — observability must never break scoring
            return False

    def score(self, image_path: str, query: str) -> float:
        """Score a single image against a single text query.

        Returns a ``Score`` (float subclass) in 0-1. Independent per query —
        not relative to other queries. Higher = stronger match.
        ``.truncated`` and ``.revision`` name whether the query was cut
        and which weights produced the number.
        """
        self._validate_query(query)
        self._ensure_loaded()
        truncated = self._warn_if_truncated(query)

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

        return Score(prob, truncated=truncated, revision=self._resolved_revision)

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

        truncated_map = {q: self._warn_if_truncated(q) for q in unique_queries}

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

        rev = self._resolved_revision
        return {
            q: Score(float(p), truncated=truncated_map[q], revision=rev)
            for q, p in zip(unique_queries, probs)
        }

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

    def _score_with_text_inputs(
        self, image_path: str, text_inputs: dict, truncated: bool = False
    ) -> Score:
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

        return Score(prob, truncated=truncated, revision=self._resolved_revision)

    def _score_batch_loop(
        self, image_paths: list[str], text_inputs: dict, truncated: bool
    ) -> list[Score]:
        """Per-image forward. The reference implementation for the equality gate."""
        scores: list[Score] = []
        total = len(image_paths)
        for i, path in enumerate(image_paths):
            scores.append(
                self._score_with_text_inputs(path, text_inputs, truncated=truncated)
            )
            if total > 1 and (i + 1) % 25 == 0 and (i + 1) < total:
                logger.info("Batch progress: %d/%d", i + 1, total)
        return scores

    def _score_batch_stacked(
        self, image_paths: list[str], text_inputs: dict, truncated: bool
    ) -> list[Score]:
        """Many images, one (or chunked) forward.

        ANDON F-W5-ENGINE-001 — HELD OPEN, wave 9. This is still the loop.

        A stacked forward is NOT bit-identical to the per-image loop: a batched
        matmul reduces in a different order, which is arithmetic, not a bug.
        Padding every chunk to a fixed size does NOT recover the single-image
        number either — no batch size except 1 reproduces ``score(x)``, and
        n=1 IS the loop. That much was known before wave 9.

        What wave 9 measured, on this pin (RTX 5090, torch 2.11.0+cu128,
        fp32), is why naming the batch size in the payload does not close it:

        * The divergence REACHES THE PAYLOAD. Over the 11 vendored fixture
          images at batch_size=8, 4 of 11 print a different number than the
          loop — e.g. tower ``5.8359e-12`` vs ``5.836e-12``, goblin_cook
          ``2.3848e-08`` vs ``2.3851e-08``. ``display_round`` keeps five
          SIGNIFICANT digits for a score too small to survive 4-decimal
          rounding (so a tiny sigmoid does not print as a calibrated zero),
          and SigLIP2 scores non-matching images at 1e-12..1e-5 — so most of a
          real batch lands in that branch, where a ~1e-5..1e-4 relative
          divergence is plainly visible. "Invisible at 4 dp" holds only for
          scores that survive 4-decimal rounding.
        * It would put TWO CALIBRATIONS in one server. ``image_contains(x)``
          stays at batch size 1 by design, so it and ``image_score_batch([x])``
          would print different digits for the same image, same query, same
          revision. A stamp explains that; it does not reconcile it.
        * The throughput win is real but hardware-shaped and NON-MONOTONIC:
          1.65x at bs=8, 1.95x at bs=12, 1.38x at bs=16, and 0.47x at bs=100
          (median of 7 reps, N=24). Any baked-in chunk size is tuned to one
          rig while also DETERMINING the score.
        * There is no number-preserving version of the win. The forward pass is
          82% of batch cost (35.96 of 43.6 ms/img); batching the image
          preprocessing instead is bit-identical but buys 0.99x.

        VRAM is not the constraint and never was: ~35 MB marginal per image,
        8.2 GB peak at bs=100 of 32.6 GB, so a chunk never stops fitting inside
        the 100-image cap. Throughput is what bounds a chunk, and its optimum
        is not monotonic.

        The halted design is IMPLEMENTABLE — a fixed batch size is bit-
        reproducible and padding content provably does not touch the real
        images' scores (``test_fixed_batch_size_is_reproducible``). What blocks
        it is a product judgement about shipping a second calibration, not an
        engineering gap. Evidence lives in
        ``test_stacking_divergence_is_payload_visible``; a RED there is the
        signal to re-open this.
        """
        return self._score_batch_loop(image_paths, text_inputs, truncated)

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
        truncated = self.query_truncated(query)
        return self._score_batch_stacked(image_paths, text_inputs, truncated)

    @staticmethod
    def _embed_cache_key(image_path: str):
        """Identity of the BYTES, not just the path.

        Keying on the path alone would hand back an embedding for a file that
        has since been overwritten — the instrument would state something it
        did not measure. mtime + size makes a rewritten file a different key.
        Returns None when the file cannot be stat'ed, so an unreadable path
        falls through uncached and raises the normal load error.
        """
        try:
            st = Path(image_path).stat()
        except OSError:
            return None
        return (str(Path(image_path).resolve()), st.st_mtime_ns, st.st_size)

    def embed_image(self, image_path: str) -> np.ndarray:
        """Extract the image embedding vector.

        Returns a 1D numpy array (normalized). Use for cosine similarity
        comparisons between images.

        Embeddings are memoised in-process (see ``EMBED_CACHE_MAX``). This
        cannot move a number: ``embed_image`` is bit-identical across repeat
        calls on a pinned revision, so a hit returns exactly what a recompute
        would have. Callers get a private copy so a mutation cannot poison it.
        """
        key = self._embed_cache_key(image_path)
        if key is not None:
            with self._embed_cache_lock:
                hit = self._embed_cache.get(key)
                if hit is not None:
                    self._embed_cache.move_to_end(key)
                    return hit.copy()

        emb = self._embed_image_uncached(image_path)

        if key is not None and EMBED_CACHE_MAX > 0:
            with self._embed_cache_lock:
                self._embed_cache[key] = emb.copy()
                self._embed_cache.move_to_end(key)
                while len(self._embed_cache) > EMBED_CACHE_MAX:
                    self._embed_cache.popitem(last=False)
        return emb

    def _embed_image_uncached(self, image_path: str) -> np.ndarray:
        """The real forward pass. Split out so the cache is testable and so a
        subclass/stub can be counted."""
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

    def similarities_to_reference(
        self, reference: str, candidates: list[str]
    ) -> list[float]:
        """Cosine similarity of each candidate to one reference embedding.

        Encodes the reference once. In-memory only — no disk index.
        """
        if not candidates:
            raise ValueError("At least one candidate is required")
        ref = self.embed_image(reference)
        return [float(np.dot(ref, self.embed_image(c))) for c in candidates]

    def compare(self, image_a: str, image_b: str) -> float:
        """Compute cosine similarity between two images.

        Returns a float in [-1, 1]. Higher = more similar.
        """
        emb_a = self.embed_image(image_a)
        emb_b = self.embed_image(image_b)
        return float(np.dot(emb_a, emb_b))

    def verify(self, image_path: str, target: str, contrasts: list[str]) -> dict:
        """Relative honest verdict: is ``target`` a better description of the
        image than any caller-supplied ``contrast``?

        Composes ``score_multi`` (so it inherits query validation, truncation,
        and the forward-pass lock — no new inference code). Returns a DECISION +
        margin + a confidence band that DESCRIBES the measured gap, rather than
        an absolute threshold dressed up as certainty. ``contrasts`` is REQUIRED:
        this verb is relative, not absolute.
        """
        if not contrasts:
            raise ValueError(
                "image_verify needs at least one contrast alternative; it is "
                "RELATIVE, not absolute. For a raw score use image_contains."
            )
        scores = self.score_multi(image_path, [target] + list(contrasts))
        target_score = scores[target]
        # Best alternative = max over the contrasts, EXCLUDING the target itself
        # (score_multi dedupes, so a target repeated in contrasts must not be
        # allowed to beat itself).
        alt_scores = {c: scores[c] for c in contrasts if c != target}
        if not alt_scores:
            raise ValueError(
                "image_verify needs at least one contrast that differs from the target."
            )
        best_alt = max(alt_scores, key=alt_scores.get)
        best_alt_score = alt_scores[best_alt]
        margin = target_score - best_alt_score
        present = target_score > best_alt_score
        confidence = confidence_from_magnitude(abs(margin))
        disp_target, disp_alt = round_pair_preserving_order(target_score, best_alt_score)
        truncated = self.query_truncated(target) or any(
            self.query_truncated(c) for c in contrasts
        )
        return {
            "present": present,
            "target": target,
            "target_score": disp_target,
            "best_alternative": best_alt,
            "best_alternative_score": disp_alt,
            "margin": round_margin(margin),
            "confidence": confidence,
            "truncated": truncated,
            "revision": self._resolved_revision,
        }

    def status(self) -> dict:
        """Return engine status info."""
        import transformers

        info = {
            "ai_eyes_version": _package_version,
            "model_id": self.model_id,
            "revision": self._resolved_revision if self.loaded else self.revision,
            "device": self.device,
            "loaded": self.loaded,
            "cache_dir": self.cache_dir or "default",
            "python_version": sys.version,
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
        }
        if self.loaded:
            param_dtype = next(self._model.parameters()).dtype
            info["dtype"] = str(param_dtype).replace("torch.", "")
            param_count = sum(p.numel() for p in self._model.parameters())
            info["parameters"] = f"{param_count/1e6:.0f}M"
            if self._vram_mb is not None:
                info["vram_mb"] = self._vram_mb
        else:
            info["dtype"] = (
                self.dtype if self.dtype in ("float16", "bfloat16") else "float32"
            )
        return info

    def selftest(self) -> dict:
        """Run a few DECISIVE, known orderings on bundled reference images to
        prove the model loaded and is calibrated. Returns a pass/fail report.

        Reference images ship as package data (``assets/selftest/``) so the
        check works from an installed wheel, not just the source tree. Loads the
        model on first use (via ``score``).
        """
        refs = Path(__file__).resolve().parent / "assets" / "selftest"
        knight = str(refs / "knight.png")
        cook = str(refs / "cook.png")
        cheetah = str(refs / "cheetah.jpg")
        checks: list[dict] = []

        def _order(name: str, a: float, b: float, note: str) -> None:
            disp_a, disp_b = round_pair_preserving_order(a, b, ndigits=5)
            checks.append({
                "name": name,
                "expected": note,
                "measured_a": display_round(disp_a, 5),
                "measured_b": display_round(disp_b, 5),
                "ok": a > b,
            })

        # Each pair has large measured headroom (Stage-A calibration) so the
        # self-test itself is not flaky.
        _order(
            "armed_vs_unarmed",
            self.score(knight, "a knight with a sword and shield"),
            self.score(cook, "a knight with a sword and shield"),
            "armed knight > unarmed cook for a knight-with-weapon query",
        )
        _order(
            "photo_true_vs_wrong_label",
            self.score(cheetah, "a cheetah"),
            self.score(cheetah, "a bus"),
            "cheetah photo scores 'a cheetah' > 'a bus'",
        )
        _order(
            "self_vs_cross_similarity",
            self.compare(knight, knight),
            self.compare(knight, cook),
            "compare(knight, knight) > compare(knight, cook)",
        )

        info = self.status()
        return {
            "passed": all(c["ok"] for c in checks),
            "checks": checks,
            "model_id": info["model_id"],
            "revision": info["revision"],
            "device": info["device"],
            "torch_version": info["torch_version"],
            "transformers_version": info["transformers_version"],
        }
