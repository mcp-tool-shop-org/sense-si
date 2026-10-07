"""
Dogfood swarm fixtures — real images, real model, real inference.

No mocks. SigLIP2 loads once per session and every test hits the GPU.
This IS the dogfood: if the model can't distinguish a sword from a
cooking pot, we want to know.

Fixtures are self-contained: photo assets are vendored under
``tests/assets/`` so the suite runs on any rig without external workspace
paths. The model cache is resolved from the environment (HF_HOME /
HF_HUB_CACHE, or AI_EYES_MODEL_DIR) rather than a hardcoded drive.

Sprite fixtures are quarantined pending a fixture decision — see the Wave 0
report and the ``_SPRITE_FIXTURES_PENDING`` note below.
"""

import os
from pathlib import Path

import pytest

from ai_eyes_mcp.engine import SigLIPEngine


def assert_identical_scores(loop, stacked):
    """F-W5-TESTS-001: stacked must match the per-image loop exactly, not closely."""
    assert len(loop) == len(stacked), f"len loop={len(loop)} stacked={len(stacked)}"
    for i, (a, b) in enumerate(zip(loop, stacked)):
        fa, fb = float(a), float(b)
        assert fa == fb, (
            f"index {i}: loop={fa!r} stacked={fb!r} (identical required, not close)"
        )


def is_missing_weights_error(exc: BaseException) -> bool:
    """True only when the weights are genuinely not on disk / not downloadable.

    CUDA OOM, driver errors, and ImportError are broken-load failures and
    must fail the suite, not skip-green (W1-TESTS-001).
    """
    if isinstance(exc, (RuntimeError, ImportError)):
        return False
    name = type(exc).__name__
    if name in {
        "LocalEntryNotFoundError",
        "EntryNotFoundError",
        "RepositoryNotFoundError",
        "RevisionNotFoundError",
    }:
        return True
    if isinstance(exc, OSError):
        msg = str(exc).lower()
        needles = (
            "not found",
            "no such file",
            "couldn't find",
            "could not find",
            "cannot find",
            "offline",
            "local_files_only",
            "is not a local folder",
        )
        return any(n in msg for n in needles)
    return False

# ---------------------------------------------------------------------------
# Vendored image assets — portable, committed under tests/assets/
# ---------------------------------------------------------------------------

ASSETS_DIR = Path(__file__).resolve().parent / "assets"
PHOTO_DIR = ASSETS_DIR / "photos"


def _photo(name: str) -> str:
    """Resolve a vendored photo.

    A missing photo is a hard error, NOT a skip: these assets are committed
    to the repo, so absence means a broken checkout — which the suite must
    surface loudly rather than silently skip past (the exact anti-pattern the
    original ``F:/`` fixtures fell into).
    """
    p = PHOTO_DIR / name
    if not p.is_file():
        raise FileNotFoundError(
            f"Vendored photo missing: {p}. Expected under tests/assets/photos/ "
            f"(committed to the repo). Re-vendor from a gradio "
            f"media_assets/images directory if the checkout is incomplete."
        )
    return str(p)


# ---------------------------------------------------------------------------
# Sprite fixtures — own-IP fantasy character sprites (Stage A)
#
# Generated on Comfy Cloud (Flux, non-anime) and vendored under
# tests/assets/sprites/. Three load-bearing axes: character IDENTITY (the two
# knight views share one armor design), WEAPON-VISIBLE (knight / battleaxe /
# armed orc = yes; cook / bard / desperate orc = no), and THEME legibility (the
# cook reads as a cook, the bard as a bard). The absolute-threshold assertions
# in test_engine_dogfood.py were RE-MEASURED on these exact images — the old
# magic numbers were calibrated on the gone April-2026 pixel-art sprites and
# were not reused.
# ---------------------------------------------------------------------------

SPRITE_DIR = ASSETS_DIR / "sprites"


def _sprite(name: str) -> str:
    """Resolve a vendored sprite (hard error if missing — committed asset)."""
    p = SPRITE_DIR / name
    if not p.is_file():
        raise FileNotFoundError(
            f"Vendored sprite missing: {p}. Expected under tests/assets/sprites/ "
            f"(committed to the repo)."
        )
    return str(p)


# --- Natural photos (vendored: gradio demo images, redistributable) ---


@pytest.fixture(scope="session")
def photo_cheetah():
    return _photo("cheetah.jpg")


@pytest.fixture(scope="session")
def photo_lion():
    return _photo("lion.jpg")


@pytest.fixture(scope="session")
def photo_bus():
    return _photo("bus.png")


@pytest.fixture(scope="session")
def photo_tower():
    return _photo("tower.jpg")


# --- Sprites with known content (own-IP, tests/assets/sprites/) ---


@pytest.fixture(scope="session")
def knight_sword_front():
    """Knight holding a sword and shield — front view."""
    return _sprite("knight_sword_front.png")


@pytest.fixture(scope="session")
def knight_sword_left():
    """Knight holding a sword and shield — same character, side view."""
    return _sprite("knight_sword_left.png")


@pytest.fixture(scope="session")
def knight_battleaxe_front():
    """Knight holding a battleaxe — front view."""
    return _sprite("knight_battleaxe_front.png")


@pytest.fixture(scope="session")
def goblin_cook_front():
    """Goblin cook — no weapon, cooking theme."""
    return _sprite("goblin_cook_front.png")


@pytest.fixture(scope="session")
def hero_bard_front():
    """Bard hero — front view, musical/performance theme."""
    return _sprite("hero_bard_front.png")


# --- Enemy sprites (own-IP, tests/assets/sprites/) ---


@pytest.fixture(scope="session")
def avar_armed_front():
    """Armed orc enemy — visible weapon (war-axe)."""
    return _sprite("avar_armed_front.png")


@pytest.fixture(scope="session")
def avar_desperate_front():
    """Desperate orc — unarmed variant, empty hands."""
    return _sprite("avar_desperate_front.png")


# --- Engine (loaded once per session) ---


@pytest.fixture(scope="session")
def engine():
    """Session-scoped SigLIP2 engine — loads model once, reused for all tests.

    ``cache_dir`` defaults to ``None`` so the HuggingFace cache location is
    resolved from the environment (HF_HOME / HF_HUB_CACHE). Set
    ``AI_EYES_MODEL_DIR`` to override with an explicit hub directory. No
    hardcoded rig paths — the suite stays portable.
    """
    e = SigLIPEngine(
        cache_dir=os.environ.get("AI_EYES_MODEL_DIR") or None,
    )
    # Force load now so we see any load failures immediately
    try:
        e._ensure_loaded()
    except Exception as exc:
        if is_missing_weights_error(exc):
            pytest.skip(f"SigLIP2 model not available: {exc}")
        raise
    return e
