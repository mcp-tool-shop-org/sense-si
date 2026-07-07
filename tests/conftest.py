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
# Sprite fixtures — QUARANTINED (Wave 0)
#
# The original sprite sources (sprite-foundry pipeline/gold/crops and
# the-fractured-road ch1-enemies) are April-2026 generation artifacts that no
# longer exist on the rig, and — being unreleased game art — must NOT be
# vendored into this (potentially public) repo. Until a public-safe sprite
# fixture set is decided (candidate: vendor from the SHIPPED public
# @sprite-foundry packs — heroes / villains / goblin / townsfolk-hd), these
# fixtures skip with an explicit, visible reason.
#
# This is a DELIBERATE, LOUD quarantine — the opposite of the silent F:/-path
# skip it replaces. Every quarantined test names this decision in its skip
# reason so the gap is impossible to miss. Image-AGNOSTIC tests that merely
# needed "a valid image" have been re-pointed to the vendored photos and run
# for real; only tests that assert game-sprite SEMANTICS land here.
# ---------------------------------------------------------------------------

_SPRITE_FIXTURES_PENDING = (
    "QUARANTINED (Wave 0): game-sprite fixture unavailable — original sprite "
    "assets are absent on this rig and unreleased game art must not be vendored "
    "into this repo. Pending sprite-fixture decision (see Wave 0 report). "
    "Deliberate visible quarantine, not a silent skip."
)


def _sprite_pending():
    pytest.skip(_SPRITE_FIXTURES_PENDING)


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


# --- Sprites with known content (quarantined — pending Wave 0 decision) ---


@pytest.fixture(scope="session")
def knight_sword_front():
    """Knight holding a sword and shield — front view."""
    _sprite_pending()


@pytest.fixture(scope="session")
def knight_sword_left():
    """Knight holding a sword and shield — left view (same character)."""
    _sprite_pending()


@pytest.fixture(scope="session")
def knight_battleaxe_front():
    """Knight holding a battleaxe — front view."""
    _sprite_pending()


@pytest.fixture(scope="session")
def goblin_cook_front():
    """Goblin cook — no weapon, cooking theme."""
    _sprite_pending()


@pytest.fixture(scope="session")
def goblin_merchant_front():
    """Goblin merchant — no weapon, commerce theme."""
    _sprite_pending()


@pytest.fixture(scope="session")
def hero_bard_front():
    """Bard hero — front view, musical/performance theme."""
    _sprite_pending()


# --- Enemy sprites (quarantined — pending Wave 0 decision) ---


@pytest.fixture(scope="session")
def avar_armed_front():
    """Armed avar enemy — has visible weapon."""
    _sprite_pending()


@pytest.fixture(scope="session")
def avar_desperate_front():
    """Desperate avar — unarmed variant."""
    _sprite_pending()


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
    except (OSError, RuntimeError, ImportError) as exc:
        pytest.skip(f"SigLIP2 model not available: {exc}")
    return e
