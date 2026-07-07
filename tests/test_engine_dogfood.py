"""
Engine dogfood tests — real SigLIP2 inference against real images.

These tests verify the core promise: discriminative scoring that doesn't
hallucinate. Every assertion is a claim about what the model should be
able to distinguish with calibrated sigmoid scores.

Categories:
  1. Honesty — armed sprites score high for weapons, unarmed score low
  2. Discrimination — classify picks the right label from candidates
  3. Photo grounding — natural photos scored correctly
  4. Compare — same character angles are more similar than different chars
  5. Batch — batch scoring matches individual scores
  6. Status — engine reports correct state
  7. Query sensitivity — calibration curve documentation
  8. Determinism — same input same output
  9. Concurrency — thread safety
  10. RGBA — alpha channel handling
  11. Env var override — invalid threshold env var falls back gracefully
  12. Performance — regression ceiling on single score() call
"""

import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pytest
from PIL import Image

from ai_eyes_mcp.engine import SigLIPEngine

# ---------------------------------------------------------------------------
# Calibration constants — siglip2-so400m-patch14-384, 2026-04-09
# Update these if the model checkpoint or revision changes.
# ---------------------------------------------------------------------------
ARMED_DESCRIPTIVE_MIN = 0.4     # armed sprite + descriptive query (re-measured on own-IP knight: 0.885)
UNARMED_WEAPON_MAX = 0.01       # unarmed sprite + weapon query
BARD_WEAPON_MAX = 0.05          # bard + weapon query (re-measured on own-IP bard: 0.0003)
PHOTO_SPECIFIC_MIN = 0.1        # natural photo + specific label ("a cheetah")
PHOTO_WRONG_CLASS_MAX = 0.1     # natural photo + wrong class ("cheetah" on bus)
PHOTO_GENERIC_MAX = 0.01        # natural photo + generic label ("an animal")
SELF_SIMILARITY_MIN = 0.99      # image compared to itself
EMBEDDING_NORM_TOLERANCE = 0.01 # unit-norm tolerance for embeddings

# Engine construction default — shared with conftest.py engine fixture.
# None lets transformers resolve the HuggingFace cache from the environment
# (HF_HOME / HF_HUB_CACHE); set AI_EYES_MODEL_DIR to override. No rig paths.
DEFAULT_CACHE_DIR = os.environ.get("AI_EYES_MODEL_DIR") or None


# ===========================================================================
# 1. HONESTY — the founding promise
# ===========================================================================

class TestHonesty:
    """Armed vs unarmed sprites: the reason ai-eyes exists.

    Key finding: SigLIP2 sigmoid scores are query-phrasing-sensitive.
    Generic queries ("a sword") score near zero even on matching images.
    Style-aware queries ("pixel art knight with sword") score 1000x higher.
    Relative ranking (classify) is robust regardless of phrasing.
    """

    def test_knight_sword_with_descriptive_query(self, engine, knight_sword_front):
        """Knight with sword scores high when query matches image style."""
        score = engine.score(knight_sword_front, "a knight with a sword and shield")
        assert score > ARMED_DESCRIPTIVE_MIN, f"Knight w/ sword scored {score} for descriptive query — should be high"

    def test_knight_classified_as_knight(self, engine, knight_sword_front):
        """Relative ranking: knight >> cook/merchant/bard."""
        scores = engine.score_multi(
            knight_sword_front, ["knight", "cook", "merchant", "bard"]
        )
        assert scores["knight"] > scores["cook"], f"Knight should outscore cook: {scores}"
        assert scores["knight"] > scores["merchant"], f"Knight should outscore merchant: {scores}"
        assert scores["knight"] > scores["bard"], f"Knight should outscore bard: {scores}"

    def test_goblin_cook_scores_low_for_sword(self, engine, goblin_cook_front):
        """A goblin cook should score LOW for weapon queries."""
        score = engine.score(goblin_cook_front, "a character holding a sword")
        assert score < UNARMED_WEAPON_MAX, f"Goblin cook scored {score} for 'sword' — should be low"

    def test_armed_vs_unarmed_via_classify(self, engine, knight_sword_front, goblin_cook_front):
        """Classify reliably separates armed from unarmed sprites.

        Absolute scoring with generic queries can invert due to style bias
        (a goblin may look more 'pixel art character' than a 3D-rendered knight).
        Relative ranking eliminates this — each image is scored against the
        SAME label set, and the correct label wins.
        """
        knight_scores = engine.score_multi(knight_sword_front, ["armed warrior", "unarmed civilian"])
        goblin_scores = engine.score_multi(goblin_cook_front, ["armed warrior", "unarmed civilian"])

        assert knight_scores["armed warrior"] > knight_scores["unarmed civilian"], (
            f"Knight should classify as armed: {knight_scores}"
        )
        # Goblin cook doesn't need to classify as "unarmed civilian" — just shouldn't
        # classify as "armed warrior" more strongly than the knight does
        assert knight_scores["armed warrior"] > goblin_scores["armed warrior"], (
            f"Knight ({knight_scores['armed warrior']:.6f}) should outscore "
            f"goblin ({goblin_scores['armed warrior']:.6f}) for 'armed warrior'"
        )

    def test_knight_battleaxe_detected(self, engine, knight_battleaxe_front):
        """Knight with battleaxe scores high for style-matched query."""
        score = engine.score(knight_battleaxe_front, "a knight with a battleaxe")
        assert score > 0.5, f"Knight w/ battleaxe scored {score} for 'a knight with a battleaxe' — should be high"

    def test_bard_scores_low_for_weapon(self, engine, hero_bard_front):
        """A bard hero should score low for weapons."""
        score = engine.score(hero_bard_front, "a character holding a weapon")
        assert score < BARD_WEAPON_MAX, f"Bard scored {score} for 'weapon' — should be low"

    def test_avar_armed_vs_desperate(self, engine, avar_armed_front, avar_desperate_front):
        """Armed avar should outscore desperate (unarmed) avar on weapons."""
        armed = engine.score(avar_armed_front, "a character holding a weapon")
        unarmed = engine.score(avar_desperate_front, "a character holding a weapon")
        assert armed > unarmed, (
            f"Armed avar ({armed:.4f}) should outscore desperate avar ({unarmed:.4f})"
        )


# ===========================================================================
# 2. DISCRIMINATION — multi-label classification
# ===========================================================================

class TestDiscrimination:
    """Can the model pick the right label from candidates?"""

    def test_classify_cheetah(self, engine, photo_cheetah):
        """Cheetah photo should rank 'cheetah' highest."""
        scores = engine.score_multi(photo_cheetah, ["cheetah", "bus", "building", "sword"])
        best = max(scores, key=scores.get)
        assert best == "cheetah", f"Expected 'cheetah' but got '{best}': {scores}"

    def test_classify_lion(self, engine, photo_lion):
        """Lion photo should rank 'lion' highest."""
        scores = engine.score_multi(photo_lion, ["lion", "cheetah", "bus", "tower"])
        best = max(scores, key=scores.get)
        assert best == "lion", f"Expected 'lion' but got '{best}': {scores}"

    def test_classify_bus(self, engine, photo_bus):
        """Bus photo should rank 'bus' highest."""
        scores = engine.score_multi(photo_bus, ["bus", "lion", "cheetah", "tower"])
        best = max(scores, key=scores.get)
        assert best == "bus", f"Expected 'bus' but got '{best}': {scores}"

    def test_classify_tower(self, engine, photo_tower):
        """Tower photo should rank 'tower' or 'building' highest."""
        scores = engine.score_multi(photo_tower, ["tower", "building", "bus", "lion"])
        best = max(scores, key=scores.get)
        assert best in ("tower", "building"), f"Expected tower/building but got '{best}': {scores}"

    def test_classify_knight_as_warrior(self, engine, knight_sword_front):
        """Knight sprite should rank 'warrior' or 'knight' over 'cook' or 'merchant'."""
        scores = engine.score_multi(
            knight_sword_front,
            ["warrior with sword", "cook with pot", "merchant with goods", "bard with instrument"],
        )
        best = max(scores, key=scores.get)
        assert best == "warrior with sword", f"Expected 'warrior with sword' but got '{best}': {scores}"

    def test_classify_goblin_cook(self, engine, goblin_cook_front):
        """Goblin cook should NOT rank 'warrior' highest."""
        scores = engine.score_multi(
            goblin_cook_front,
            ["warrior with sword", "cook", "merchant", "knight"],
        )
        assert scores.get("warrior with sword", 0) < scores.get("cook", 0), (
            f"Cook should outscore warrior: {scores}"
        )


# ===========================================================================
# 3. PHOTO GROUNDING — natural image sanity
# ===========================================================================

class TestPhotoGrounding:
    """Basic sanity checks on natural photos.

    Specific queries ("a cheetah") score high. Generic queries
    ("an animal") score low — the model rewards specificity.
    """

    def test_cheetah_specific_query(self, engine, photo_cheetah):
        score = engine.score(photo_cheetah, "a cheetah")
        assert score > PHOTO_SPECIFIC_MIN, f"Cheetah scored {score} for 'a cheetah' — should be obvious"

    def test_bus_specific_query(self, engine, photo_bus):
        score = engine.score(photo_bus, "bus")
        assert score > PHOTO_SPECIFIC_MIN, f"Bus scored {score} for 'bus' — should be obvious"

    def test_cheetah_not_a_bus(self, engine, photo_cheetah):
        score = engine.score(photo_cheetah, "a bus")
        assert score < PHOTO_WRONG_CLASS_MAX, f"Cheetah scored {score} for 'bus' — should be near zero"

    def test_bus_not_a_cheetah(self, engine, photo_bus):
        score = engine.score(photo_bus, "a cheetah")
        assert score < UNARMED_WEAPON_MAX, f"Bus scored {score} for 'cheetah' — should be near zero"

    def test_generic_queries_score_low(self, engine, photo_cheetah, photo_bus):
        """Documents known behavior: generic queries score low on SigLIP2 sigmoid."""
        animal_score = engine.score(photo_cheetah, "an animal")
        vehicle_score = engine.score(photo_bus, "a vehicle")
        # These are TRUE but score low in absolute terms — that's the model's behavior
        assert animal_score < PHOTO_GENERIC_MAX, "Generic 'an animal' expected to score low"
        assert vehicle_score < PHOTO_GENERIC_MAX, "Generic 'a vehicle' expected to score low"


# ===========================================================================
# 4. COMPARE — image similarity
# ===========================================================================

class TestCompare:
    """Cosine similarity between image embeddings."""

    def test_same_knight_different_angles(self, engine, knight_sword_front, knight_sword_left):
        """Same knight, different angles — should be fairly similar."""
        sim = engine.compare(knight_sword_front, knight_sword_left)
        assert sim > 0.5, f"Same knight different angles: similarity {sim} — expected >0.5"

    def test_knight_vs_goblin(self, engine, knight_sword_front, goblin_cook_front):
        """Knight vs goblin cook — should be less similar than same-character angles."""
        sim = engine.compare(knight_sword_front, goblin_cook_front)
        # Not asserting low value — sprites share style. Just check it's below same-char.
        assert sim < 0.95, f"Knight vs goblin: similarity {sim} — shouldn't be near-identical"

    def test_same_char_more_similar_than_different(
        self, engine, knight_sword_front, knight_sword_left, goblin_cook_front
    ):
        """Same character similarity > cross-character similarity."""
        same = engine.compare(knight_sword_front, knight_sword_left)
        diff = engine.compare(knight_sword_front, goblin_cook_front)
        assert same > diff, (
            f"Same-char ({same:.4f}) should be more similar than cross-char ({diff:.4f})"
        )

    def test_cheetah_vs_lion(self, engine, photo_cheetah, photo_lion):
        """Big cats: similar but distinguishable."""
        sim = engine.compare(photo_cheetah, photo_lion)
        assert 0.3 < sim < 0.95, f"Cheetah/lion similarity {sim} — should be moderate"

    def test_cheetah_vs_bus(self, engine, photo_cheetah, photo_bus):
        """Animal vs vehicle: should be dissimilar."""
        sim = engine.compare(photo_cheetah, photo_bus)
        assert sim < 0.7, f"Cheetah/bus similarity {sim} — should be low"

    def test_self_similarity(self, engine, photo_cheetah):
        """Image compared to itself should be ~1.0. (Image-agnostic — re-pointed
        to a vendored photo in Wave 0; original used a sprite fixture.)"""
        sim = engine.compare(photo_cheetah, photo_cheetah)
        assert sim > SELF_SIMILARITY_MIN, f"Self-similarity {sim} — should be ~1.0"

    def test_embedding_is_normalized(self, engine, photo_cheetah):
        """Embeddings should be unit-normalized. (Image-agnostic — re-pointed.)"""
        emb = engine.embed_image(photo_cheetah)
        norm = np.linalg.norm(emb)
        assert abs(norm - 1.0) < EMBEDDING_NORM_TOLERANCE, f"Embedding norm {norm} — should be ~1.0"

    def test_embedding_dimensionality(self, engine, photo_cheetah):
        """Embedding must be a 1-D vector with positive length. (Image-agnostic.)"""
        emb = engine.embed_image(photo_cheetah)
        assert emb.ndim == 1, f"Expected 1-D embedding, got ndim={emb.ndim}"
        assert emb.shape[0] > 0, f"Embedding length must be >0, got {emb.shape[0]}"


# ===========================================================================
# 4b. EMBEDDING CONTRACT — F0-2 regression (transformers 4.x / 5.x compat)
# ===========================================================================

class TestEmbeddingContract:
    """F0-2 regression. ``get_image_features`` returns a bare tensor on
    transformers 4.x but a ``BaseModelOutputWithPooling`` on 5.x. ``embed_image``
    must return a 1-D unit-normalized vector on BOTH — and, critically, that
    vector must be the pooled IMAGE embedding (``pooler_output``), not an
    arbitrary normalized field. The semantic-ordering assertion is what pins the
    correct field: a wrong-but-unit-normalized field (e.g. a raw patch-sequence
    mean) would still pass the shape/norm checks but break the ordering.
    """

    def test_f02_embed_is_1d_unit_vector(self, engine, photo_cheetah):
        emb = engine.embed_image(photo_cheetah)
        assert emb.ndim == 1, f"expected 1-D embedding, got ndim={emb.ndim}"
        assert emb.shape[0] > 0, "embedding must be non-empty"
        norm = float(np.linalg.norm(emb))
        assert abs(norm - 1.0) < EMBEDDING_NORM_TOLERANCE, f"embedding norm {norm} — must be unit"

    def test_f02_embedding_field_preserves_semantic_ordering(
        self, engine, photo_cheetah, photo_lion, photo_bus
    ):
        """Two big cats must be more similar than a cat and a vehicle. Guards
        against get_image_features returning a wrong-but-normalized field on a
        future transformers version."""
        cat_cat = engine.compare(photo_lion, photo_cheetah)
        cat_vehicle = engine.compare(photo_lion, photo_bus)
        assert cat_cat > cat_vehicle, (
            f"lion~cheetah ({cat_cat:.4f}) must exceed lion~bus ({cat_vehicle:.4f}) — "
            f"get_image_features field is not the pooled image embedding"
        )


# ===========================================================================
# 5. BATCH — consistency
# ===========================================================================

class TestBatch:
    """Batch scoring should match individual scores."""

    def test_batch_matches_individual(
        self, engine, photo_lion, photo_bus, photo_cheetah
    ):
        """Batch scores should match individual score() calls. (Image-agnostic —
        re-pointed to vendored photos in Wave 0.)"""
        paths = [photo_lion, photo_bus, photo_cheetah]
        query = "a character holding a sword"

        batch = engine.score_batch(paths, query)
        individual = [engine.score(p, query) for p in paths]

        for i, (b, s) in enumerate(zip(batch, individual)):
            assert abs(b - s) < 0.001, (
                f"Image {i}: batch={b:.4f} vs individual={s:.4f} — should match"
            )

    def test_batch_ordering(
        self, engine, knight_sword_front, goblin_cook_front, photo_bus
    ):
        """Knight should score highest for style-matched query."""
        paths = [knight_sword_front, goblin_cook_front, photo_bus]
        scores = engine.score_batch(paths, "a knight with a sword")
        assert scores[0] > scores[1], f"Knight ({scores[0]:.6f}) should outscore goblin ({scores[1]:.6f})"
        assert scores[0] > scores[2], f"Knight ({scores[0]:.6f}) should outscore bus ({scores[2]:.6f})"


# ===========================================================================
# 6. QUERY SENSITIVITY — documenting the calibration curve
# ===========================================================================

class TestQuerySensitivity:
    """Documents SigLIP2's query-phrasing sensitivity.

    This is NOT a bug — it's the model's calibration behavior. Specific,
    style-matched queries produce strong sigmoid scores. Generic/abstract
    queries produce near-zero scores even on matching images. The model
    rewards specificity.
    """

    def test_specific_beats_generic_on_sprites(self, engine, knight_sword_front):
        """'a knight in plate armor' >> 'character' on the knight sprite."""
        specific = engine.score(knight_sword_front, "a knight in plate armor")
        generic = engine.score(knight_sword_front, "character")
        assert specific > generic * 10, (
            f"Specific ({specific:.6f}) should vastly outscore generic ({generic:.6f})"
        )

    def test_specific_beats_generic_on_photos(self, engine, photo_cheetah):
        """'a cheetah' >> 'an animal' on cheetah photo."""
        specific = engine.score(photo_cheetah, "a cheetah")
        generic = engine.score(photo_cheetah, "an animal")
        assert specific > generic * 10, (
            f"Specific ({specific:.6f}) should vastly outscore generic ({generic:.6f})"
        )

    def test_classify_robust_despite_low_absolute_scores(self, engine, knight_sword_front):
        """Relative ranking works even when absolute scores are low."""
        scores = engine.score_multi(
            knight_sword_front,
            ["knight", "cook", "merchant", "bard"],
        )
        # knight should dominate regardless of absolute magnitude
        best = max(scores, key=scores.get)
        assert best == "knight", f"Expected 'knight' but got '{best}': {scores}"


# ===========================================================================
# 7. STATUS
# ===========================================================================

class TestStatus:
    """Engine status reporting."""

    def test_status_shape(self, engine):
        """Status should report model, device, loaded state."""
        s = engine.status()
        assert "model_id" in s
        assert "device" in s
        assert "loaded" in s
        assert s["loaded"] is True
        assert "parameters" in s

    def test_status_model_id(self, engine):
        """Should report the SigLIP2 model."""
        s = engine.status()
        assert "siglip2" in s["model_id"].lower()

    def test_status_model_pinned_to_so400m(self, engine):
        """Guard against silent model-version drift — must be so400m variant."""
        s = engine.status()
        assert "so400m" in s["model_id"], (
            f"Expected so400m model variant, got '{s['model_id']}' — "
            "update DEFAULT_MODEL_ID deliberately, not by accident"
        )

    def test_lazy_loading_not_loaded_on_construction(self):
        """A fresh engine instance should NOT load the model until first use."""
        fresh = SigLIPEngine(cache_dir=DEFAULT_CACHE_DIR)
        assert fresh.loaded is False, "Engine should be lazy — not loaded on construction"


# ===========================================================================
# 8. DETERMINISM — same input → same output
# ===========================================================================

class TestDeterminism:
    """Scoring the same image+query twice must produce identical results.

    SigLIP2 inference uses torch.no_grad and eval mode — there should be
    zero randomness. This test guards against accidental dropout, stochastic
    augmentation, or floating-point non-determinism across calls.
    """

    def test_score_deterministic(self, engine, photo_cheetah):
        """Two calls with the same image and query must return the exact same score.
        (Image-agnostic — re-pointed to a vendored photo in Wave 0.)"""
        query = "pixel art knight with sword and shield"
        score_a = engine.score(photo_cheetah, query)
        score_b = engine.score(photo_cheetah, query)
        assert score_a == score_b, (
            f"Determinism violation: score_a={score_a}, score_b={score_b} — "
            f"same image+query must produce identical results"
        )

    def test_score_vs_score_multi_consistency(self, engine, photo_cheetah):
        """engine.score(img, X) must match engine.score_multi(img, [X])[X] within tolerance.

        Both paths ultimately run the same SigLIP2 forward pass, so their
        sigmoid outputs should agree within floating-point tolerance.
        (Image-agnostic — re-pointed to a vendored photo in Wave 0.)
        """
        query = "pixel art knight with sword"
        single = engine.score(photo_cheetah, query)
        multi = engine.score_multi(photo_cheetah, [query])[query]
        assert abs(single - multi) < 0.001, (
            f"score() returned {single:.6f} but score_multi() returned {multi:.6f} — "
            f"should agree within 0.001"
        )


# ===========================================================================
# 9. CONCURRENCY — thread safety
# ===========================================================================

class TestConcurrency:
    """Thread safety: concurrent scoring must not corrupt results.

    SigLIP2 uses torch.no_grad per call and eval mode — concurrent reads
    from different threads should be safe. This test verifies no data
    races or mangled outputs under parallel load.
    """

    def test_concurrent_scoring_valid_results(
        self, engine, photo_cheetah, photo_lion, photo_bus
    ):
        """3 threads scoring different images must all return valid floats in [0,1].
        (Image-agnostic — re-pointed to vendored photos in Wave 0.)"""
        jobs = [
            (photo_cheetah, "pixel art knight"),
            (photo_lion, "goblin cook"),
            (photo_bus, "a cheetah"),
        ]

        results = {}
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = {
                pool.submit(engine.score, path, query): (path, query)
                for path, query in jobs
            }
            for future in as_completed(futures):
                path, query = futures[future]
                score = future.result()
                results[(path, query)] = score

        for (path, query), score in results.items():
            assert isinstance(score, float), (
                f"Expected float, got {type(score)} for query '{query}'"
            )
            assert 0.0 <= score <= 1.0, (
                f"Score {score} out of [0,1] range for query '{query}'"
            )

    def test_concurrent_matches_sequential(
        self, engine, photo_cheetah, photo_lion, photo_bus
    ):
        """Concurrent results must match sequential results within tolerance.
        (Image-agnostic — re-pointed to vendored photos in Wave 0.)"""
        jobs = [
            (photo_cheetah, "pixel art knight"),
            (photo_lion, "goblin cook"),
            (photo_bus, "a cheetah"),
        ]

        # Sequential baseline
        sequential = {
            query: engine.score(path, query) for path, query in jobs
        }

        # Concurrent run
        concurrent = {}
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = {
                pool.submit(engine.score, path, query): query
                for path, query in jobs
            }
            for future in as_completed(futures):
                query = futures[future]
                concurrent[query] = future.result()

        for query in sequential:
            assert abs(sequential[query] - concurrent[query]) < 0.001, (
                f"Query '{query}': sequential={sequential[query]:.6f} vs "
                f"concurrent={concurrent[query]:.6f} — should match within 0.001"
            )


# ===========================================================================
# 10. RGBA — alpha channel handling
# ===========================================================================

class TestRGBAHandling:
    """RGBA images with alpha channels must be handled gracefully.

    The engine converts all images to RGB via PIL .convert("RGB"),
    stripping the alpha channel. This test verifies that RGBA sprites
    (common in game art) don't crash the pipeline.
    """

    def test_rgba_image_scores_valid(self, engine, tmp_path):
        """A programmatically created RGBA image should score without errors."""
        # Create a 64x64 red square with alpha channel
        img = Image.new("RGBA", (64, 64), (255, 0, 0, 200))
        rgba_path = tmp_path / "rgba_test.png"
        img.save(rgba_path)

        score = engine.score(str(rgba_path), "a red square")
        assert isinstance(score, float), f"Expected float, got {type(score)}"
        assert 0.0 <= score <= 1.0, f"Score {score} out of [0,1] range"

    def test_rgba_transparent_image_scores_valid(self, engine, tmp_path):
        """A fully transparent RGBA image should still return a valid score."""
        img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
        rgba_path = tmp_path / "transparent_test.png"
        img.save(rgba_path)

        score = engine.score(str(rgba_path), "empty image")
        assert isinstance(score, float), f"Expected float, got {type(score)}"
        assert 0.0 <= score <= 1.0, f"Score {score} out of [0,1] range"


# ===========================================================================
# 11. ENV VAR OVERRIDE — invalid threshold falls back gracefully
# ===========================================================================

class TestEnvVarOverride:
    """DEFAULT_THRESHOLD is evaluated at import time from AI_EYES_DEFAULT_THRESHOLD.

    Since module constants are baked in at import, the only reliable way to
    test the fallback path is to import in a fresh subprocess with the bad
    env var set.
    """

    def test_invalid_threshold_env_var_falls_back(self):
        """Setting AI_EYES_DEFAULT_THRESHOLD='banana' must fall back to 0.02.

        Spawns a fresh Python process with the invalid env var and checks
        that the module imports cleanly with the correct fallback value.
        """
        env = os.environ.copy()
        env["AI_EYES_DEFAULT_THRESHOLD"] = "banana"

        result = subprocess.run(
            [
                sys.executable, "-c",
                "from ai_eyes_mcp.engine import DEFAULT_THRESHOLD; "
                "assert DEFAULT_THRESHOLD == 0.02, "
                "f'Expected 0.02 but got {DEFAULT_THRESHOLD}'"
            ],
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, (
            f"Subprocess failed (rc={result.returncode}).\n"
            f"  stdout: {result.stdout}\n"
            f"  stderr: {result.stderr}"
        )


# ===========================================================================
# 12. PERFORMANCE — regression ceiling
# ===========================================================================

class TestPerformance:
    """Catch catastrophic performance regressions.

    A single score() call on a loaded engine should complete well under
    10 seconds. This is a generous ceiling — normal GPU inference is <1s.
    The test exists to catch regressions like accidental CPU fallback,
    runaway preprocessing, or model reload on every call.
    """

    def test_single_score_under_10_seconds(self, engine, photo_cheetah):
        """A single score() call must complete in <10 seconds.
        (Image-agnostic — re-pointed to a vendored photo in Wave 0.)"""
        start = time.perf_counter()
        score = engine.score(photo_cheetah, "pixel art knight with sword")
        elapsed = time.perf_counter() - start

        assert isinstance(score, float), f"Expected float, got {type(score)}"
        assert elapsed < 10.0, (
            f"score() took {elapsed:.2f}s — expected <10s. "
            f"Possible regression: CPU fallback, model reload per call, "
            f"or preprocessing bottleneck."
        )
