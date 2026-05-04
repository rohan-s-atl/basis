from __future__ import annotations

import logging
import math

from app.services.cache_service import cache

logger = logging.getLogger(__name__)

SIMILARITY_THRESHOLD = 0.92
_EMBEDDING_MODEL = "text-embedding-3-small"
_EMBEDDING_CACHE_TTL = 86400  # 24 h

_ANCHORS = {
    "positive": "strong economic growth market rally bullish upside opportunity",
    "negative": "severe economic shock market crash bearish downside systemic risk",
    "volatile": "extreme market volatility uncertainty fear turbulence instability",
}

_anchor_cache: dict[str, list[float]] = {}


def embed_text(text: str) -> list[float] | None:
    from app.core.config import settings

    if not settings.openai_api_key:
        return None

    cache_key = f"emb:{hash(text[:500])}"
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    try:
        from openai import OpenAI
        client = OpenAI(api_key=settings.openai_api_key)
        response = client.embeddings.create(model=_EMBEDDING_MODEL, input=text[:8000])
        embedding = response.data[0].embedding
        cache.set(cache_key, embedding, _EMBEDDING_CACHE_TTL)
        return embedding
    except Exception as exc:
        logger.warning("Embedding request failed: %s", exc)
        return None


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def embedding_norm(embedding: list[float]) -> float:
    return round(math.sqrt(sum(x * x for x in embedding)), 6)


def anchor_similarities(embedding: list[float]) -> dict[str, float]:
    result = {"pos_sim": 0.5, "neg_sim": 0.5, "vol_sim": 0.5}
    key_map = {"positive": "pos_sim", "negative": "neg_sim", "volatile": "vol_sim"}

    for anchor_key, phrase in _ANCHORS.items():
        if anchor_key not in _anchor_cache:
            anchor_emb = embed_text(phrase)
            if anchor_emb:
                _anchor_cache[anchor_key] = anchor_emb

        if anchor_key in _anchor_cache:
            sim = cosine_similarity(embedding, _anchor_cache[anchor_key])
            result[key_map[anchor_key]] = round(sim, 6)

    return result


def is_semantic_duplicate(
    new_embedding: list[float],
    candidate_embeddings: list[list[float]],
) -> bool:
    return most_similar_duplicate_index(new_embedding, candidate_embeddings) is not None


def most_similar_duplicate_index(
    new_embedding: list[float],
    candidate_embeddings: list[list[float]],
) -> int | None:
    best_index: int | None = None
    best_similarity = SIMILARITY_THRESHOLD
    for index, existing in enumerate(candidate_embeddings):
        similarity = cosine_similarity(new_embedding, existing)
        if similarity >= best_similarity:
            best_similarity = similarity
            best_index = index
    return best_index
