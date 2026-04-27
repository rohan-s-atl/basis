import hashlib


def build_article_hash(article: dict) -> str:
    title = str(article.get("title") or "").strip().lower()
    description = str(article.get("description") or "").strip().lower()
    published_at = str(article.get("publishedAt") or "").strip()
    raw = f"{title}|{description}|{published_at}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()
