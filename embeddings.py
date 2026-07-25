"""Semantic embeddings for the shared-context retriever.

This used to be feature hashing over tokens, trigrams, and bigrams. That is a
lexical signal wearing a vector's clothing: two paraphrases with no shared words
hash into disjoint buckets, so cosine similarity is ~0 and the "semantic" leg
recovers nothing that BM25 missed. Measured on benchmarks/retrieval_v2 (a split
where queries share no content word with their target), it scored 0.05 recall@5
against 0.00 for lexical alone.

Now it runs a real sentence embedding model (fastembed, ONNX, CPU) locally. No
API key, no network after the one-time model download, so the local-only
guarantee still holds.

The model loads lazily. store.py only embeds during a search, so the session
hooks that just record events never pay the load cost.

If the model cannot be loaded, embedding falls back to the old hashed scheme but
reports a different id from active_model(). That id is the cache key for stored
vectors, so the two kinds can never be silently mixed, and the UI can tell the
user retrieval is degraded rather than quietly claiming semantic search.
"""
import hashlib
import math
import os
import re
import sys
import threading
from array import array

SEMANTIC_MODEL_NAME = os.environ.get(
    "AGENT_MEMORY_SYNC_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"
)
SEMANTIC_MODEL_ID = SEMANTIC_MODEL_NAME.rsplit("/", 1)[-1]
FALLBACK_MODEL_ID = "hashed-ngram-v1"
FALLBACK_DIM = 256

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)

_lock = threading.Lock()
_model = None
_model_failed = False
_failure_reason = ""


def _model_cache_dir() -> str:
    """Persistent model cache. fastembed defaults to a temp dir, which means the
    model is re-downloaded whenever temp is cleared."""
    override = os.environ.get("AGENT_MEMORY_SYNC_MODEL_CACHE")
    if override:
        return override
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA", os.path.expanduser("~"))
    elif sys.platform == "darwin":
        base = os.path.join(os.path.expanduser("~"), "Library", "Application Support")
    else:
        base = os.environ.get(
            "XDG_DATA_HOME", os.path.join(os.path.expanduser("~"), ".local", "share")
        )
    return os.path.join(base, "AgentMemorySync", "models")


def _load_model():
    """Return the embedding model, or None if it is unavailable."""
    global _model, _model_failed, _failure_reason
    if _model is not None or _model_failed:
        return _model
    with _lock:
        if _model is not None or _model_failed:
            return _model
        try:
            from fastembed import TextEmbedding

            cache_dir = _model_cache_dir()
            os.makedirs(cache_dir, exist_ok=True)
            _model = TextEmbedding(model_name=SEMANTIC_MODEL_NAME, cache_dir=cache_dir)
        except Exception as exc:
            _model_failed = True
            _failure_reason = f"{type(exc).__name__}: {exc}"
            _model = None
        return _model


def active_model() -> str:
    """Id of the model that actually produced vectors, used as the cache key."""
    return SEMANTIC_MODEL_ID if _load_model() is not None else FALLBACK_MODEL_ID


def is_semantic() -> bool:
    return _load_model() is not None


def status() -> dict:
    """Reportable retrieval state, so the UI can show what is really running."""
    semantic = is_semantic()
    return {
        "model": active_model(),
        "semantic": semantic,
        "detail": (
            f"Local sentence embeddings via fastembed ({SEMANTIC_MODEL_NAME})."
            if semantic
            else f"Falling back to lexical hashing; semantic model unavailable. {_failure_reason}".strip()
        ),
    }


def content_fingerprint(text: str) -> str:
    """Stable hash of embedding input, used to skip recomputation."""
    return hashlib.sha256((text or "").encode("utf-8", "ignore")).hexdigest()


def _hashed_features(text: str):
    tokens = _TOKEN_RE.findall((text or "").casefold())
    for token in tokens:
        yield "w:" + token
        if len(token) >= 4:
            for i in range(len(token) - 2):
                yield "c:" + token[i : i + 3]
    for first, second in zip(tokens, tokens[1:]):
        yield "b:" + first + "_" + second


def _hashed_embed(text: str) -> bytes:
    vector = [0.0] * FALLBACK_DIM
    for feature in _hashed_features(text):
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        index = int.from_bytes(digest[:4], "big") % FALLBACK_DIM
        vector[index] += 1.0 if digest[4] & 1 else -1.0
    norm = math.sqrt(sum(value * value for value in vector))
    if norm > 0:
        vector = [value / norm for value in vector]
    return array("f", vector).tobytes()


def _pack(values) -> bytes:
    norm = math.sqrt(sum(float(v) * float(v) for v in values))
    if norm > 0:
        return array("f", [float(v) / norm for v in values]).tobytes()
    return array("f", [float(v) for v in values]).tobytes()


def embed_texts(texts: list[str]) -> list[bytes]:
    """Embed a batch. Batching matters: the model amortizes far better than
    one call per document when indexing a corpus."""
    if not texts:
        return []
    model = _load_model()
    if model is None:
        return [_hashed_embed(text) for text in texts]
    try:
        return [_pack(vector) for vector in model.embed(list(texts))]
    except Exception:
        return [_hashed_embed(text) for text in texts]


def embed_text(text: str) -> bytes:
    """Return an L2-normalized float32 vector as raw bytes."""
    return embed_texts([text or ""])[0]


def cosine_similarity(vector_a: bytes, vector_b: bytes) -> float:
    """Dot product of two vectors from embed_text, valid since both are
    already L2-normalized, so this equals cosine similarity."""
    a = array("f")
    a.frombytes(vector_a)
    b = array("f")
    b.frombytes(vector_b)
    if len(a) != len(b):
        return 0.0
    return sum(x * y for x, y in zip(a, b))
