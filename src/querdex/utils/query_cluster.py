from __future__ import annotations

import hashlib
import re
from math import sqrt

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_EMBED_DIM = 32


def compute_query_embedding(query: str) -> list[float]:
    tokens = _TOKEN_RE.findall(query.lower())
    vector = [0.0] * _EMBED_DIM
    for token in tokens:
        digest = hashlib.sha1(token.encode("utf-8"), usedforsecurity=False).digest()
        idx = digest[0] % _EMBED_DIM
        sign = 1.0 if digest[1] % 2 == 0 else -1.0
        vector[idx] += sign * (1.0 + (digest[2] / 255.0))
    norm = sqrt(sum(v * v for v in vector))
    if norm <= 0:
        return vector
    return [v / norm for v in vector]


def compute_query_cluster_id(query: str) -> str:
    embedding = compute_query_embedding(query)
    quantized = ",".join(f"{value:.2f}" for value in embedding)
    digest = hashlib.sha1(quantized.encode("utf-8"), usedforsecurity=False).hexdigest()
    return digest[:12]
