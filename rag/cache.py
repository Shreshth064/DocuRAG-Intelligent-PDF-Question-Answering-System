"""An optional Redis-backed cache for answered queries.

A repeated question against an unchanged corpus can skip retrieval and the
LLM entirely: the previous answer is returned straight from Redis. The
cache is deliberately best-effort. Every interaction with Redis is guarded,
and if the server is unreachable the cache disables itself and the caller
proceeds uncached. A cache outage can therefore never fail a request or
crash startup -- it only removes the speed-up.
"""

import hashlib
import json
import logging
import os

import redis

logger = logging.getLogger(__name__)

DEFAULT_REDIS_URL = "redis://localhost:6379"
DEFAULT_TTL_SECONDS = 24 * 60 * 60
DEFAULT_NAMESPACE = "docurag:query"

# Kept short so an unreachable Redis degrades quickly instead of stalling a
# request or startup behind a long TCP timeout.
_CONNECT_TIMEOUT_SECONDS = 0.5


def redis_url() -> str:
    """The configured Redis URL, or the local default."""
    return os.environ.get("REDIS_URL", DEFAULT_REDIS_URL)


def _connect(url: str) -> redis.Redis:
    """Open a Redis connection and prove it is live with a PING.

    Factored out so the connection can be swapped in tests without a real
    server, and so from_env has a single failure point to guard.
    """
    client = redis.Redis.from_url(
        url,
        socket_connect_timeout=_CONNECT_TIMEOUT_SECONDS,
        socket_timeout=_CONNECT_TIMEOUT_SECONDS,
    )
    client.ping()
    return client


class QueryCache:
    """Caches query responses keyed on the question and the corpus identity.

    Constructed with a live client for a working cache, or with None for a
    disabled one. from_env builds the disabled variant automatically when
    Redis cannot be reached, so callers never have to special-case an outage.
    """

    def __init__(
        self,
        client: "redis.Redis | None",
        ttl: int = DEFAULT_TTL_SECONDS,
        namespace: str = DEFAULT_NAMESPACE,
    ):
        self._client = client
        self._ttl = ttl
        self._namespace = namespace

    @classmethod
    def from_env(cls, url: str | None = None, **kwargs) -> "QueryCache":
        """Build a cache from REDIS_URL, degrading to disabled on any failure."""
        url = url or redis_url()
        try:
            client = _connect(url)
        except Exception:
            logger.warning(
                "Redis unreachable at %s; running without a query cache", url
            )
            return cls(None, **kwargs)
        logger.info("query cache connected to Redis at %s", url)
        return cls(client, **kwargs)

    @property
    def enabled(self) -> bool:
        """True when a live client is attached and lookups may hit Redis."""
        return self._client is not None

    def _key(self, question: str, store_id: str) -> str:
        # store_id first so a rebuilt corpus never collides with old answers;
        # the NUL separator keeps the two fields from blurring together.
        digest = hashlib.sha256(
            f"{store_id}\x00{question}".encode()
        ).hexdigest()
        return f"{self._namespace}:{digest}"

    def get(self, question: str, store_id: str) -> "dict | None":
        """Return the cached response, or None on a miss or any Redis error."""
        if self._client is None:
            return None
        key = self._key(question, store_id)
        try:
            raw = self._client.get(key)
        except Exception:
            # Broad on purpose: a read failure must degrade to a miss, never
            # surface to the request.
            logger.warning("cache read failed; serving uncached", exc_info=True)
            return None
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except (ValueError, TypeError):
            logger.warning("discarding corrupt cache entry at %s", key)
            return None

    def set(self, question: str, store_id: str, value: dict) -> None:
        """Store a response under its key, swallowing any Redis error."""
        if self._client is None:
            return
        try:
            self._client.set(
                self._key(question, store_id), json.dumps(value), ex=self._ttl
            )
        except Exception:
            # A write failure only costs a future cache hit; never propagate.
            logger.warning("cache write failed; continuing", exc_info=True)
