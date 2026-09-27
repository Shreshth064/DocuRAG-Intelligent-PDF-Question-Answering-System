"""Unit tests for the Redis query cache.

Everything runs offline: a live client is a fakeredis instance, an outage
is the RaisingRedis double, and from_env's connection is monkeypatched, so
no test ever opens a socket.
"""

import json

import pytest

from rag import cache as cache_module
from rag.cache import DEFAULT_REDIS_URL, QueryCache, redis_url


@pytest.fixture
def raising_redis():
    from conftest import RaisingRedis

    return RaisingRedis()


# --- keying ---

def test_key_changes_with_question(fake_redis):
    c = QueryCache(fake_redis)

    assert c._key("a", "store") != c._key("b", "store")


def test_key_changes_with_store_identity(fake_redis):
    c = QueryCache(fake_redis)

    assert c._key("q", "store-1") != c._key("q", "store-2")


def test_key_is_namespaced(fake_redis):
    c = QueryCache(fake_redis, namespace="ns")

    assert c._key("q", "s").startswith("ns:")


# --- miss then hit ---

def test_miss_returns_none(fake_redis):
    assert QueryCache(fake_redis).get("q", "store") is None


def test_set_then_get_round_trips(fake_redis):
    c = QueryCache(fake_redis)
    value = {"answer": "42", "pages": [1, 2], "num_sources": 2}

    c.set("q", "store", value)

    assert c.get("q", "store") == value


def test_set_applies_ttl(fake_redis):
    c = QueryCache(fake_redis, ttl=123)
    c.set("q", "store", {"answer": "x"})

    assert fake_redis.ttl(c._key("q", "store")) == 123


def test_hit_is_scoped_to_store_identity(fake_redis):
    c = QueryCache(fake_redis)
    c.set("q", "store-1", {"answer": "old"})

    # Same question, different corpus -> not a hit.
    assert c.get("q", "store-2") is None


# --- disabled cache (client is None) ---

def test_disabled_cache_is_not_enabled():
    assert QueryCache(None).enabled is False


def test_enabled_cache_reports_enabled(fake_redis):
    assert QueryCache(fake_redis).enabled is True


def test_disabled_get_returns_none():
    assert QueryCache(None).get("q", "store") is None


def test_disabled_set_is_a_noop():
    # Must not raise even though there is no client behind it.
    QueryCache(None).set("q", "store", {"answer": "x"})


# --- graceful degradation on Redis errors ---

def test_get_swallows_redis_errors(raising_redis):
    assert QueryCache(raising_redis).get("q", "store") is None


def test_set_swallows_redis_errors(raising_redis):
    # A write failure must not propagate.
    QueryCache(raising_redis).set("q", "store", {"answer": "x"})


def test_corrupt_entry_is_discarded(fake_redis):
    c = QueryCache(fake_redis)
    fake_redis.set(c._key("q", "store"), b"{not valid json")

    assert c.get("q", "store") is None


# --- from_env ---

def test_connect_pings_and_returns_the_client(fake_redis, monkeypatch):
    # Exercise the real _connect body offline: from_url yields the fake, whose
    # ping() succeeds, standing in for a reachable server.
    monkeypatch.setattr(
        cache_module.redis.Redis, "from_url", lambda *a, **k: fake_redis
    )

    assert cache_module._connect("redis://localhost:6379") is fake_redis


def test_from_env_connects_when_redis_is_reachable(fake_redis, monkeypatch):
    captured = {}

    def fake_connect(url):
        captured["url"] = url
        return fake_redis

    monkeypatch.setattr(cache_module, "_connect", fake_connect)

    c = QueryCache.from_env("redis://example:6379")

    assert c.enabled is True
    assert captured["url"] == "redis://example:6379"


def test_from_env_degrades_when_redis_is_unreachable(monkeypatch):
    def boom(url):
        raise ConnectionError("no server")

    monkeypatch.setattr(cache_module, "_connect", boom)

    c = QueryCache.from_env("redis://nowhere:6379")

    assert c.enabled is False


def test_from_env_uses_configured_url_by_default(fake_redis, monkeypatch):
    seen = {}

    def fake_connect(url):
        seen["url"] = url
        return fake_redis

    monkeypatch.setattr(cache_module, "_connect", fake_connect)
    monkeypatch.setenv("REDIS_URL", "redis://configured:6379")

    QueryCache.from_env()

    assert seen["url"] == "redis://configured:6379"


# --- redis_url helper ---

def test_redis_url_defaults_to_localhost(monkeypatch):
    monkeypatch.delenv("REDIS_URL", raising=False)

    assert redis_url() == DEFAULT_REDIS_URL


def test_redis_url_reads_environment(monkeypatch):
    monkeypatch.setenv("REDIS_URL", "redis://custom:6379")

    assert redis_url() == "redis://custom:6379"
