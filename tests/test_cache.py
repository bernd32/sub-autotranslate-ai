"""On-disk translation memory."""

from __future__ import annotations

from pathlib import Path

from sub_autotranslate_tool.cache import NullCache, TranslationCache

KEY_ARGS = dict(
    payload="Staff Room",
    model="deepseek/deepseek-v4.1-flash",
    source_language="English",
    target_language="Russian",
    kind="sign",
    speaker="",
    fingerprint="abc123",
)


def make_key(**overrides) -> str:
    return TranslationCache.key(**{**KEY_ARGS, **overrides})


def test_stores_and_returns_a_translation(tmp_path: Path) -> None:
    cache = TranslationCache(tmp_path / "tm.sqlite3")
    key = make_key()
    assert cache.get(key) is None
    cache.put_many([(key, "Staff Room", "Учительская", "model", "sign")])
    assert cache.get(key) == "Учительская"
    cache.close()


def test_survives_reopening(tmp_path: Path) -> None:
    path = tmp_path / "tm.sqlite3"
    key = make_key()
    first = TranslationCache(path)
    first.put_many([(key, "Staff Room", "Учительская", "model", "sign")])
    first.close()

    second = TranslationCache(path)
    assert second.get(key) == "Учительская"
    second.close()


def test_key_changes_with_anything_that_changes_the_output() -> None:
    base = make_key()
    for field, value in [
        ("payload", "Other text"),
        ("model", "other/model"),
        ("source_language", "Japanese"),
        ("target_language", "German"),
        ("kind", "dialogue"),
        ("speaker", "Alice"),
        ("fingerprint", "different"),
    ]:
        assert make_key(**{field: value}) != base, field


def test_identical_inputs_give_the_same_key() -> None:
    assert make_key() == make_key()


def test_clear_empties_the_store(tmp_path: Path) -> None:
    cache = TranslationCache(tmp_path / "tm.sqlite3")
    cache.put_many([(make_key(), "a", "б", "model", "sign")])
    assert cache.count() == 1
    assert cache.clear() == 1
    assert cache.count() == 0
    assert cache.get(make_key()) is None
    cache.close()


def test_unwritable_location_disables_the_cache_instead_of_failing(
    tmp_path: Path,
) -> None:
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("", encoding="utf-8")
    cache = TranslationCache(blocker / "sub" / "tm.sqlite3")
    assert not cache.enabled
    cache.put_many([(make_key(), "a", "б", "model", "sign")])
    assert cache.get(make_key()) is None


def test_null_cache_stores_nothing() -> None:
    cache = NullCache()
    assert not cache.enabled
    cache.put_many([(make_key(), "a", "б", "model", "sign")])
    assert cache.get(make_key()) is None
    assert cache.count() == 0
