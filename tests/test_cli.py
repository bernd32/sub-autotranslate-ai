"""End-to-end CLI runs with the network stubbed out."""

from __future__ import annotations

from pathlib import Path

import pytest

from sub_autotranslate_tool import cache as cache_module
from sub_autotranslate_tool import cli
from sub_autotranslate_tool import translator as translator_module
from sub_autotranslate_tool.config import DEFAULT_CONFIG_TEMPLATE
from sub_autotranslate_tool.glossary import Glossary
from sub_autotranslate_tool.subtitles import AssDocument

from .test_translator import FakeResponse, FakeSession, completion, echo, items_of


@pytest.fixture
def config_file(tmp_path: Path) -> Path:
    path = tmp_path / "config.toml"
    path.write_text(DEFAULT_CONFIG_TEMPLATE, encoding="utf-8")
    return path


@pytest.fixture
def sessions(monkeypatch: pytest.MonkeyPatch) -> list[FakeSession]:
    """Collect every session the run creates, so requests can be inspected."""
    created: list[FakeSession] = []

    def factory() -> FakeSession:
        session = FakeSession(echo)
        created.append(session)
        return session

    monkeypatch.setattr(translator_module.requests, "Session", factory)
    return created


def run(*args: str) -> int:
    return cli.main(list(args))


def test_translates_a_file_end_to_end(
    tmp_path: Path, config_file: Path, sessions: list[FakeSession], sample_ass: str
) -> None:
    src = tmp_path / "episode.ass"
    src.write_text(sample_ass, encoding="utf-8")

    code = run(
        str(src), "--config", str(config_file), "--api-key", "k",
        "--no-cache", "--no-glossary",
    )
    assert code == 0

    out = tmp_path / "episode.ru.ass"
    assert out.is_file()
    rendered = out.read_text(encoding="utf-8")
    # Structure preserved, dialogue changed, markup-only lines untouched.
    assert rendered.count("Dialogue: ") == sample_ass.count("Dialogue: ")
    assert "[V4+ Styles]" in rendered
    assert "{top}" in rendered
    assert "{\\an7\\pos(0,0)\\p1}m 0 0 l 10 0 l 10 10" in rendered
    assert "Good morning, everyone." not in rendered
    assert len(AssDocument(rendered).segments) == len(AssDocument(sample_ass).segments)


def test_directory_input_reuses_one_translator(
    tmp_path: Path, config_file: Path, sessions: list[FakeSession], sample_ass: str
) -> None:
    for name in ("ep01.ass", "ep02.ass"):
        (tmp_path / name).write_text(sample_ass, encoding="utf-8")
    out_dir = tmp_path / "out"

    assert run(
        str(tmp_path), "-o", str(out_dir), "--config", str(config_file),
        "--api-key", "k", "--no-cache", "--no-glossary",
    ) == 0
    assert (out_dir / "ep01.ru.ass").is_file()
    assert (out_dir / "ep02.ru.ass").is_file()
    # One session for the whole run, so stats and context carry across files.
    assert len(sessions) == 1


def test_unreadable_file_does_not_abort_the_run(
    tmp_path: Path, config_file: Path, sessions: list[FakeSession], sample_ass: str
) -> None:
    (tmp_path / "good.ass").write_text(sample_ass, encoding="utf-8")
    (tmp_path / "bad.ass").write_text("[Script Info]\nnothing here\n", encoding="utf-8")
    out_dir = tmp_path / "out"

    code = run(
        str(tmp_path), "-o", str(out_dir), "--config", str(config_file),
        "--api-key", "k", "--no-cache", "--no-glossary",
    )
    assert code == 1  # reported as a failure
    assert (out_dir / "good.ru.ass").is_file()  # but the good file was done


def test_series_context_is_picked_up_from_the_directory(
    tmp_path: Path, config_file: Path, sessions: list[FakeSession], sample_ass: str
) -> None:
    (tmp_path / "ep01.ass").write_text(sample_ass, encoding="utf-8")
    (tmp_path / ".subctx.txt").write_text(
        "Alice is the class president.", encoding="utf-8"
    )

    assert run(
        str(tmp_path), "--config", str(config_file), "--api-key", "k",
        "--no-cache", "--no-glossary",
    ) == 0
    system = sessions[0].sent[0]["messages"][0]["content"]
    assert "Alice is the class president." in system


def test_explicit_context_file_must_exist(
    tmp_path: Path, config_file: Path, sessions: list[FakeSession], sample_ass: str
) -> None:
    (tmp_path / "ep01.ass").write_text(sample_ass, encoding="utf-8")
    with pytest.raises(SystemExit):
        run(
            str(tmp_path), "--config", str(config_file), "--api-key", "k",
            "--context", str(tmp_path / "missing.txt"),
        )


def test_glossary_is_written_then_reused(
    tmp_path: Path, config_file: Path, monkeypatch: pytest.MonkeyPatch, sample_ass: str
) -> None:
    glossary_path = tmp_path / "glossary.toml"
    src = tmp_path / "ep01.ass"
    src.write_text(sample_ass, encoding="utf-8")

    calls: list[dict] = []

    def handler(payload: dict) -> FakeResponse:
        calls.append(payload)
        user = payload["messages"][-1]["content"]
        if "glossaries" in payload["messages"][0]["content"]:
            assert "Alice" in user
            return FakeResponse(completion("Alice = Алиса | f\nBob = Боб | m"))
        return echo(payload)

    monkeypatch.setattr(
        translator_module.requests, "Session", lambda: FakeSession(handler)
    )

    assert run(
        str(src), "--config", str(config_file), "--api-key", "k",
        "--no-cache", "--glossary", str(glossary_path),
    ) == 0

    saved = Glossary.load(glossary_path)
    assert saved.terms == {"Alice": "Алиса", "Bob": "Боб"}
    assert "Alice = Алиса (f)" in calls[-1]["messages"][0]["content"]

    # Second run: the glossary comes off disk, no pre-pass request.
    calls.clear()
    assert run(
        str(src), "--config", str(config_file), "--api-key", "k",
        "--no-cache", "--glossary", str(glossary_path),
    ) == 0
    assert all("glossaries" not in c["messages"][0]["content"] for c in calls)


def test_cache_reuse_between_runs(
    tmp_path: Path, config_file: Path, sessions: list[FakeSession],
    monkeypatch: pytest.MonkeyPatch, sample_ass: str
) -> None:
    # DEFAULT_CACHE_PATH is resolved at import time, so patch the constant
    # rather than XDG_CACHE_HOME: a test must never touch the real cache.
    monkeypatch.setattr(
        cache_module, "DEFAULT_CACHE_PATH", tmp_path / "cache" / "tm.sqlite3"
    )
    src = tmp_path / "ep01.ass"
    src.write_text(sample_ass, encoding="utf-8")
    args = (str(src), "--config", str(config_file), "--api-key", "k", "--no-glossary")

    assert run(*args) == 0
    first_requests = len(sessions[0].sent)
    assert first_requests > 0

    assert run(*args) == 0
    assert sessions[1].sent == []  # everything served from the memory


def test_missing_api_key_is_reported(tmp_path: Path, config_file: Path, sample_ass: str,
                                     monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    (tmp_path / "ep01.ass").write_text(sample_ass, encoding="utf-8")
    with pytest.raises(SystemExit):
        run(str(tmp_path / "ep01.ass"), "--config", str(config_file))


def test_unsupported_input_is_rejected(tmp_path: Path, config_file: Path) -> None:
    path = tmp_path / "notes.txt"
    path.write_text("hello", encoding="utf-8")
    with pytest.raises(SystemExit):
        run(str(path), "--config", str(config_file), "--api-key", "k")
