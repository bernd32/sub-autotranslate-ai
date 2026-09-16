"""The translation pipeline, driven against a stubbed OpenRouter transport."""

from __future__ import annotations

import re
import tomllib
from dataclasses import fields
from pathlib import Path

import pytest

from sub_autotranslate_tool.cache import TranslationCache
from sub_autotranslate_tool.config import DEFAULT_CONFIG_TEMPLATE, Config
from sub_autotranslate_tool.glossary import Glossary
from sub_autotranslate_tool.subtitles import AssDocument, Segment
from sub_autotranslate_tool.markup import target_script_pattern, validate_translation
from sub_autotranslate_tool.translator import (
    Translator,
    parse_numbered_response,
    render_template,
    strip_echoed_label,
)

# ---------------------------------------------------------------------------
# Stub transport
# ---------------------------------------------------------------------------

_MARKER_RE = re.compile(r"(<\d+/>|\\N|\\n|\\h|</?i>)")
_ITEM_RE = re.compile(r"^\[(\d+)\]\s*(?:<[^>]*>\s*)?(.*)$")


def _cyrillicize(text: str) -> str:
    """Same-length stand-in for a translation, so validation is exercised."""
    out = []
    for ch in text:
        if "a" <= ch <= "z":
            out.append(chr(0x430 + (ord(ch) - ord("a"))))
        elif "A" <= ch <= "Z":
            out.append(chr(0x410 + (ord(ch) - ord("A"))))
        else:
            out.append(ch)
    return "".join(out)


def fake_translate(text: str) -> str:
    return "".join(
        part if _MARKER_RE.fullmatch(part) else _cyrillicize(part)
        for part in _MARKER_RE.split(text)
    )


def items_of(payload: dict) -> dict[int, str]:
    """The numbered lines of a request, as the model would see them."""
    user = payload["messages"][-1]["content"]
    found = {}
    for line in user.split("\n"):
        match = _ITEM_RE.match(line)
        if match:
            found[int(match.group(1))] = match.group(2)
    return found


class FakeResponse:
    def __init__(self, data: dict, status_code: int = 200) -> None:
        self._data = data
        self.status_code = status_code
        self.text = str(data)

    def json(self) -> dict:
        return self._data


def completion(content: str | None, finish_reason: str = "stop", **usage) -> dict:
    return {
        "choices": [{"message": {"content": content}, "finish_reason": finish_reason}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 50, **usage},
    }


def echo(payload: dict) -> FakeResponse:
    """Translate every item the request asked for."""
    lines = [f"[{n}] {fake_translate(t)}" for n, t in sorted(items_of(payload).items())]
    return FakeResponse(completion("\n".join(lines)))


def echo_with_labels(payload: dict) -> FakeResponse:
    """Like `echo`, but repeats the <sign>/<song>/<speaker> marker.

    Real models do this routinely despite the prompt telling them not to.
    """
    lines = []
    for number, text in sorted(items_of(payload).items()):
        user = payload["messages"][-1]["content"]
        marker = re.search(rf"^\[{number}\]\s*(<[^>]*>)", user, re.M)
        prefix = f"{marker.group(1)} " if marker else ""
        lines.append(f"[{number}] {prefix}{fake_translate(text)}")
    return FakeResponse(completion("\n".join(lines)))


class FakeSession:
    def __init__(self, handler) -> None:
        self.handler = handler
        self.sent: list[dict] = []
        self.headers: dict = {}
        self.proxies: dict = {}

    def post(self, url, json=None, timeout=None):  # noqa: A002 - requests' API
        self.sent.append(json)
        result = self.handler(json)
        return result(json) if callable(result) else result


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_config(**overrides) -> Config:
    data = tomllib.loads(DEFAULT_CONFIG_TEMPLATE)
    known = {f.name for f in fields(Config)}
    config = Config(**{k: v for k, v in data.items() if k in known})
    config.retry_delay = 0.0
    config.glossary_auto = False
    config.cache = False
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def make_translator(handler=echo, config: Config | None = None, **kwargs) -> Translator:
    translator = Translator(config or make_config(), "test-key", **kwargs)
    translator.session = FakeSession(handler)
    return translator


def segs(*items) -> list[Segment]:
    out = []
    for item in items:
        if isinstance(item, str):
            out.append(Segment(text=item, style="Default", speaker=""))
        else:
            text, style, speaker = item
            out.append(Segment(text=text, style=style, speaker=speaker))
    return out


# ---------------------------------------------------------------------------
# parse_numbered_response  (C1)
# ---------------------------------------------------------------------------


def test_parses_bracketed_items() -> None:
    assert parse_numbered_response("[1] one\n[2] two", 2) == {1: "one", 2: "two"}


def test_parses_loose_numbering() -> None:
    assert parse_numbered_response("1. one\n2) two", 2) == {1: "one", 2: "two"}


def test_enumeration_inside_a_subtitle_is_not_a_new_item() -> None:
    """The C1 bug: a cue whose own text is a numbered list must stay intact."""
    content = "[1] Rules:\n1. First\n2. Second\n[2] Understood?"
    assert parse_numbered_response(content, 2) == {
        1: "Rules:\n1. First\n2. Second",
        2: "Understood?",
    }


def test_out_of_sequence_loose_number_is_treated_as_text() -> None:
    content = "[1] see item\n7) not an item"
    assert parse_numbered_response(content, 3) == {1: "see item\n7) not an item"}


def test_partial_response_returns_what_it_could_read() -> None:
    assert parse_numbered_response("[1] one\n[3] three", 3) == {1: "one", 3: "three"}


def test_numbers_beyond_the_batch_are_ignored() -> None:
    assert parse_numbered_response("[9] nine", 2) == {}


def test_multiline_item_keeps_its_line_breaks() -> None:
    assert parse_numbered_response("[1] first\nsecond", 1) == {1: "first\nsecond"}


# ---------------------------------------------------------------------------
# render_template  (C2)
# ---------------------------------------------------------------------------


def test_template_tolerates_ass_tags_in_a_custom_prompt() -> None:
    template = "Translate to {target_language}. Keep {\\i1} and {\\pos(1,2)} as is."
    out = render_template(template, {"target_language": "Russian"})
    assert out == "Translate to Russian. Keep {\\i1} and {\\pos(1,2)} as is."


def test_template_still_accepts_old_doubled_braces() -> None:
    out = render_template("keep {{\\i1}} intact", {})
    assert out == "keep {\\i1} intact"


def test_custom_prompt_with_braces_does_not_crash_a_run() -> None:
    config = make_config(prompt="To {target_language}. Keep {\\an8} as is.")
    translator = make_translator(config=config)
    result = translator.translate_segments(segs("Hello there, friend."))
    assert result[0] == fake_translate("Hello there, friend.")


# ---------------------------------------------------------------------------
# Prompt structure  (A1, A2, A3, B5)
# ---------------------------------------------------------------------------


def test_instructions_go_into_a_stable_system_message() -> None:
    config = make_config(batch_size=2, context_overlap=0)
    translator = make_translator(config=config)
    translator.translate_segments(segs("One line here.", "Two lines here.", "Three now."))

    systems = [p["messages"][0] for p in translator.session.sent]
    assert len(translator.session.sent) == 2
    assert all(m["role"] == "system" for m in systems)
    # Identical prefix across batches is what makes provider caching work.
    assert systems[0]["content"] == systems[1]["content"]
    assert "[N]" in systems[0]["content"]


def test_speaker_is_sent_with_each_line() -> None:
    translator = make_translator()
    translator.translate_segments(segs(("Good morning.", "Default", "Alice")))
    user = translator.session.sent[0]["messages"][-1]["content"]
    assert "<Alice> Good morning." in user


def test_speaker_can_be_switched_off() -> None:
    translator = make_translator(config=make_config(send_speaker=False))
    translator.translate_segments(segs(("Good morning.", "Default", "Alice")))
    user = translator.session.sent[0]["messages"][-1]["content"]
    assert "<Alice>" not in user


def test_signs_and_songs_are_labelled() -> None:
    translator = make_translator()
    translator.translate_segments(
        segs(("Staff Room", "signs", "sign"), ("A lyric line", "OP Romaji", ""))
    )
    user = translator.session.sent[0]["messages"][-1]["content"]
    assert "<sign> Staff Room" in user
    assert "<song> A lyric line" in user


def test_previous_batch_is_offered_as_context() -> None:
    config = make_config(batch_size=1, context_overlap=2)
    translator = make_translator(config=config)
    translator.translate_segments(segs("First line here.", "Second line here."))

    first, second = (p["messages"][-1]["content"] for p in translator.session.sent)
    assert "continuity" not in first
    assert "continuity" in second
    assert "First line here. ->" in second


def test_context_overlap_can_be_disabled() -> None:
    config = make_config(batch_size=1, context_overlap=0)
    translator = make_translator(config=config)
    translator.translate_segments(segs("First line here.", "Second line here."))
    assert "continuity" not in translator.session.sent[1]["messages"][-1]["content"]


def test_glossary_and_context_are_pinned_into_the_system_prompt() -> None:
    translator = make_translator(series_context="Alice is the class president.")
    translator.attach_glossary(Glossary(terms={"Alice": "Алиса"}, genders={"Alice": "f"}))
    translator.translate_segments(segs("Hello there."))
    system = translator.session.sent[0]["messages"][0]["content"]
    assert "Alice = Алиса (f)" in system
    assert "Alice is the class president." in system


def test_glossary_is_appended_when_a_custom_prompt_has_no_placeholder() -> None:
    """A prompt predating {glossary} must not silently discard it."""
    config = make_config(prompt="Translate to {target_language}.")
    translator = make_translator(config=config, series_context="Alice leads the club.")
    translator.attach_glossary(Glossary(terms={"Alice": "Алиса"}))
    translator.translate_segments(segs("Hello there."))
    system = translator.session.sent[0]["messages"][0]["content"]
    assert "Alice = Алиса" in system
    assert "Alice leads the club." in system


def test_glossary_is_not_duplicated_when_the_placeholder_exists() -> None:
    config = make_config(prompt="To {target_language}.{glossary}")
    translator = make_translator(config=config)
    translator.attach_glossary(Glossary(terms={"Alice": "Алиса"}))
    translator.translate_segments(segs("Hello there."))
    system = translator.session.sent[0]["messages"][0]["content"]
    assert system.count("Alice = Алиса") == 1


def test_anthropic_models_get_an_explicit_cache_breakpoint() -> None:
    translator = make_translator(config=make_config(model="anthropic/claude-opus-5"))
    translator.translate_segments(segs("Hello there."))
    system = translator.session.sent[0]["messages"][0]
    assert system["content"][0]["cache_control"] == {"type": "ephemeral"}


def test_other_models_get_a_plain_system_string() -> None:
    translator = make_translator()
    translator.translate_segments(segs("Hello there."))
    assert isinstance(translator.session.sent[0]["messages"][0]["content"], str)


# ---------------------------------------------------------------------------
# Not sending what needn't be sent  (B1, B2, B3)
# ---------------------------------------------------------------------------


def test_markup_is_not_sent_and_is_restored() -> None:
    translator = make_translator()
    result = translator.translate_segments(
        segs(("{\\pos(400,120)\\blur0.6}Staff Room", "signs", "sign"))
    )
    sent = items_of(translator.session.sent[0])
    assert sent == {1: "Staff Room"}
    assert result[0] == "{\\pos(400,120)\\blur0.6}" + fake_translate("Staff Room")


def test_untranslatable_lines_are_never_sent() -> None:
    translator = make_translator()
    result = translator.translate_segments(
        segs("{op}", ("{\\p1}m 0 0 l 10 0", "signs", ""), "Real dialogue here.")
    )
    assert items_of(translator.session.sent[0]) == {1: "Real dialogue here."}
    assert result[0] == "{op}"
    assert result[1] == "{\\p1}m 0 0 l 10 0"
    assert translator.stats.lines_skipped == 2


def test_skip_styles_are_copied_verbatim() -> None:
    config = make_config(skip_styles=["signs"])
    translator = make_translator(config=config)
    result = translator.translate_segments(
        segs(("Staff Room", "signs", ""), "Real dialogue here.")
    )
    assert items_of(translator.session.sent[0]) == {1: "Real dialogue here."}
    assert result[0] == "Staff Room"


def test_repeated_lines_are_translated_once() -> None:
    translator = make_translator()
    result = translator.translate_segments(segs(*[("Meeting Room", "signs", "")] * 5))
    assert items_of(translator.session.sent[0]) == {1: "Meeting Room"}
    assert len(set(result)) == 1
    assert translator.stats.lines_deduped == 4
    assert translator.stats.lines_sent == 1


def test_dedup_can_be_disabled() -> None:
    translator = make_translator(config=make_config(dedup=False))
    translator.translate_segments(segs(*["Meeting Room"] * 3))
    assert len(items_of(translator.session.sent[0])) == 3


def test_duplicates_with_different_speakers_stay_separate() -> None:
    translator = make_translator()
    translator.translate_segments(
        segs(("What?", "Default", "Alice"), ("What?", "Default", "Bob"))
    )
    assert len(items_of(translator.session.sent[0])) == 2


# ---------------------------------------------------------------------------
# Retry behaviour  (B6, B7, A11)
# ---------------------------------------------------------------------------


def test_only_missing_lines_are_requested_again() -> None:
    """B7: a partial answer must not cost a full second batch."""
    calls = []

    def handler(payload: dict) -> FakeResponse:
        items = items_of(payload)
        calls.append(sorted(items.values()))
        if len(calls) == 1:
            # Answer everything except item 2.
            lines = [
                f"[{n}] {fake_translate(t)}" for n, t in items.items() if n != 2
            ]
            return FakeResponse(completion("\n".join(lines)))
        return echo(payload)

    translator = make_translator(handler)
    result = translator.translate_segments(
        segs("Line one here.", "Line two here.", "Line three here.")
    )

    assert calls[1] == ["Line two here."]  # only the gap was re-sent
    assert result == [fake_translate(t) for t in
                      ("Line one here.", "Line two here.", "Line three here.")]
    assert translator.stats.repairs == 1


def test_echoed_marker_is_stripped_not_rejected() -> None:
    """Regression: echoing "<sign> " must not cost a re-request."""
    translator = make_translator(echo_with_labels)
    result = translator.translate_segments(
        segs(
            ("Staff Room", "signs", ""),
            ("A lyric line here", "OP Romaji", ""),
            ("Good morning, everyone.", "Default", "Alice"),
        )
    )
    assert result == [
        fake_translate("Staff Room"),
        fake_translate("A lyric line here"),
        fake_translate("Good morning, everyone."),
    ]
    assert translator.stats.lines_rejected == 0
    assert translator.stats.lines_failed == 0
    assert len(translator.session.sent) == 1


def test_marker_like_text_of_its_own_is_left_alone() -> None:
    assert strip_echoed_label("<sign> Учительская", "sign") == "Учительская"
    assert strip_echoed_label("<Alice> Привет", "Alice") == "Привет"
    assert strip_echoed_label("<SIGN>Учительская", "sign") == "Учительская"
    # A different marker, or none sent at all: leave the answer untouched.
    assert strip_echoed_label("<i>Привет</i>", "sign") == "<i>Привет</i>"
    assert strip_echoed_label("<song> Привет", "") == "<song> Привет"
    assert strip_echoed_label("<Bob> Привет", "Alice") == "<Bob> Привет"


def test_speaker_marker_does_not_count_as_inline_markup() -> None:
    """The bug behind the rejection storm: <song> is not an <i>-style tag."""
    script = target_script_pattern("Russian")
    assert validate_translation("This is serious!", "<song> Это серьёзно!", script=script) is None
    # Real inline tags are still compared.
    assert validate_translation("<i>Late?</i>", "Опаздывает?", script=script)


def test_repeated_validation_failures_stop_early() -> None:
    """A model answering the same wrong way must not be paid for 5 times."""
    def drops_line_breaks(payload: dict) -> FakeResponse:
        lines = [
            f"[{n}] {fake_translate(t).replace(chr(92) + 'N', ' ')}"
            for n, t in items_of(payload).items()
        ]
        return FakeResponse(completion("\n".join(lines)))

    config = make_config(max_retries=8)
    translator = make_translator(drops_line_breaks, config=config)
    result = translator.translate_segments(segs("First part\\Nsecond part here"))

    assert result[0] == "First part\\Nsecond part here"  # source kept
    assert translator.stats.lines_failed == 1
    assert len(translator.session.sent) == 3  # not max_retries + 1


def test_rejections_do_not_split_the_batch() -> None:
    """Halving re-sends every line; it only helps for truncated replies."""
    def always_untranslated(payload: dict) -> FakeResponse:
        return FakeResponse(
            completion("\n".join(f"[{n}] {t}" for n, t in items_of(payload).items()))
        )

    translator = make_translator(always_untranslated)
    lines = segs(*[f"Line number {i} of the batch." for i in range(8)])
    translator.translate_segments(lines)

    # Every request carries the full set: no halving, no re-sent halves.
    assert all(len(items_of(p)) == 8 for p in translator.session.sent)
    assert len(translator.session.sent) == 3


def test_invalid_translation_is_requested_again() -> None:
    def handler(payload: dict) -> FakeResponse:
        items = items_of(payload)
        if len(items) > 1:
            # Drop the \N from item 1: must be rejected by validation.
            lines = []
            for n, text in items.items():
                translated = fake_translate(text)
                if n == 1:
                    translated = translated.replace("\\N", " ")
                lines.append(f"[{n}] {translated}")
            return FakeResponse(completion("\n".join(lines)))
        return echo(payload)

    translator = make_translator(handler)
    result = translator.translate_segments(
        segs("First part\\Nsecond part", "Another line here.")
    )
    assert result[0] == fake_translate("First part\\Nsecond part")
    assert translator.stats.lines_rejected == 1


def test_a_line_that_never_validates_keeps_its_source_text() -> None:
    def handler(payload: dict) -> FakeResponse:
        items = items_of(payload)
        return FakeResponse(
            completion("\n".join(f"[{n}] {t}" for n, t in items.items()))
        )

    config = make_config(max_retries=1)
    translator = make_translator(handler, config=config)
    result = translator.translate_segments(segs("{\\i1}Good morning, everyone.{\\i0}"))

    # Untranslated English is rejected, and the original line is preserved
    # exactly rather than written out half-processed.
    assert result[0] == "{\\i1}Good morning, everyone.{\\i0}"
    assert translator.stats.lines_failed == 1


def test_billed_requests_are_counted_even_with_empty_content() -> None:
    """B6: an empty answer still costs money and must show up in the report."""
    calls = []

    def handler(payload: dict) -> FakeResponse:
        calls.append(payload)
        if len(calls) == 1:
            return FakeResponse(completion(None))
        return echo(payload)

    translator = make_translator(handler)
    translator.translate_segments(segs("Hello there, friend."))
    assert translator.stats.requests == 2
    assert translator.stats.prompt_tokens == 200


def test_http_errors_are_counted_separately() -> None:
    calls = []

    def handler(payload: dict) -> FakeResponse:
        calls.append(payload)
        if len(calls) == 1:
            return FakeResponse({"error": "rate limited"}, status_code=429)
        return echo(payload)

    translator = make_translator(handler)
    translator.translate_segments(segs("Hello there, friend."))
    assert translator.stats.requests == 1
    assert translator.stats.failed_requests == 1


def test_truncated_response_splits_the_batch() -> None:
    seen = []

    def handler(payload: dict) -> FakeResponse:
        items = items_of(payload)
        seen.append(len(items))
        if len(items) > 1:
            return FakeResponse(completion("[1] partial", finish_reason="length"))
        return echo(payload)

    translator = make_translator(handler)
    result = translator.translate_segments(segs("One line.", "Two lines.", "Three now."))
    assert seen[0] == 3 and max(seen[1:]) < 3
    assert all(r for r in result)
    assert translator.stats.requests == len(seen)


# ---------------------------------------------------------------------------
# Translation memory  (B4)
# ---------------------------------------------------------------------------


def test_cache_serves_a_second_run(tmp_path: Path) -> None:
    cache = TranslationCache(tmp_path / "tm.sqlite3")
    lines = segs("A repeated sign here.")

    first = make_translator(cache=cache)
    expected = first.translate_segments(lines)
    assert len(first.session.sent) == 1

    second = make_translator(cache=cache)
    assert second.translate_segments(lines) == expected
    assert second.session.sent == []
    assert second.stats.lines_cached == 1
    cache.close()


def test_changing_the_prompt_invalidates_the_cache(tmp_path: Path) -> None:
    cache = TranslationCache(tmp_path / "tm.sqlite3")
    lines = segs("A repeated sign here.")
    make_translator(cache=cache).translate_segments(lines)

    changed = make_translator(config=make_config(prompt="Different instructions."), cache=cache)
    changed.translate_segments(lines)
    assert len(changed.session.sent) == 1
    cache.close()


def test_failed_lines_do_not_enter_the_cache(tmp_path: Path) -> None:
    """A kept-as-is source line must not be served later as a translation."""
    cache = TranslationCache(tmp_path / "tm.sqlite3")

    def useless(payload: dict) -> FakeResponse:
        items = items_of(payload)
        return FakeResponse(
            completion("\n".join(f"[{n}] {t}" for n, t in items.items()))
        )

    first = make_translator(useless, config=make_config(max_retries=0), cache=cache)
    lines = segs("Good morning, everyone.")
    assert first.translate_segments(lines)[0] == "Good morning, everyone."
    assert first.stats.lines_failed == 1

    second = make_translator(cache=cache)
    assert second.translate_segments(lines)[0] == fake_translate("Good morning, everyone.")
    assert len(second.session.sent) == 1
    cache.close()


def test_a_legitimately_unchanged_line_is_still_cached(tmp_path: Path) -> None:
    """Same-script language pairs may leave a line as is; that is a result."""
    cache = TranslationCache(tmp_path / "tm.sqlite3")
    config = make_config(source_language="German", target_language="Dutch")

    def unchanged(payload: dict) -> FakeResponse:
        items = items_of(payload)
        return FakeResponse(
            completion("\n".join(f"[{n}] {t}" for n, t in items.items()))
        )

    lines = segs("Rotterdam Centraal")
    first = make_translator(unchanged, config=config, cache=cache)
    assert first.translate_segments(lines)[0] == "Rotterdam Centraal"
    assert first.stats.lines_failed == 0

    second = make_translator(unchanged, config=config, cache=cache)
    second.translate_segments(lines)
    assert second.session.sent == []
    assert second.stats.lines_cached == 1
    cache.close()


def test_cache_is_shared_across_episodes(tmp_path: Path) -> None:
    """The season-level win: identical signs cost nothing after episode one."""
    cache = TranslationCache(tmp_path / "tm.sqlite3")
    translator = make_translator(cache=cache)
    episode = segs(("Meeting Room", "signs", ""), "Unique dialogue one.")
    translator.translate_segments(episode)
    sent_before = len(translator.session.sent)

    next_episode = segs(("Meeting Room", "signs", ""), "Unique dialogue two.")
    translator.translate_segments(next_episode)
    assert items_of(translator.session.sent[sent_before]) == {1: "Unique dialogue two."}
    cache.close()


# ---------------------------------------------------------------------------
# End to end over a document
# ---------------------------------------------------------------------------


def test_whole_document_keeps_its_structure(sample_ass: str) -> None:
    doc = AssDocument(sample_ass)
    translator = make_translator()
    for seg, text in zip(doc.segments, translator.translate_segments(doc.segments)):
        seg.text = text
    out = doc.render()

    assert out.count("Dialogue: ") == sample_ass.count("Dialogue: ")
    assert "Comment: 0,0:00:17.00,0:00:19.00,Default,Bob,0,0,0,,this must not be translated" in out
    assert "{\\an7\\pos(0,0)\\p1}m 0 0 l 10 0 l 10 10" in out  # drawing untouched
    assert "{top}" in out
    # Re-parsing the result must still yield the same number of events.
    assert len(AssDocument(out).segments) == len(doc.segments)
