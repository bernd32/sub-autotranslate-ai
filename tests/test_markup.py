"""Markup extraction, line classification and translation validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from sub_autotranslate_tool.markup import (
    DIALOGUE,
    SIGN,
    SKIP,
    SONG,
    classify_line,
    speaker_label,
    split_markup,
    target_script_pattern,
    validate_translation,
)
from sub_autotranslate_tool.subtitles import AssDocument, load_subtitles

# ---------------------------------------------------------------------------
# split_markup / restore
# ---------------------------------------------------------------------------

LINES = [
    "Plain spoken line.",
    "{\\pos(400,120)}Staff Room",
    "{\\pos(400,120)\\blur0.6}Notice\\NNo entry",
    "{\\i1}Is she late?{\\i0}",
    "Start{\\fs30}middle{\\b1}end",
    "{op}",
    "{\\an7\\pos(0,0)\\p1}m 0 0 l 10 0 l 10 10",
    "\\h\\h",
    "{\\fad(200,200)}Wait!{\\fscx110}",
    "no tags but a \\N break",
    "{unclosed tag text",
]


@pytest.mark.parametrize("line", LINES)
def test_restore_is_the_inverse_of_split(line: str) -> None:
    """The identity path must be exact: a failed translation falls back to it."""
    markup = split_markup(line)
    assert markup.restore(markup.payload) == line


def test_leading_and_trailing_tags_are_not_sent() -> None:
    markup = split_markup("{\\pos(400,120)\\blur0.6}Staff Room{\\fscx110}")
    assert markup.payload == "Staff Room"
    assert markup.prefix == "{\\pos(400,120)\\blur0.6}"
    assert markup.suffix == "{\\fscx110}"
    assert markup.inner == ()


def test_interior_tags_become_placeholders() -> None:
    markup = split_markup("Start{\\fs30}middle{\\b1}end")
    assert markup.payload == "Start<0/>middle<1/>end"
    assert markup.inner == ("{\\fs30}", "{\\b1}")
    assert markup.restore("Начало<0/>середина<1/>конец") == (
        "Начало{\\fs30}середина{\\b1}конец"
    )


def test_line_break_markers_stay_in_the_payload() -> None:
    markup = split_markup("{\\pos(1,2)}Notice\\NNo entry")
    assert markup.payload == "Notice\\NNo entry"


def test_vector_drawing_is_not_translatable() -> None:
    markup = split_markup("{\\an7\\pos(0,0)\\p1}m 0 0 l 10 0 l 10 10")
    assert not markup.translatable
    assert markup.reason == "vector drawing"


@pytest.mark.parametrize("line", ["{op}", "{part a}", "\\h\\h", "  ", "123", "?!"])
def test_lines_without_letters_are_not_translatable(line: str) -> None:
    assert not split_markup(line).translatable


def test_untranslatable_line_restores_unchanged() -> None:
    markup = split_markup("{op}")
    assert markup.restore("anything the model said") == "{op}"


def test_dropped_placeholder_does_not_lose_the_override_block() -> None:
    markup = split_markup("Start{\\fs30}end")
    # Model forgot <0/>; the block is appended rather than silently dropped.
    assert markup.restore("Начало конец") == "Начало конец{\\fs30}"


def test_markup_removal_shrinks_a_sign_heavy_line() -> None:
    line = (
        "{\\pos(261,296)\\blur0.6}Line one\\N{\\fs30}Line two"
        "\\N{\\fs45\\b1}Line three\\N{\\fs30\\b0}Line four"
    )
    markup = split_markup(line)
    # Interior tags still cost a placeholder each, so the win here is modest;
    # the big saving comes from positioning prefixes and drawings, measured
    # over a whole file in test_markup_reduces_payload_on_real_files.
    assert len(markup.payload) < len(line) * 0.7


def test_markup_reduces_payload_on_real_files(
    real_subtitle_files: list[Path],
) -> None:
    """The point of B1: measurably fewer characters reach the model."""
    measured = False
    for path in real_subtitle_files:
        if path.suffix.lower() not in (".ass", ".ssa"):
            continue
        segments = load_subtitles(path).segments
        raw = sum(len(s.text) for s in segments)
        sent = sum(
            len(m.payload)
            for m in (split_markup(s.text) for s in segments)
            if m.translatable
        )
        assert sent < raw, f"no reduction on {path.name}"
        measured = True
    if not measured:
        pytest.skip("no ASS files in test_files/")


# ---------------------------------------------------------------------------
# classify_line
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "style,speaker,expected",
    [
        ("Default", "Alice", DIALOGUE),
        ("Main", "Tsukishiro", DIALOGUE),
        ("top", "Alice", DIALOGUE),  # "top" must not match the song word "op"
        ("Main_Top", "Alice", DIALOGUE),
        ("Italics", "", DIALOGUE),
        ("signs", "sign", SIGN),
        ("ep title", "", SIGN),
        ("names", "", SIGN),
        ("OP Romaji", "", SONG),
        ("ED-Kanji", "", SONG),
        ("insert song", "", SONG),
        ("Default", "sign", SIGN),  # the Name field describes the line
    ],
)
def test_classify_line(style: str, speaker: str, expected: str) -> None:
    assert classify_line(style, speaker) == expected


def test_config_lists_override_the_guess() -> None:
    assert classify_line("Default", "", skip_styles=["default"]) == SKIP
    assert classify_line("Weird", "", sign_styles=["weird"]) == SIGN
    assert classify_line("Weird", "", song_styles=["weird"]) == SONG
    assert classify_line("signs", "", skip_styles=["sign*"]) == SKIP


def test_speaker_label_ignores_non_characters() -> None:
    assert speaker_label("Tsukishiro") == "Tsukishiro"
    assert speaker_label("  Alice  ") == "Alice"
    assert speaker_label("sign") == ""
    assert speaker_label("") == ""
    assert speaker_label("---") == ""


# ---------------------------------------------------------------------------
# validate_translation
# ---------------------------------------------------------------------------

CYRILLIC = target_script_pattern("Russian")


def test_good_translation_passes() -> None:
    assert validate_translation("Good morning.", "Доброе утро.", script=CYRILLIC) is None


def test_untranslated_line_is_rejected() -> None:
    assert validate_translation("Good morning.", "Good morning.", script=CYRILLIC)


def test_unknown_target_language_skips_the_script_check() -> None:
    assert target_script_pattern("Klingon") is None
    assert validate_translation("Good morning.", "Good morning.", script=None) is None


def test_dropped_placeholder_is_rejected() -> None:
    assert validate_translation("a<0/>b", "а б", script=CYRILLIC)


def test_dropped_line_break_is_rejected() -> None:
    assert validate_translation("Notice\\NNo entry", "Уведомление Вход запрещён", script=CYRILLIC)


def test_preserved_line_break_passes() -> None:
    assert (
        validate_translation("Notice\\NNo entry", "Уведомление\\NВход запрещён", script=CYRILLIC)
        is None
    )


def test_changed_html_tags_are_rejected() -> None:
    assert validate_translation("<i>Late?</i>", "Опаздывает?", script=CYRILLIC)
    assert (
        validate_translation("<i>Late?</i>", "<i>Опаздывает?</i>", script=CYRILLIC) is None
    )


def test_empty_and_numbered_answers_are_rejected() -> None:
    assert validate_translation("Hello", "   ", script=CYRILLIC)
    assert validate_translation("Hello there friend", "[3] Привет, друг", script=CYRILLIC)


def test_absurd_length_is_rejected() -> None:
    src = "We should get going now."
    assert validate_translation(src, "Да" * 60, script=CYRILLIC)
    assert validate_translation(src, "Да", script=CYRILLIC)


def test_short_lines_are_not_length_checked() -> None:
    # Russian expands a lot on interjections; no ratio check below 12 chars.
    assert validate_translation("Huh?", "Что происходит?", script=CYRILLIC) is None


# ---------------------------------------------------------------------------
# Integration with the parser
# ---------------------------------------------------------------------------


def test_segments_carry_style_and_speaker(sample_ass: str) -> None:
    doc = AssDocument(sample_ass)
    assert (doc.segments[0].style, doc.segments[0].speaker) == ("Default", "Alice")
    assert (doc.segments[2].style, doc.segments[2].speaker) == ("signs", "sign")
    assert doc.segments[6].style == "OP Romaji"


def test_ssa_v4_field_layout_is_read_from_the_format_line() -> None:
    text = (
        "[Events]\n"
        "Format: Marked, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
        "Marked=0,0:00:01.00,0:00:03.00,Default,Alice,0,0,0,,Hello, world\n"
        "Dialogue: Marked=0,0:00:03.00,0:00:05.00,Default,Bob,0,0,0,,Hi, there\n"
    )
    doc = AssDocument(text)
    assert [s.text for s in doc.segments] == ["Hi, there"]
    assert doc.segments[0].speaker == "Bob"
    assert doc.render() == text
