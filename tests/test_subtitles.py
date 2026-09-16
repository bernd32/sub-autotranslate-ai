"""Parse/render contract for SRT and ASS documents.

The central guarantee these tests lock down: parsing a file and rendering it
back without touching any segment must reproduce the input exactly (after
line-ending normalisation), and replacing segment text must change *only*
the Text field of dialogue events. Everything in the tag-handling and
batching layers relies on that.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sub_autotranslate_tool.subtitles import (
    AssDocument,
    SrtDocument,
    SubtitleFormatError,
    load_subtitles,
    read_text,
)


def normalize(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------


def test_ass_roundtrip_is_lossless(sample_ass: str) -> None:
    doc = AssDocument(sample_ass)
    assert doc.render() == sample_ass


def test_srt_roundtrip_is_lossless(sample_srt: str) -> None:
    doc = SrtDocument(sample_srt)
    assert doc.render() == sample_srt


def test_roundtrip_on_real_files(real_subtitle_files: list[Path]) -> None:
    """parse -> render must not alter the real fansub files."""
    for path in real_subtitle_files:
        original = normalize(read_text(path))
        rendered = load_subtitles(path).render()
        assert rendered == original, f"round-trip changed {path.name}"


def test_roundtrip_via_tmp_file(tmp_path: Path, sample_ass: str) -> None:
    path = tmp_path / "sample.ass"
    path.write_text(sample_ass, encoding="utf-8")
    assert load_subtitles(path).render() == sample_ass


# ---------------------------------------------------------------------------
# ASS structure
# ---------------------------------------------------------------------------


def test_ass_collects_only_dialogue_events(sample_ass: str) -> None:
    doc = AssDocument(sample_ass)
    texts = [s.text for s in doc.segments]
    assert len(texts) == 9  # the Comment line is not translatable
    assert not any("must not be translated" in t for t in texts)


def test_ass_keeps_commas_inside_text_field(sample_ass: str) -> None:
    doc = AssDocument(sample_ass)
    assert doc.segments[0].text == "Good morning, everyone."


def test_ass_translated_text_may_contain_commas(sample_ass: str) -> None:
    doc = AssDocument(sample_ass)
    doc.segments[0].text = "Доброе утро, все вместе, разом."
    rendered = AssDocument(doc.render())
    assert rendered.segments[0].text == "Доброе утро, все вместе, разом."
    # Timing/style fields must be untouched.
    line = [ln for ln in doc.render().split("\n") if "Доброе утро" in ln][0]
    assert line.startswith("Dialogue: 0,0:00:01.00,0:00:03.00,Default,Alice,0,0,0,,")


def test_ass_replacing_all_text_preserves_header_and_styles(sample_ass: str) -> None:
    doc = AssDocument(sample_ass)
    for i, seg in enumerate(doc.segments):
        seg.text = f"строка {i}"
    out = doc.render()
    for section in ("[Script Info]", "[Aegisub Project Garbage]", "[V4+ Styles]"):
        assert section in out
    assert "Style: signs,Arial,36,&H00FFFFFF,0,0,8,1" in out
    assert "Title: synthetic, with a comma" in out
    assert out.count("Dialogue: ") == sample_ass.count("Dialogue: ")


def test_ass_dialogue_outside_events_section_is_ignored() -> None:
    text = (
        "[Script Info]\n"
        "Dialogue: 0,0:00:01.00,0:00:03.00,Default,,0,0,0,,not an event\n"
        "\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
        "Dialogue: 0,0:00:01.00,0:00:03.00,Default,,0,0,0,,real event\n"
    )
    doc = AssDocument(text)
    assert [s.text for s in doc.segments] == ["real event"]
    assert doc.render() == text


def test_ass_without_dialogue_raises() -> None:
    with pytest.raises(SubtitleFormatError):
        AssDocument("[Script Info]\nScriptType: v4.00+\n")


# ---------------------------------------------------------------------------
# SRT structure
# ---------------------------------------------------------------------------


def test_srt_multiline_cue_is_one_segment(sample_srt: str) -> None:
    doc = SrtDocument(sample_srt)
    assert doc.segments[2].text == "1. First point\n2. Second point"


def test_srt_replacing_text_preserves_timing(sample_srt: str) -> None:
    doc = SrtDocument(sample_srt)
    doc.segments[0].text = "Доброе утро."
    out = doc.render()
    assert "00:00:01,000 --> 00:00:03,000\nДоброе утро." in out


def test_srt_accepts_crlf_and_missing_index() -> None:
    text = "00:00:01,000 --> 00:00:03,000\r\nHello\r\n\r\n"
    doc = SrtDocument(text)
    assert [s.text for s in doc.segments] == ["Hello"]
    assert doc.render() == "1\n00:00:01,000 --> 00:00:03,000\nHello\n"


def test_srt_without_cues_raises() -> None:
    with pytest.raises(SubtitleFormatError):
        SrtDocument("just some text\nwithout timings\n")


# ---------------------------------------------------------------------------
# Loading / encodings
# ---------------------------------------------------------------------------


def test_load_subtitles_rejects_unknown_extension(tmp_path: Path) -> None:
    path = tmp_path / "subs.vtt"
    path.write_text("WEBVTT\n", encoding="utf-8")
    with pytest.raises(SubtitleFormatError):
        load_subtitles(path)


def test_read_text_strips_utf8_bom(tmp_path: Path) -> None:
    path = tmp_path / "bom.srt"
    path.write_bytes("﻿1\n00:00:01,000 --> 00:00:02,000\nHi\n".encode("utf-8"))
    assert read_text(path).startswith("1\n")


def test_read_text_decodes_cp1251(tmp_path: Path) -> None:
    path = tmp_path / "cp1251.srt"
    path.write_bytes("1\n00:00:01,000 --> 00:00:02,000\nПривет\n".encode("cp1251"))
    assert "Привет" in read_text(path)
