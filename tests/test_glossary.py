"""Glossary candidate collection, response parsing and round-tripping."""

from __future__ import annotations

from pathlib import Path

from sub_autotranslate_tool.glossary import (
    Glossary,
    build_user_message,
    collect_candidates,
    parse_glossary_response,
)


def test_speaker_names_are_always_candidates() -> None:
    candidates = collect_candidates([], ["Tsukishiro", "Alice", "Tsukishiro", "sign", ""])
    assert candidates == ["Tsukishiro", "Alice"]


def test_recurring_mid_sentence_capitals_are_candidates() -> None:
    payloads = [
        "We should ask Hoshino about it.",
        "Hoshino is on the student council.",
        "The Committee will decide tomorrow.",
        "I spoke to the Committee already.",
    ]
    candidates = collect_candidates(payloads, [])
    assert "Hoshino" in candidates
    assert "Committee" in candidates


def test_sentence_initial_words_are_not_candidates() -> None:
    payloads = ["Morning already?", "Morning is the worst.", "Morning again."]
    assert "Morning" not in collect_candidates(payloads, [])


def test_words_after_a_line_break_are_sentence_initial() -> None:
    payloads = ["Hello\\NEveryone is here.", "Bye\\NEveryone left."]
    assert "Everyone" not in collect_candidates(payloads, [])


def test_single_occurrences_are_not_candidates() -> None:
    assert collect_candidates(["We met Kobayashi once."], []) == []


def test_common_capitalised_words_are_filtered() -> None:
    payloads = ["Well, I think so.", "Yes, I agree.", "Oh, I see."] * 2
    assert collect_candidates(payloads, []) == []


def test_candidate_limit_is_respected() -> None:
    assert len(collect_candidates([], [f"Name{i}" for i in range(80)], limit=10)) == 10


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


def test_parses_terms_and_genders() -> None:
    content = "Alice = Алиса | f\nBob = Боб | m\nStaff Room = Учительская | -"
    glossary = parse_glossary_response(content, ["Alice", "Bob", "Staff Room"])
    assert glossary.terms == {
        "Alice": "Алиса",
        "Bob": "Боб",
        "Staff Room": "Учительская",
    }
    assert glossary.genders == {"Alice": "f", "Bob": "m"}


def test_gender_is_optional() -> None:
    glossary = parse_glossary_response("Alice = Алиса", ["Alice"])
    assert glossary.terms == {"Alice": "Алиса"}
    assert glossary.genders == {}


def test_invented_terms_are_dropped() -> None:
    glossary = parse_glossary_response("Nobody = Никто", ["Alice"])
    assert glossary.terms == {}


def test_commentary_lines_are_ignored() -> None:
    content = "Here is the glossary:\n# a note\nAlice = Алиса | f\n\nHope this helps!"
    glossary = parse_glossary_response(content, ["Alice"])
    assert glossary.terms == {"Alice": "Алиса"}


def test_candidate_matching_is_case_insensitive_but_keeps_the_original() -> None:
    glossary = parse_glossary_response("alice = Алиса", ["Alice"])
    assert glossary.terms == {"Alice": "Алиса"}


# ---------------------------------------------------------------------------
# Prompt block and file format
# ---------------------------------------------------------------------------


def test_prompt_block_marks_gender() -> None:
    glossary = Glossary(terms={"Alice": "Алиса", "Gym": "Спортзал"}, genders={"Alice": "f"})
    assert glossary.as_prompt_block() == "Alice = Алиса (f)\nGym = Спортзал"


def test_empty_glossary_is_falsy_and_renders_nothing() -> None:
    assert not Glossary()
    assert Glossary().as_prompt_block() == ""


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    original = Glossary(
        terms={"Alice": "Алиса", 'The "Club"': "Клуб"}, genders={"Alice": "f"}
    )
    path = tmp_path / "glossary.toml"
    original.save(path)
    loaded = Glossary.load(path)
    assert loaded.terms == original.terms
    assert loaded.genders == original.genders


def test_user_message_lists_every_term() -> None:
    message = build_user_message(["Alice", "Bob"], "English", "Russian")
    assert "Alice" in message and "Bob" in message
    assert "Russian" in message and "English" in message
