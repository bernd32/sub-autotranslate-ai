"""Separating non-linguistic markup from translatable text.

ASS dialogue text is a mix of actual speech and override blocks
(``{\\pos(400,120)\\blur0.6}``), line-break markers (``\\N``, ``\\h``) and
sometimes vector drawings (``{\\p1}m 0 0 l 10 0``). Sending all of that to
the model is expensive (on sign-heavy releases the tags outweigh the text)
and unreliable — the model is expected to copy coordinates byte-for-byte and
does not always manage.

`split_markup` therefore peels the markup off before translation:

* leading and trailing override blocks are removed entirely and re-attached
  afterwards, which covers the common ``{\\pos(...)}Text`` shape;
* override blocks *between* pieces of text become numbered placeholders
  (``<0/>``) that the model is told to keep in place;
* lines with no letters at all (``{op}``, pure drawings, ``\\h`` padding) are
  marked untranslatable and never sent.

`Markup.restore` is the exact inverse: ``split_markup(t).restore(payload)``
reproduces *t*, so a failed translation can always fall back to the original
line without corrupting the file.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from fnmatch import fnmatch

# An ASS override block. Blocks cannot nest, so a non-greedy scan to the next
# closing brace is correct; an unbalanced "{" is left as ordinary text.
_TAG_BLOCK_RE = re.compile(r"\{[^{}]*\}")

# {\p1} and friends switch the renderer into vector-drawing mode: everything
# after it is coordinates, not words. \p0 turns drawing back off.
_DRAWING_RE = re.compile(r"\\p\s*[1-9]")

_PLACEHOLDER_TEMPLATE = "<{}/>"
_PLACEHOLDER_RE = re.compile(r"<(\d+)/>")

# ASS text escapes: hard line break, soft line break, hard space.
_ASS_ESCAPE_RE = re.compile(r"\\[Nnh]")

# Markers that must survive translation verbatim; the counts are compared
# before and after to catch a model that "helpfully" reflows the line.
_COUNTED_MARKERS = (r"\N", r"\n", r"\h")

_HTML_TAG_RE = re.compile(r"</?([a-zA-Z][a-zA-Z0-9]*)[^>]*>")

# Inline formatting tags that subtitle formats actually use. Anything else in
# angle brackets is not markup: it may be a <speaker> marker the model echoed
# back, or plain text like "<3". Comparing those as tags rejects perfectly
# good translations.
_INLINE_TAGS = frozenset(
    {"i", "b", "u", "s", "font", "ruby", "rt", "rp", "c", "v", "lang"}
)


@dataclass(frozen=True)
class Markup:
    """The markup skeleton of one subtitle line plus its bare text."""

    prefix: str = ""
    suffix: str = ""
    inner: tuple[str, ...] = ()
    payload: str = ""
    translatable: bool = True
    reason: str = ""

    def restore(self, translated: str) -> str:
        """Put *translated* back into the original markup skeleton."""
        if not self.translatable:
            return self.prefix

        def _sub(match: re.Match[str]) -> str:
            idx = int(match.group(1))
            return self.inner[idx] if idx < len(self.inner) else ""

        body = _PLACEHOLDER_RE.sub(_sub, translated)
        # A placeholder the model dropped would silently lose its override
        # block; append any unused ones so styling is never lost outright.
        used = {int(m.group(1)) for m in _PLACEHOLDER_RE.finditer(translated)}
        missing = "".join(
            raw for i, raw in enumerate(self.inner) if i not in used
        )
        return f"{self.prefix}{body}{missing}{self.suffix}"


def has_letters(text: str) -> bool:
    """True if *text* contains a letter that belongs to actual language.

    The ASS escapes \\N, \\n and \\h are stripped first: their letters are
    markup, so a line padded with "\\h\\h" must not look translatable.
    """
    return any(ch.isalpha() for ch in _ASS_ESCAPE_RE.sub(" ", text))


def split_markup(text: str) -> Markup:
    """Split one subtitle line into markup skeleton + translatable payload."""
    parts: list[tuple[str, str]] = []
    pos = 0
    for match in _TAG_BLOCK_RE.finditer(text):
        if match.start() > pos:
            parts.append(("text", text[pos : match.start()]))
        parts.append(("tag", match.group(0)))
        pos = match.end()
    if pos < len(text):
        parts.append(("text", text[pos:]))

    if any(_DRAWING_RE.search(raw) for kind, raw in parts if kind == "tag"):
        return Markup(prefix=text, translatable=False, reason="vector drawing")

    # Only text carrying letters is worth a translation: "\h\h" or "123" are
    # positioning/padding, not language.
    content = [i for i, (kind, raw) in enumerate(parts) if kind == "text" and has_letters(raw)]
    if not content:
        return Markup(prefix=text, translatable=False, reason="no translatable text")

    first, last = content[0], content[-1]
    prefix = "".join(raw for _, raw in parts[:first])
    suffix = "".join(raw for _, raw in parts[last + 1 :])

    inner: list[str] = []
    payload_bits: list[str] = []
    for kind, raw in parts[first : last + 1]:
        if kind == "tag":
            payload_bits.append(_PLACEHOLDER_TEMPLATE.format(len(inner)))
            inner.append(raw)
        else:
            payload_bits.append(raw)

    return Markup(
        prefix=prefix,
        suffix=suffix,
        inner=tuple(inner),
        payload="".join(payload_bits),
    )


# ---------------------------------------------------------------------------
# Line kind
# ---------------------------------------------------------------------------

DIALOGUE = "dialogue"
SIGN = "sign"
SONG = "song"
SKIP = "skip"

# Matched against style names split into words, so "top" does not become a
# song because of "op" and "Main_Top" stays dialogue.
_SIGN_WORDS = frozenset(
    {
        "sign", "signs", "title", "titles", "caption", "captions",
        "note", "notes", "overlay", "overlays", "screen", "name", "names",
        "label", "board", "надпись", "надписи", "табличка", "название",
        "названия",
    }
)
_SONG_WORDS = frozenset(
    {
        "op", "ed", "oped", "song", "songs", "lyric", "lyrics", "insert",
        "karaoke", "kara", "romaji", "opening", "ending", "песня", "песни",
        "караоке",
    }
)

_WORD_SPLIT_RE = re.compile(r"[^0-9A-Za-z\u0400-\u04FF]+")

# Values seen in the Name field that describe the line, not a character.
_NON_SPEAKER_NAMES = frozenset(
    {"sign", "signs", "title", "note", "song", "op", "ed", "text", "caption"}
)


def _words(value: str) -> set[str]:
    return {w for w in _WORD_SPLIT_RE.split(value.lower()) if w}


def _matches_any(style: str, patterns: list[str]) -> bool:
    low = style.lower()
    return any(fnmatch(low, p.lower()) for p in patterns if p)


def classify_line(
    style: str,
    speaker: str,
    *,
    skip_styles: list[str] | None = None,
    sign_styles: list[str] | None = None,
    song_styles: list[str] | None = None,
) -> str:
    """Decide how a line should be treated, from its Style and Name fields.

    Signs, song lyrics and dialogue need different translation rules (a shop
    sign should not get a full sentence with final punctuation), and some
    styles should not be touched at all. Explicit config lists win over the
    name-based guess; both accept fnmatch globs.
    """
    if skip_styles and _matches_any(style, skip_styles):
        return SKIP
    if sign_styles and _matches_any(style, sign_styles):
        return SIGN
    if song_styles and _matches_any(style, song_styles):
        return SONG

    words = _words(style)
    if words & _SONG_WORDS:
        return SONG
    if words & _SIGN_WORDS:
        return SIGN
    if speaker.strip().lower() in _NON_SPEAKER_NAMES:
        return SIGN
    return DIALOGUE


def speaker_label(speaker: str) -> str:
    """The Name field, if it actually names a character."""
    value = speaker.strip()
    if not value or value.lower() in _NON_SPEAKER_NAMES:
        return ""
    if not has_letters(value):
        return ""
    return value


# ---------------------------------------------------------------------------
# Validation of a translated line
# ---------------------------------------------------------------------------

# Scripts the target language is expected to be written in; used only to
# catch a line the model left untranslated. Languages absent from the map
# are not script-checked.
_TARGET_SCRIPTS = {
    "russian": r"[\u0400-\u04FF]",
    "русский": r"[\u0400-\u04FF]",
    "ukrainian": r"[\u0400-\u04FF]",
    "belarusian": r"[\u0400-\u04FF]",
    "bulgarian": r"[\u0400-\u04FF]",
    "serbian": r"[\u0400-\u04FF]",
    "japanese": r"[\u3040-\u30FF\u4E00-\u9FFF]",
    "chinese": r"[\u4E00-\u9FFF]",
    "korean": r"[\uAC00-\uD7AF]",
    "greek": r"[\u0370-\u03FF]",
    "hebrew": r"[\u0590-\u05FF]",
    "arabic": r"[\u0600-\u06FF]",
}

_LEADING_NUMBER_RE = re.compile(r"^\s*\[\s*\d+\s*\]")

MAX_LENGTH_RATIO = 3.5
MIN_LENGTH_RATIO = 0.3
_RATIO_MIN_LENGTH = 12


def target_script_pattern(target_language: str) -> re.Pattern[str] | None:
    low = target_language.strip().lower()
    for name, pattern in _TARGET_SCRIPTS.items():
        if name in low:
            return re.compile(pattern)
    return None


def _marker_counts(text: str) -> dict[str, int]:
    return {m: text.count(m) for m in _COUNTED_MARKERS}


def _placeholders(text: str) -> list[int]:
    return sorted(int(m.group(1)) for m in _PLACEHOLDER_RE.finditer(text))


def _html_tags(text: str) -> list[str]:
    return sorted(
        m.group(0).lower()
        for m in _HTML_TAG_RE.finditer(text)
        if m.group(1).lower() in _INLINE_TAGS
    )


def validate_translation(
    source: str,
    translated: str,
    *,
    script: re.Pattern[str] | None = None,
) -> str | None:
    """Return a problem description, or None if the translation looks sane.

    Catches the failure modes that actually happen: dropped placeholders and
    line breaks, a line echoed back untranslated, response numbering leaking
    into the text, and wildly wrong lengths (a sign of the model answering
    the wrong item or adding commentary).
    """
    if not translated.strip():
        return "empty translation"

    if _LEADING_NUMBER_RE.match(translated):
        return "response numbering leaked into the text"

    if _placeholders(translated) != _placeholders(source):
        return "markup placeholders changed"

    src_markers, dst_markers = _marker_counts(source), _marker_counts(translated)
    if src_markers != dst_markers:
        changed = [m for m in _COUNTED_MARKERS if src_markers[m] != dst_markers[m]]
        return f"count of {', '.join(changed)} changed"

    if _html_tags(translated) != _html_tags(source):
        return "inline tags changed"

    if script is not None and has_letters(source) and not script.search(translated):
        return "not written in the target script"

    if len(source) >= _RATIO_MIN_LENGTH:
        ratio = len(translated) / len(source)
        if ratio > MAX_LENGTH_RATIO:
            return f"suspiciously long (x{ratio:.1f})"
        if ratio < MIN_LENGTH_RATIO:
            return f"suspiciously short (x{ratio:.1f})"

    return None
