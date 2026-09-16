"""Glossary of names and recurring terms.

Batches are translated independently, so without a shared glossary the same
character name or faction can come out three different ways in one episode.
One cheap pre-pass collects the proper nouns from the whole run, asks the
model for a canonical rendering of each (plus grammatical gender, which many
target languages need for past-tense verbs), and the result is pinned into
the system prompt of every batch.

The glossary is a plain TOML file, so it can be hand-corrected once and
reused for a whole season via --glossary.
"""

from __future__ import annotations

import logging
import re
import tomllib
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from .markup import speaker_label

logger = logging.getLogger(__name__)

GLOSSARY_SYSTEM_PROMPT = (
    "You build glossaries for subtitle translators. You answer with nothing "
    "but the requested lines."
)

GLOSSARY_USER_TEMPLATE = """\
Below are proper nouns and recurring terms taken from {source_language} \
subtitles of an animated series. Give the canonical {target_language} \
rendering for each, to be used consistently throughout the translation.

Rules:
- One line per term, in the exact format: Source = Target | g
- "g" is the grammatical gender of a person's name in {target_language}: \
m (male), f (female), or - for anything that is not a person.
- Keep the Source side exactly as given.
- Use the rendering established in {target_language} for the series or genre \
where one exists; otherwise transcribe by pronunciation.
- If a term should stay untranslated, repeat it on the Target side.
- No comments, no numbering, no extra lines.

Terms:
{terms}"""

_TERM_LINE_RE = re.compile(r"^\s*(.+?)\s*=\s*(.+?)\s*(?:\|\s*([mfMF-])\s*)?$")

# Frequent capitalised words that are not names. The "not at the start of a
# sentence" rule catches most of them; these are the ones that survive it.
_STOPWORDS = frozenset(
    {
        "i", "i'm", "i'll", "i've", "i'd", "mr", "mrs", "ms", "miss", "sir",
        "madam", "god", "ok", "okay", "oh", "ah", "eh", "huh", "hey", "yeah",
        "yes", "no", "well", "but", "and", "the", "a", "an", "so", "then",
        "monday", "tuesday", "wednesday", "thursday", "friday", "saturday",
        "sunday", "january", "february", "march", "april", "may", "june",
        "july", "august", "september", "october", "november", "december",
    }
)

_WORD_RE = re.compile(r"[A-Z][a-zA-Z'’\-]{2,}")
# Characters after which a capital letter is simply the start of a sentence.
_SENTENCE_END = set(".!?…:;\"'“”«»()[]-–—*")


@dataclass
class Glossary:
    terms: dict[str, str] = field(default_factory=dict)
    genders: dict[str, str] = field(default_factory=dict)

    def __bool__(self) -> bool:
        return bool(self.terms)

    def as_prompt_block(self) -> str:
        """The glossary as it is pinned into the system prompt."""
        if not self.terms:
            return ""
        rows = []
        for source, target in sorted(self.terms.items()):
            gender = self.genders.get(source, "")
            suffix = f" ({gender})" if gender in ("m", "f") else ""
            rows.append(f"{source} = {target}{suffix}")
        return "\n".join(rows)

    def save(self, path: Path) -> None:
        lines = [
            "# Glossary for sub-autotranslate-tool.",
            "# Hand-edit freely: these renderings are pinned for every batch.",
            "",
            "[terms]",
        ]
        lines += [
            f'{_toml_key(source)} = {_toml_str(target)}'
            for source, target in sorted(self.terms.items())
        ]
        known = {k: v for k, v in self.genders.items() if v in ("m", "f")}
        if known:
            lines += ["", "# Grammatical gender of character names: m / f.", "[genders]"]
            lines += [
                f"{_toml_key(source)} = {_toml_str(gender)}"
                for source, gender in sorted(known.items())
            ]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> Glossary:
        with path.open("rb") as fh:
            data = tomllib.load(fh)
        terms = {str(k): str(v) for k, v in (data.get("terms") or {}).items()}
        genders = {str(k): str(v) for k, v in (data.get("genders") or {}).items()}
        return cls(terms=terms, genders=genders)


def _toml_key(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


_toml_str = _toml_key


def collect_candidates(
    payloads: list[str], speakers: list[str], limit: int = 60
) -> list[str]:
    """Proper nouns worth pinning, most frequent first.

    Speaker names from the ASS Name field are the most reliable source and
    are always included; capitalised words from the text need to appear more
    than once and away from a sentence start to qualify.
    """
    candidates: list[str] = []
    seen: set[str] = set()

    for name in speakers:
        label = speaker_label(name)
        if label and label.lower() not in seen:
            seen.add(label.lower())
            candidates.append(label)

    # A proper noun has to recur (total >= 2) and has to appear at least once
    # away from a sentence start: that is what separates a name from an
    # ordinary word that happens to open sentences ("Morning again.").
    total: Counter[str] = Counter()
    mid_sentence: Counter[str] = Counter()
    for payload in payloads:
        for match in _WORD_RE.finditer(payload):
            word = match.group(0)
            if word.lower() in _STOPWORDS:
                continue
            total[word] += 1
            if not _is_sentence_start(payload, match.start()):
                mid_sentence[word] += 1

    for word, count in total.most_common():
        if count < 2 or not mid_sentence[word] or word.lower() in seen:
            continue
        seen.add(word.lower())
        candidates.append(word)
        if len(candidates) >= limit:
            break

    return candidates[:limit]


def _is_sentence_start(text: str, index: int) -> bool:
    prefix = text[:index]
    if prefix.endswith("\\N") or prefix.endswith("\\n"):
        return True
    stripped = prefix.rstrip()
    if not stripped:
        return True
    return stripped[-1] in _SENTENCE_END


def parse_glossary_response(content: str, candidates: list[str]) -> Glossary:
    """Parse "Source = Target | g" lines, keeping only known candidates."""
    by_lower = {c.lower(): c for c in candidates}
    glossary = Glossary()
    for raw_line in content.split("\n"):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = _TERM_LINE_RE.match(line)
        if not match:
            continue
        source, target, gender = match.group(1), match.group(2), match.group(3)
        canonical = by_lower.get(source.strip().lower())
        if canonical is None or not target.strip():
            continue
        glossary.terms[canonical] = target.strip()
        if gender and gender.lower() in ("m", "f"):
            glossary.genders[canonical] = gender.lower()
    return glossary


def build_user_message(
    candidates: list[str], source_language: str, target_language: str
) -> str:
    return GLOSSARY_USER_TEMPLATE.format(
        source_language=source_language,
        target_language=target_language,
        terms="\n".join(candidates),
    )
