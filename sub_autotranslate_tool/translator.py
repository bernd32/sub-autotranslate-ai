"""LLM translation backend (OpenRouter).

The pipeline for one document:

1. Every segment is classified (dialogue / sign / song / skip) and split into
   a markup skeleton plus bare text, so override tags and drawings never
   reach the model (see `markup`).
2. Identical texts collapse into a single translation unit, and units already
   present in the on-disk translation memory are filled from there. Only what
   is left is actually sent.
3. Units go out in numbered batches together with the speaker of each line,
   the glossary, the series context and a few already-translated lines for
   continuity across the batch boundary.
4. Every answer is validated (markup preserved, target script, plausible
   length). Lines that fail are re-requested *on their own* rather than by
   re-sending the whole batch; a batch is only split when nothing at all came
   back usable. A line that never validates keeps its source text and is
   reported.

The static instructions live in a separate system message so that providers
can cache that prefix across the many batches of a run.
"""

from __future__ import annotations

import logging
import re
import sys
import time
from dataclasses import dataclass, field

import requests

from .cache import NullCache, TranslationCache, make_fingerprint
from .config import Config
from .glossary import (
    GLOSSARY_SYSTEM_PROMPT,
    Glossary,
    build_user_message,
    collect_candidates,
    parse_glossary_response,
)
from .markup import (
    SIGN,
    SKIP,
    SONG,
    Markup,
    classify_line,
    speaker_label,
    split_markup,
    target_script_pattern,
    validate_translation,
)
from .subtitles import Segment

API_URL = "https://openrouter.ai/api/v1/chat/completions"

# How often a batch may be halved when a response is unusable as a whole.
_MAX_SPLIT_DEPTH = 4

# How many rounds a set of lines may be re-requested purely because their
# translations failed validation. A model that answers the same wrong way
# twice will keep doing so, and every extra round is billed.
_MAX_VALIDATION_ROUNDS = 3

logger = logging.getLogger(__name__)


class TranslationError(RuntimeError):
    pass


class ResponseTruncated(TranslationError):
    """The model hit max_tokens: the batch is too large, retrying won't help."""


@dataclass
class UsageStats:
    requests: int = 0
    failed_requests: int = 0  # not billed (network/HTTP errors)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost: float = 0.0  # USD, taken from OpenRouter's usage.cost field
    lines_total: int = 0
    lines_sent: int = 0  # unique lines actually given to the model
    lines_cached: int = 0
    lines_deduped: int = 0
    lines_skipped: int = 0  # markup-only lines and skipped styles
    lines_failed: int = 0  # kept their source text
    lines_rejected: int = 0  # failed validation at least once
    repairs: int = 0  # follow-up requests for individual lines


@dataclass
class _Unit:
    """One unique text to translate, shared by all segments that contain it."""

    payload: str
    kind: str
    speaker: str
    key: str
    segments: list[int] = field(default_factory=list)
    translation: str | None = None
    # True when the source text was kept because translation never succeeded;
    # such a line must not enter the cache as if it were a translation.
    failed: bool = False


@dataclass
class _Line:
    """A document segment prepared for translation."""

    markup: Markup
    kind: str
    speaker: str


# A metadata marker at the very start of an answer, e.g. "<sign> ".
_LEADING_MARKER_RE = re.compile(r"^\s*<([^<>]{1,60})>[ \t]*")


def strip_echoed_label(answer: str, label: str) -> str:
    """Remove a metadata marker the model echoed back into its answer.

    The prompt asks the model not to repeat the <speaker>/<sign>/<song>
    marker, but many models do it anyway. Rejecting those answers throws away
    a perfectly good translation and pays for a re-request, so the marker we
    sent is simply stripped. Only that exact marker is removed: a line whose
    own text begins with something in angle brackets is left alone.
    """
    if not label:
        return answer
    match = _LEADING_MARKER_RE.match(answer)
    if match and match.group(1).strip().casefold() == label.strip().casefold():
        return answer[match.end() :]
    return answer


def render_template(template: str, values: dict[str, str]) -> str:
    """Substitute {placeholders} without str.format's brace rules.

    A user's prompt legitimately contains ASS override tags such as {\\i1};
    str.format chokes on those, so substitution is done by plain replacement.
    Doubled braces are still collapsed, so prompts written for the old
    format-based rendering keep working.
    """
    out = template
    for key, value in values.items():
        out = out.replace("{" + key + "}", value)
    return out.replace("{{", "{").replace("}}", "}")


class Translator:
    def __init__(
        self,
        config: Config,
        api_key: str,
        cache: TranslationCache | None = None,
        series_context: str = "",
    ) -> None:
        self.config = config
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/sub-autotranslate-tool",
                "X-Title": "sub-autotranslate-tool",
            }
        )
        if config.proxy:
            # Applies to both http and https (and socks5:// schemes).
            self.session.proxies.update(
                {"http": config.proxy, "https": config.proxy}
            )
        self.stats = UsageStats()
        self.cache = cache if cache is not None else NullCache()
        self.series_context = series_context.strip()
        self.glossary = Glossary()
        self._script = target_script_pattern(config.target_language)
        self._recent: list[tuple[str, str]] = []
        self._refresh_prompt()

    # ------------------------------------------------------------------
    # Prompt assembly
    # ------------------------------------------------------------------

    def attach_glossary(self, glossary: Glossary) -> None:
        """Pin a glossary into the system prompt of every later request."""
        self.glossary = glossary
        self._refresh_prompt()

    def _refresh_prompt(self) -> None:
        glossary_block = ""
        if self.glossary:
            glossary_block = (
                "\n\nGlossary — use exactly these renderings, every time:\n"
                + self.glossary.as_prompt_block()
            )
        context_block = ""
        if self.series_context:
            context_block = (
                "\n\nAbout this series (background for you, never translate "
                "it):\n" + self.series_context
            )
        template = self.config.prompt
        self._system_prompt = render_template(
            template,
            {
                "source_language": self.config.source_language,
                "target_language": self.config.target_language,
                "glossary": glossary_block,
                "context": context_block,
                # Batch size belongs in the per-batch user message; kept here
                # only so that older prompts containing {count} still render.
                "count": "the listed",
            },
        )
        # A custom prompt written before {glossary}/{context} existed would
        # otherwise drop them silently — after the glossary pre-pass has
        # already been paid for. Append instead, and say so.
        for name, block in (("glossary", glossary_block), ("context", context_block)):
            if block and "{" + name + "}" not in template:
                logger.warning(
                    "the configured prompt has no {%s} placeholder; "
                    "appending the %s at the end of the system prompt",
                    name,
                    name,
                )
                self._system_prompt += block
        # Cached translations must not outlive the instructions that produced
        # them, so the prompt is part of the cache key.
        self._fingerprint = make_fingerprint(self._system_prompt)

    def _system_message(self) -> dict:
        mode = (self.config.prompt_cache or "auto").strip().lower()
        explicit = mode == "on" or (
            mode == "auto" and self.config.model.startswith("anthropic/")
        )
        if explicit:
            # Anthropic-style models only cache a prefix that is marked as
            # such; most other providers cache long prefixes automatically.
            return {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": self._system_prompt,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            }
        return {"role": "system", "content": self._system_prompt}

    # ------------------------------------------------------------------
    # Glossary pre-pass
    # ------------------------------------------------------------------

    def build_glossary(self, segments: list[Segment]) -> Glossary:
        """Ask the model for canonical renderings of the recurring names."""
        payloads: list[str] = []
        speakers: list[str] = []
        for seg in segments:
            markup = split_markup(seg.text)
            if markup.translatable:
                payloads.append(markup.payload)
            speakers.append(seg.speaker)

        candidates = collect_candidates(payloads, speakers)
        if not candidates:
            logger.info("no glossary candidates found")
            return Glossary()

        logger.info("building glossary for %d term(s)", len(candidates))
        user = build_user_message(
            candidates, self.config.source_language, self.config.target_language
        )
        try:
            content = self.raw_completion(GLOSSARY_SYSTEM_PROMPT, user)
        except TranslationError as exc:
            logger.warning("glossary pre-pass failed, continuing without it: %s", exc)
            return Glossary()

        glossary = parse_glossary_response(content, candidates)
        logger.info("glossary: %d/%d term(s) resolved", len(glossary.terms), len(candidates))
        return glossary

    def raw_completion(self, system: str, user: str) -> str:
        """One request with no numbering contract, for the glossary pre-pass."""
        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "reasoning": {"enabled": self.config.enable_reasoning},
        }
        return self._post(payload)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def translate_segments(
        self, segments: list[Segment], progress_label: str = ""
    ) -> list[str]:
        """Translate document segments, returning the new text of each."""
        self._recent = []
        lines = [self._prepare(seg) for seg in segments]
        self.stats.lines_total += len(lines)

        units, by_segment = self._collect_units(lines)
        pending = self._fill_from_cache(units)
        self.stats.lines_sent += len(pending)

        batch_size = max(1, self.config.batch_size)
        done = 0
        for start in range(0, len(pending), batch_size):
            batch = pending[start : start + batch_size]
            self._translate_units(batch)
            self._store(batch)
            self._remember(batch)
            done += len(batch)
            print(
                f"\r{progress_label}{done}/{len(pending)} lines translated",
                end="",
                file=sys.stderr,
                flush=True,
            )
        if pending:
            print(file=sys.stderr)

        results = [seg.text for seg in segments]
        for idx, unit in by_segment.items():
            if unit.translation is not None:
                results[idx] = lines[idx].markup.restore(unit.translation)
        return results

    # ------------------------------------------------------------------
    # Preparation / deduplication / cache
    # ------------------------------------------------------------------

    def _prepare(self, seg: Segment) -> _Line:
        kind = classify_line(
            seg.style,
            seg.speaker,
            skip_styles=self.config.skip_styles,
            sign_styles=self.config.sign_styles,
            song_styles=self.config.song_styles,
        )
        speaker = ""
        if self.config.send_speaker and kind not in (SIGN, SONG):
            speaker = speaker_label(seg.speaker)
        return _Line(markup=split_markup(seg.text), kind=kind, speaker=speaker)

    def _collect_units(
        self, lines: list[_Line]
    ) -> tuple[list[_Unit], dict[int, _Unit]]:
        units: dict[str, _Unit] = {}
        order: list[_Unit] = []
        by_segment: dict[int, _Unit] = {}

        for idx, line in enumerate(lines):
            if line.kind == SKIP or not line.markup.translatable:
                self.stats.lines_skipped += 1
                logger.debug(
                    "not translating line %d (%s): %r",
                    idx + 1,
                    "skipped style" if line.kind == SKIP else line.markup.reason,
                    line.markup.prefix[:60],
                )
                continue

            key = self.cache.key(
                payload=line.markup.payload,
                model=self.config.model,
                source_language=self.config.source_language,
                target_language=self.config.target_language,
                kind=line.kind,
                speaker=line.speaker,
                fingerprint=self._fingerprint,
            )
            group = key if self.config.dedup else f"{idx}:{key}"
            unit = units.get(group)
            if unit is None:
                unit = _Unit(
                    payload=line.markup.payload,
                    kind=line.kind,
                    speaker=line.speaker,
                    key=key,
                )
                units[group] = unit
                order.append(unit)
            else:
                self.stats.lines_deduped += 1
            unit.segments.append(idx)
            by_segment[idx] = unit

        return order, by_segment

    def _fill_from_cache(self, units: list[_Unit]) -> list[_Unit]:
        pending: list[_Unit] = []
        for unit in units:
            hit = self.cache.get(unit.key)
            if hit is None:
                pending.append(unit)
            else:
                unit.translation = hit
                self.stats.lines_cached += len(unit.segments)
        return pending

    def _store(self, units: list[_Unit]) -> None:
        rows = [
            (unit.key, unit.payload, unit.translation, self.config.model, unit.kind)
            for unit in units
            if unit.translation is not None and not unit.failed
        ]
        self.cache.put_many(rows)

    def _remember(self, units: list[_Unit]) -> None:
        """Keep the tail of the batch as continuity context for the next one."""
        limit = max(0, self.config.context_overlap)
        if not limit:
            return
        for unit in units:
            if unit.translation is not None and unit.kind not in (SIGN, SONG):
                self._recent.append((unit.payload, unit.translation))
        self._recent = self._recent[-limit:]

    # ------------------------------------------------------------------
    # Batching / repair
    # ------------------------------------------------------------------

    def _translate_units(self, units: list[_Unit], depth: int = 0) -> None:
        pending = list(units)
        validation_rounds = 0

        for attempt in range(self.config.max_retries + 1):
            if not pending:
                return
            try:
                answers = self._request(pending)
            except ResponseTruncated as exc:
                if len(pending) > 1 and depth < _MAX_SPLIT_DEPTH:
                    logger.warning("%s; splitting %d lines", exc, len(pending))
                    mid = len(pending) // 2
                    self._translate_units(pending[:mid], depth + 1)
                    self._translate_units(pending[mid:], depth + 1)
                    return
                answers = {}
            except TranslationError as exc:
                self._wait(attempt, str(exc))
                continue

            unresolved: list[_Unit] = []
            rejected = 0
            for position, unit in enumerate(pending, start=1):
                answer = answers.get(position)
                if answer is None:
                    unresolved.append(unit)
                    continue
                answer = strip_echoed_label(answer, self._label_text(unit))
                problem = validate_translation(
                    unit.payload, answer, script=self._script
                )
                if problem is None:
                    unit.translation = answer
                else:
                    self.stats.lines_rejected += 1
                    rejected += 1
                    logger.warning(
                        "rejected translation (%s): %r -> %r",
                        problem,
                        unit.payload[:60],
                        answer[:60],
                    )
                    unresolved.append(unit)

            if not unresolved:
                return

            if len(unresolved) < len(pending):
                # Partial success: only the leftovers are re-sent, not the
                # whole batch.
                self.stats.repairs += 1

            pending = unresolved

            # Splitting is for a batch the model could not answer (a reply
            # truncated by max_tokens). When the lines came back and only
            # failed validation, halving re-sends everything for nothing.
            if rejected == len(pending):
                validation_rounds += 1
                if validation_rounds >= _MAX_VALIDATION_ROUNDS:
                    logger.warning(
                        "%d line(s) keep failing validation; not retrying further",
                        len(pending),
                    )
                    break

            self._wait(attempt, f"{len(pending)} line(s) unresolved")

        for unit in pending:
            unit.translation = unit.payload
            unit.failed = True
            self.stats.lines_failed += len(unit.segments)
            logger.warning("keeping source text for: %r", unit.payload[:80])

    def _request(self, units: list[_Unit]) -> dict[int, str]:
        user_content = self._build_user_message(units)
        logger.debug(
            "LLM request (model=%s, %d lines):\n%s\n---\n%s",
            self.config.model,
            len(units),
            self._system_prompt,
            user_content,
        )
        payload = {
            "model": self.config.model,
            "messages": [
                self._system_message(),
                {"role": "user", "content": user_content},
            ],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            # OpenRouter unified reasoning control: explicitly disable
            # reasoning unless enabled in the config. Some reasoning models
            # (e.g. DeepSeek) return null content otherwise.
            "reasoning": {"enabled": self.config.enable_reasoning},
        }
        content = self._post(payload)
        answers = parse_numbered_response(content, len(units))
        if not answers:
            raise TranslationError("no numbered items in the response")
        return answers

    def _build_user_message(self, units: list[_Unit]) -> str:
        blocks: list[str] = []

        if self._recent:
            rows = "\n".join(f"{src} -> {dst}" for src, dst in self._recent)
            blocks.append(
                "Preceding lines with their translations, for continuity only. "
                "Do not translate them and do not include them in your answer:\n"
                f"{rows}"
            )

        numbered = []
        for position, unit in enumerate(units, start=1):
            numbered.append(f"[{position}] {self._label(unit)}{unit.payload}")
        blocks.append(
            f"Translate these {len(units)} subtitles. "
            f"Answer with exactly {len(units)} lines, [1] to [{len(units)}]:\n"
            + "\n".join(numbered)
        )
        return "\n\n".join(blocks)

    @staticmethod
    def _label_text(unit: _Unit) -> str:
        """The bare metadata marker of a line, without brackets."""
        if unit.kind == SIGN:
            return "sign"
        if unit.kind == SONG:
            return "song"
        return unit.speaker

    @classmethod
    def _label(cls, unit: _Unit) -> str:
        text = cls._label_text(unit)
        return f"<{text}> " if text else ""

    # ------------------------------------------------------------------
    # Transport
    # ------------------------------------------------------------------

    def _post(self, payload: dict) -> str:
        try:
            resp = self.session.post(API_URL, json=payload, timeout=180)
        except requests.RequestException as exc:
            self.stats.failed_requests += 1
            raise TranslationError(f"network error: {exc}") from exc

        if resp.status_code != 200:
            self.stats.failed_requests += 1
            raise TranslationError(f"HTTP {resp.status_code}: {resp.text[:300]}")

        try:
            data = resp.json()
        except ValueError as exc:
            self.stats.failed_requests += 1
            raise TranslationError(f"malformed API response: {exc}") from exc

        # Count usage before inspecting the content: a response with empty
        # content or a truncated answer is still billed, and hiding it
        # understates what retries cost.
        self.stats.requests += 1
        usage = data.get("usage") or {}
        self.stats.prompt_tokens += usage.get("prompt_tokens", 0) or 0
        self.stats.completion_tokens += usage.get("completion_tokens", 0) or 0
        cost = usage.get("cost")
        if isinstance(cost, (int, float)):
            self.stats.cost += float(cost)
        logger.debug("LLM response usage: %s", usage)

        try:
            choice = data["choices"][0]
        except (KeyError, IndexError, TypeError) as exc:
            raise TranslationError(f"malformed API response: {exc}") from exc

        message = choice.get("message") or {}
        content = message.get("content")
        finish_reason = choice.get("finish_reason", "?")
        logger.debug("LLM response (finish_reason=%s):\n%s", finish_reason, content)

        if finish_reason == "length":
            raise ResponseTruncated("response hit max_tokens")

        if not isinstance(content, str) or not content.strip():
            # Some models return null content (reasoning-only output or a
            # provider-side refusal); retryable.
            raise TranslationError(
                f"empty response content (finish_reason={finish_reason})"
            )
        return content

    def _wait(self, attempt: int, reason: str) -> None:
        if attempt >= self.config.max_retries:
            return
        delay = self.config.retry_delay * (2**attempt)
        logger.warning(
            "retry %d/%d in %.0fs (%s)",
            attempt + 1,
            self.config.max_retries,
            delay,
            reason,
        )
        time.sleep(delay)


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

# The contract asked of the model. Anything else is treated as continuation
# of the current item, which preserves line breaks inside a subtitle.
_STRICT_ITEM_RE = re.compile(r"^\s*\[\s*(\d{1,4})\s*\]\s?(.*)$")
# Tolerated fallback shapes ("3. text", "3) text"). Accepted only when the
# number continues the sequence: subtitle text itself often starts with
# "1. ..." in enumerations, and treating that as a new item would silently
# move text into the wrong subtitle.
_LOOSE_ITEM_RE = re.compile(r"^\s*(\d{1,4})\s*[.):\-]\s?(.*)$")


def parse_numbered_response(content: str, expected: int) -> dict[int, str]:
    """Parse a numbered model response into whatever items it contains.

    Returns the items that could be read, keyed by their number; missing or
    unparseable ones are simply absent so the caller can re-request only
    those instead of discarding a whole batch.

    The numbering style is decided once per response rather than per line:
    if the model used the requested "[N]" form anywhere, every other shape is
    treated as continuation text. Subtitles frequently contain their own
    enumerations ("1. ...", "2. ..."), and reading those as item boundaries
    moves text into the wrong subtitle without anything looking wrong.
    """
    strict = _scan_items(content, expected, allow_loose=False)
    if strict:
        return strict
    return _scan_items(content, expected, allow_loose=True)


def _scan_items(content: str, expected: int, *, allow_loose: bool) -> dict[int, str]:
    items: dict[int, list[str]] = {}
    current: int | None = None
    highest = 0

    for raw_line in content.split("\n"):
        number: int | None = None
        rest = ""

        match = _STRICT_ITEM_RE.match(raw_line)
        if match:
            candidate = int(match.group(1))
            if 1 <= candidate <= expected and candidate not in items:
                number, rest = candidate, match.group(2)
        elif allow_loose:
            match = _LOOSE_ITEM_RE.match(raw_line)
            if match:
                candidate = int(match.group(1))
                # Even without brackets, only a number that continues the
                # sequence can start an item.
                if (
                    candidate == highest + 1
                    and candidate <= expected
                    and candidate not in items
                ):
                    number, rest = candidate, match.group(2)

        if number is not None:
            current = number
            highest = max(highest, number)
            items[number] = [rest]
            continue

        if current is not None:
            items[current].append(raw_line)

    result: dict[int, str] = {}
    for number, parts in items.items():
        text = "\n".join(parts).strip("\n").strip()
        if text:
            result[number] = text
    return result
