"""Parsing and re-rendering of SRT and ASS subtitle files.

Both formats are parsed into a `SubtitleDocument` exposing a flat list of
translatable `Segment`s. After translation, `render()` rebuilds the file
with the translated text reinserted into the original timestamp structure,
leaving all non-dialogue content (headers, styles, timing) untouched.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Common structures
# ---------------------------------------------------------------------------


@dataclass
class Segment:
    """One translatable unit of subtitle text (may contain line breaks)."""

    text: str


class SubtitleDocument:
    """A parsed subtitle file."""

    def __init__(self) -> None:
        self.segments: list[Segment] = []

    def render(self) -> str:  # pragma: no cover - overridden
        raise NotImplementedError


class SubtitleFormatError(ValueError):
    pass


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

_ENCODINGS = ("utf-8-sig", "utf-8", "cp1251", "latin-1")


def read_text(path: Path) -> str:
    """Read a subtitle file, trying common encodings."""
    raw = path.read_bytes()
    for enc in _ENCODINGS:
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")  # pragma: no cover


# ---------------------------------------------------------------------------
# SRT
# ---------------------------------------------------------------------------

_SRT_TIMING_RE = re.compile(
    r"^\s*\d{1,2}:\d{2}:\d{2}[,.]\d{1,3}\s*-->\s*\d{1,2}:\d{2}:\d{2}[,.]\d{1,3}"
)


class SrtDocument(SubtitleDocument):
    """SRT file parsed into (index, timing, text) blocks."""

    def __init__(self, text: str) -> None:
        super().__init__()
        # blocks: list of (index_line, timing_line) parallel to self.segments
        self.blocks: list[tuple[str, str]] = []
        self._parse(text)

    def _parse(self, text: str) -> None:
        lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")

        def flush(block_lines: list[str]) -> None:
            # Drop leading/trailing blank lines within a block.
            while block_lines and not block_lines[0].strip():
                block_lines.pop(0)
            while block_lines and not block_lines[-1].strip():
                block_lines.pop()
            if not block_lines:
                return
            timing_idx = next(
                (i for i, ln in enumerate(block_lines[:3]) if _SRT_TIMING_RE.match(ln)),
                None,
            )
            if timing_idx is None:
                # Not a valid cue (e.g. stray text) — skip silently.
                return
            index_line = block_lines[0] if timing_idx > 0 else ""
            timing_line = block_lines[timing_idx]
            body = block_lines[timing_idx + 1 :]
            self.blocks.append((index_line, timing_line))
            self.segments.append(Segment("\n".join(body)))

        current: list[str] = []
        for line in lines:
            if line.strip():
                current.append(line)
            else:
                flush(current)
                current = []
        flush(current)

        if not self.segments:
            raise SubtitleFormatError("No valid SRT cues found")

    def render(self) -> str:
        out: list[str] = []
        for i, ((index_line, timing_line), seg) in enumerate(
            zip(self.blocks, self.segments), start=1
        ):
            out.append(index_line.strip() if index_line.strip() else str(i))
            out.append(timing_line.strip())
            out.extend(seg.text.split("\n"))
            out.append("")
        return "\n".join(out)


# ---------------------------------------------------------------------------
# ASS / SSA
# ---------------------------------------------------------------------------

_ASS_DIALOGUE_RE = re.compile(r"^\s*Dialogue\s*:", re.IGNORECASE)


class AssDocument(SubtitleDocument):
    """ASS/SSA file; only the Text field of Dialogue events is translated.

    The text field is the 10th comma-separated value and may itself contain
    commas, so the line is split with maxsplit=9. Override tags ({\\...})
    and \\N/\\h markers are left in place for the LLM to preserve.
    """

    def __init__(self, text: str) -> None:
        super().__init__()
        self.lines: list[str] = text.replace("\r\n", "\n").replace("\r", "\n").split(
            "\n"
        )
        # dialogue line index -> segment index
        self._dialogue_map: dict[int, int] = {}
        self._parse()

    def _parse(self) -> None:
        in_events = False
        for i, line in enumerate(self.lines):
            stripped = line.strip()
            if stripped.startswith("[") and stripped.endswith("]"):
                in_events = stripped.lower() == "[events]"
                continue
            if in_events and _ASS_DIALOGUE_RE.match(line):
                _, _, rest = line.partition(":")
                parts = rest.split(",", 9)
                if len(parts) < 10:
                    continue
                self._dialogue_map[i] = len(self.segments)
                self.segments.append(Segment(parts[9]))

        if not self.segments:
            raise SubtitleFormatError("No Dialogue events found in [Events] section")

    def render(self) -> str:
        lines = list(self.lines)
        for line_idx, seg_idx in self._dialogue_map.items():
            head, _, rest = lines[line_idx].partition(":")
            parts = rest.split(",", 9)
            parts[9] = self.segments[seg_idx].text
            lines[line_idx] = f"{head}:{','.join(parts)}"
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

SUPPORTED_EXTENSIONS = {".srt", ".ass", ".ssa"}


def load_subtitles(path: Path) -> SubtitleDocument:
    """Parse *path* into a SubtitleDocument based on its extension."""
    ext = path.suffix.lower()
    text = read_text(path)
    if ext == ".srt":
        return SrtDocument(text)
    if ext in (".ass", ".ssa"):
        return AssDocument(text)
    raise SubtitleFormatError(f"Unsupported subtitle format: {ext or '(none)'}")
