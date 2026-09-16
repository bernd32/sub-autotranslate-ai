"""Shared test fixtures.

The subtitle files in test_files/ are real fansub releases and are gitignored,
so the committed suite must not depend on them: all fixtures below are
synthetic. Tests that want the real files use the `real_subtitle_files`
fixture, which skips when the directory is absent.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
REAL_FILES_DIR = REPO_ROOT / "test_files"

# A synthetic ASS file covering the structural cases that matter:
# a non-standard section before [Events], a Format line, commas inside the
# Text field, override tags at the start/middle/end, \N breaks, a vector
# drawing, a tag-only line and a Comment line.
SAMPLE_ASS = """\
[Script Info]
ScriptType: v4.00+
PlayResX: 1920
Title: synthetic, with a comma

[Aegisub Project Garbage]
Last Style Storage: Default

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, Bold, Italic, Alignment, Encoding
Style: Default,Arial,48,&H00FFFFFF,0,0,2,1
Style: signs,Arial,36,&H00FFFFFF,0,0,8,1
Style: OP Romaji,Arial,40,&H00FFFFFF,0,0,8,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,0:00:01.00,0:00:03.00,Default,Alice,0,0,0,,Good morning, everyone.
Dialogue: 0,0:00:03.00,0:00:05.00,Default,Bob,0,0,0,,{\\i1}Is she late again?{\\i0}
Dialogue: 0,0:00:05.00,0:00:07.00,signs,sign,0,0,0,,{\\pos(400,120)\\blur0.6}Staff Room
Dialogue: 0,0:00:07.00,0:00:09.00,signs,sign,0,0,0,,{\\pos(400,120)}Notice\\NNo entry today
Dialogue: 0,0:00:09.00,0:00:11.00,signs,sign,0,0,0,,{\\an7\\pos(0,0)\\p1}m 0 0 l 10 0 l 10 10
Dialogue: 0,0:00:11.00,0:00:13.00,Default,Alice,0,0,0,,{top}
Dialogue: 0,0:00:13.00,0:00:15.00,OP Romaji,,0,0,0,,A line of the opening song
Dialogue: 0,0:00:15.00,0:00:17.00,Default,Bob,0,0,0,,Wait!
Comment: 0,0:00:17.00,0:00:19.00,Default,Bob,0,0,0,,this must not be translated
Dialogue: 0,0:00:19.00,0:00:21.00,Default,Alice,0,0,0,,Wait!
"""

SAMPLE_SRT = """\
1
00:00:01,000 --> 00:00:03,000
Good morning, everyone.

2
00:00:03,000 --> 00:00:05,000
<i>Is she late again?</i>

3
00:00:05,000 --> 00:00:07,000
1. First point
2. Second point

4
00:00:07,000 --> 00:00:09,000
Wait!
"""


@pytest.fixture
def sample_ass() -> str:
    return SAMPLE_ASS


@pytest.fixture
def sample_srt() -> str:
    return SAMPLE_SRT


@pytest.fixture
def real_subtitle_files() -> list[Path]:
    """Real fansub files from test_files/, if the user has them locally."""
    if not REAL_FILES_DIR.is_dir():
        pytest.skip("test_files/ not present")
    files = sorted(
        p
        for p in REAL_FILES_DIR.iterdir()
        if p.suffix.lower() in (".ass", ".ssa", ".srt")
    )
    if not files:
        pytest.skip("no subtitle files in test_files/")
    return files
