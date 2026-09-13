"""Extract subtitles from MKV files using ffmpeg/ffprobe.

Workflow:
  1. ffprobe the FIRST mkv in the directory to discover its subtitle
     streams (codec -> .srt for subrip, .ass/.ssa for ass).
  2. Extract the chosen subtitle stream (default: first, i.e. -map 0:s:0)
     from every mkv file in the directory.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

# ffmpeg codec name -> output extension
_CODEC_EXT = {
    "subrip": ".srt",
    "srt": ".srt",
    "ass": ".ass",
    "ssa": ".ass",
}


class MkvExtractError(RuntimeError):
    pass


@dataclass
class SubtitleStream:
    relative_index: int  # 0 = first subtitle stream (0:s:0)
    codec: str
    language: str | None
    title: str | None


def _require_tools() -> None:
    for tool in ("ffmpeg", "ffprobe"):
        try:
            subprocess.run(
                [tool, "-version"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
            )
        except (FileNotFoundError, subprocess.CalledProcessError) as exc:
            raise MkvExtractError(
                f"'{tool}' not found. Install ffmpeg (e.g. 'sudo apt install ffmpeg')."
            ) from exc


def probe_subtitle_streams(path: Path) -> list[SubtitleStream]:
    """Return all subtitle streams of an mkv file."""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "s",
        "-show_entries", "stream=index,codec_name:stream_tags=language,title",
        "-of", "json",
        str(path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, check=True, text=True)
    except subprocess.CalledProcessError as exc:
        raise MkvExtractError(f"ffprobe failed for {path}: {exc.stderr.strip()}") from exc

    try:
        streams = json.loads(result.stdout).get("streams", [])
    except json.JSONDecodeError as exc:
        raise MkvExtractError(f"could not parse ffprobe output for {path}") from exc

    out: list[SubtitleStream] = []
    for i, s in enumerate(streams):
        tags = s.get("tags") or {}
        out.append(
            SubtitleStream(
                relative_index=i,
                codec=(s.get("codec_name") or "").lower(),
                language=tags.get("language"),
                title=tags.get("title"),
            )
        )
    return out


def format_streams(streams: list[SubtitleStream]) -> str:
    rows = []
    for s in streams:
        label = s.title or s.language or "unknown"
        rows.append(f"  0:s:{s.relative_index}  codec={s.codec or '?'}  ({label})")
    return "\n".join(rows)


def detect_extension(streams: list[SubtitleStream], stream_index: int) -> str:
    if not streams:
        raise MkvExtractError("no subtitle streams found")
    if stream_index >= len(streams):
        raise MkvExtractError(
            f"subtitle stream 0:s:{stream_index} not found "
            f"(available: 0..{len(streams) - 1})"
        )
    codec = streams[stream_index].codec
    ext = _CODEC_EXT.get(codec)
    if ext is None:
        raise MkvExtractError(
            f"unsupported subtitle codec '{codec}' in stream 0:s:{stream_index}; "
            f"supported: {', '.join(sorted(_CODEC_EXT))}"
        )
    return ext


def extract_one(src: Path, out_dir: Path, stream_index: int, ext: str) -> Path:
    dst = out_dir / f"{src.stem}{ext}"
    cmd = [
        "ffmpeg",
        "-v", "error",
        "-y",  # overwrite
        "-i", str(src),
        "-map", f"0:s:{stream_index}",
        str(dst),
    ]
    try:
        subprocess.run(cmd, capture_output=True, check=True, text=True)
    except subprocess.CalledProcessError as exc:
        raise MkvExtractError(
            f"ffmpeg failed for {src.name}: {exc.stderr.strip()}"
        ) from exc
    return dst


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="mkv-sub-extract",
        description=(
            "Extract subtitle streams from all MKV files in a directory "
            "using ffmpeg."
        ),
    )
    p.add_argument("directory", nargs="?", type=Path, help="Directory with .mkv files.")
    p.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: same as the input directory).",
    )
    p.add_argument(
        "--stream",
        type=int,
        default=0,
        metavar="N",
        help="Subtitle stream index N (ffmpeg '-map 0:s:N'). Default: 0 (first).",
    )
    p.add_argument(
        "--list",
        action="store_true",
        help="List subtitle streams of the first mkv file and exit.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.directory is None:
        build_parser().error("the following arguments are required: directory")

    try:
        _require_tools()

        files = sorted(
            f for f in args.directory.glob("*.mkv") if f.is_file()
        )
        if not files:
            raise MkvExtractError(f"no .mkv files found in {args.directory}")

        first_streams = probe_subtitle_streams(files[0])
        if args.list:
            print(f"Subtitle streams in {files[0].name}:")
            print(format_streams(first_streams) or "  (none)")
            return 0

        ext = detect_extension(first_streams, args.stream)
        out_dir = args.output_dir or args.directory
        out_dir.mkdir(parents=True, exist_ok=True)

    except MkvExtractError as exc:
        raise SystemExit(f"error: {exc}") from exc

    print(
        f"Extracting stream 0:s:{args.stream} ({ext}) from "
        f"{len(files)} file(s) -> {out_dir}"
    )
    failures = 0
    for src in files:
        try:
            dst = extract_one(src, out_dir, args.stream, ext)
            print(f"  {src.name} -> {dst.name}")
        except MkvExtractError as exc:
            failures += 1
            print(f"  !! {src.name}: {exc}", file=sys.stderr)

    if failures:
        print(f"\nDone with {failures} failed file(s).", file=sys.stderr)
        return 1
    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
