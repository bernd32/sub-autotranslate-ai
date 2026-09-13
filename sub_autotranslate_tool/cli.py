"""Command-line interface for sub-autotranslate-tool."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .config import DEFAULT_CONFIG_PATH, Config, ensure_default_config, load_config
from .subtitles import SUPPORTED_EXTENSIONS, SubtitleFormatError, load_subtitles
from .translator import Translator


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="sub-autotranslate-tool",
        description=(
            "Translate SRT/ASS subtitle files with an LLM via OpenRouter. "
            "INPUT may be a single subtitle file or a directory — all supported "
            "subtitle files in the directory are translated."
        ),
    )
    p.add_argument(
        "input",
        nargs="?",
        type=Path,
        help="Subtitle file (.srt/.ass/.ssa) or a directory containing them.",
    )
    p.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for translated files (default: same directory as each input file).",
    )
    p.add_argument(
        "-c",
        "--config",
        type=Path,
        default=None,
        help=f"Path to config file (default: {DEFAULT_CONFIG_PATH}).",
    )
    p.add_argument("--model", default=None, help="Override the model from the config.")
    p.add_argument("--source-language", default=None, help="Override source language.")
    p.add_argument("--target-language", default=None, help="Override target language.")
    p.add_argument("--api-key", default=None, help="OpenRouter API key (overrides env/config).")
    p.add_argument(
        "--proxy",
        default=None,
        help="Proxy for API requests, e.g. http://127.0.0.1:8080 or socks5://127.0.0.1:1080.",
    )
    p.add_argument(
        "--init-config",
        action="store_true",
        help="Create the default config file and print its path, then exit.",
    )
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return p


def collect_input_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise SystemExit(
                f"error: unsupported file type: {input_path} "
                f"(supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))})"
            )
        return [input_path]
    if input_path.is_dir():
        files = sorted(
            f
            for f in input_path.iterdir()
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
        )
        if not files:
            raise SystemExit(f"error: no subtitle files found in {input_path}")
        return files
    raise SystemExit(f"error: input not found: {input_path}")


def output_path_for(src: Path, output_dir: Path | None, suffix: str) -> Path:
    out_dir = output_dir or src.parent
    name = f"{src.stem}.{suffix}{src.suffix}" if suffix else src.name
    return out_dir / name


def translate_file(
    src: Path, dst: Path, translator: Translator, config: Config
) -> None:
    doc = load_subtitles(src)
    label = f"{src.name}: "
    texts = [seg.text for seg in doc.segments]
    translated = translator.translate_all(texts, progress_label=label)
    for seg, new_text in zip(doc.segments, translated):
        seg.text = new_text
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(doc.render(), encoding="utf-8")
    print(f"  -> {dst}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.init_config:
        path = ensure_default_config(args.config or DEFAULT_CONFIG_PATH)
        print(f"Config file: {path}")
        print("Edit it to set your API key, model, languages and prompt.")
        return 0

    if args.input is None:
        build_parser().error("the following arguments are required: input")

    # --- configuration ---
    try:
        config = load_config(args.config)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"error: {exc}") from exc

    if args.model:
        config.model = args.model
    if args.source_language:
        config.source_language = args.source_language
    if args.target_language:
        config.target_language = args.target_language
    if args.proxy:
        config.proxy = args.proxy

    api_key = config.effective_api_key(args.api_key)
    if not api_key:
        raise SystemExit(
            "error: no OpenRouter API key.\n"
            "Set OPENROUTER_API_KEY, pass --api-key, or set api_key in "
            f"{args.config or DEFAULT_CONFIG_PATH}"
        )

    # --- input/output ---
    files = collect_input_files(args.input)
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Model: {config.model}")
    print(f"Translation: {config.source_language} -> {config.target_language}")
    if config.proxy:
        print(f"Proxy: {config.proxy}")
    print(f"Files to translate: {len(files)}")

    translator = Translator(config, api_key)
    failures: list[tuple[Path, str]] = []

    for src in files:
        dst = output_path_for(src, args.output_dir, config.output_suffix)
        if dst == src:
            dst = src.with_name(f"{src.stem}.translated{src.suffix}")
        print(f"Translating {src.name} ({src.suffix}) ...")
        try:
            translate_file(src, dst, translator, config)
        except SubtitleFormatError as exc:
            failures.append((src, str(exc)))
            print(f"  !! skipped: {exc}", file=sys.stderr)
        except KeyboardInterrupt:
            print("\nInterrupted.", file=sys.stderr)
            break

    stats = translator.stats
    print(
        f"\nDone. API requests: {stats.requests}, "
        f"tokens: {stats.prompt_tokens} in / {stats.completion_tokens} out."
    )
    if failures:
        print("Failed files:", file=sys.stderr)
        for path, reason in failures:
            print(f"  {path}: {reason}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
