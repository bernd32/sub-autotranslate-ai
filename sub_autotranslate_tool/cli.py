"""Command-line interface for sub-autotranslate-tool."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from . import __version__
from .cache import DEFAULT_CACHE_PATH, NullCache, TranslationCache
from .config import DEFAULT_CONFIG_PATH, Config, ensure_default_config, load_config
from .glossary import Glossary
from .log import LEVELS, setup_logging
from .subtitles import (
    SUPPORTED_EXTENSIONS,
    SubtitleDocument,
    SubtitleFormatError,
    load_subtitles,
)
from .translator import Translator

logger = logging.getLogger(__name__)

# Series notes picked up automatically from the input directory.
CONTEXT_FILENAMES = (".subctx.txt", "series.txt")


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
        "--context",
        type=Path,
        default=None,
        help=(
            "Text file with notes about the series (synopsis, characters and "
            "their gender, how they address each other). Pinned into every "
            "request. Picked up automatically from "
            f"{' or '.join(CONTEXT_FILENAMES)} next to the input."
        ),
    )
    p.add_argument(
        "--glossary",
        type=Path,
        default=None,
        help=(
            "TOML glossary of names/terms. Loaded if it exists, otherwise "
            "generated and written there for reuse across a season."
        ),
    )
    p.add_argument(
        "--refresh-glossary",
        action="store_true",
        help="Rebuild the glossary even if the --glossary file already exists.",
    )
    p.add_argument(
        "--no-glossary",
        action="store_true",
        help="Skip the glossary pre-pass entirely.",
    )
    p.add_argument(
        "--no-cache",
        action="store_true",
        help="Do not read or write the on-disk translation memory.",
    )
    p.add_argument(
        "--cache-clear",
        action="store_true",
        help=f"Delete all cached translations ({DEFAULT_CACHE_PATH}), then exit.",
    )
    p.add_argument(
        "--log-level",
        choices=LEVELS,
        default=None,
        help=(
            "Log level (default: from config, 'info'). Use 'debug' to log the "
            "exact text sent to and received from the LLM."
        ),
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


def translate_document(
    doc: SubtitleDocument, src: Path, dst: Path, translator: Translator
) -> None:
    translated = translator.translate_segments(
        doc.segments, progress_label=f"{src.name}: "
    )
    for seg, new_text in zip(doc.segments, translated):
        seg.text = new_text
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(doc.render(), encoding="utf-8")
    print(f"  -> {dst}")


def find_context_file(input_path: Path, explicit: Path | None) -> Path | None:
    if explicit is not None:
        if not explicit.is_file():
            raise SystemExit(f"error: context file not found: {explicit}")
        return explicit
    base = input_path if input_path.is_dir() else input_path.parent
    for name in CONTEXT_FILENAMES:
        candidate = base / name
        if candidate.is_file():
            return candidate
    return None


def resolve_glossary(args: argparse.Namespace, translator: Translator, docs) -> Glossary:
    """Load the glossary from disk, or build it once for the whole run."""
    if args.no_glossary:
        return Glossary()

    path: Path | None = args.glossary
    if path is not None and path.is_file() and not args.refresh_glossary:
        try:
            glossary = Glossary.load(path)
        except (OSError, ValueError) as exc:
            raise SystemExit(f"error: cannot read glossary {path}: {exc}") from exc
        print(f"Glossary: {len(glossary.terms)} term(s) from {path}")
        return glossary

    if not translator.config.glossary_auto and path is None:
        return Glossary()

    # One pre-pass over every input file, so a season shares one glossary.
    all_segments = [seg for doc in docs for seg in doc.segments]
    glossary = translator.build_glossary(all_segments)
    if glossary and path is not None:
        try:
            glossary.save(path)
            print(f"Glossary: {len(glossary.terms)} term(s) written to {path}")
        except OSError as exc:
            logger.warning("could not write glossary to %s: %s", path, exc)
    elif glossary:
        print(f"Glossary: {len(glossary.terms)} term(s) (pass --glossary to keep it)")
    return glossary


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.init_config:
        path = ensure_default_config(args.config or DEFAULT_CONFIG_PATH)
        print(f"Config file: {path}")
        print("Edit it to set your API key, model, languages and prompt.")
        return 0

    if args.cache_clear:
        cache = TranslationCache()
        removed = cache.clear()
        cache.close()
        print(f"Removed {removed} cached translation(s) from {cache.path}")
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

    setup_logging(args.log_level or config.log_level)
    logger.debug("configuration: %s", config)

    api_key = config.effective_api_key(args.api_key)
    if not api_key:
        logger.critical("no OpenRouter API key configured")
        raise SystemExit(
            "error: no OpenRouter API key.\n"
            "Set OPENROUTER_API_KEY, pass --api-key, or set api_key in "
            f"{args.config or DEFAULT_CONFIG_PATH}"
        )

    # --- input/output ---
    files = collect_input_files(args.input)
    logger.info(
        "translating %d file(s) with model %s (%s -> %s)",
        len(files),
        config.model,
        config.source_language,
        config.target_language,
    )
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Model: {config.model}")
    print(f"Translation: {config.source_language} -> {config.target_language}")
    if config.proxy:
        print(f"Proxy: {config.proxy}")

    # Parsing every file up front keeps a malformed file from surfacing
    # halfway through a paid run, and lets the glossary pre-pass see the
    # whole season at once.
    failures: list[tuple[Path, str]] = []
    parsed: list[tuple[Path, SubtitleDocument]] = []
    for src in files:
        try:
            parsed.append((src, load_subtitles(src)))
        except (SubtitleFormatError, OSError) as exc:
            failures.append((src, str(exc)))
            logger.error("skipping %s: %s", src.name, exc)
            print(f"  !! skipped {src.name}: {exc}", file=sys.stderr)

    if not parsed:
        print("error: no readable subtitle files", file=sys.stderr)
        return 1

    total_lines = sum(len(doc.segments) for _, doc in parsed)
    print(f"Files to translate: {len(parsed)} ({total_lines} lines)")

    context_text = ""
    context_file = find_context_file(args.input, args.context)
    if context_file is not None:
        context_text = context_file.read_text(encoding="utf-8", errors="replace")
        print(f"Series context: {context_file}")

    use_cache = config.cache and not args.no_cache
    cache = TranslationCache() if use_cache else NullCache()
    if use_cache and cache.enabled:
        print(f"Translation memory: {cache.count()} entries in {cache.path}")

    translator = Translator(config, api_key, cache=cache, series_context=context_text)

    try:
        glossary = resolve_glossary(args, translator, [doc for _, doc in parsed])
        if glossary:
            translator.attach_glossary(glossary)

        for src, doc in parsed:
            dst = output_path_for(src, args.output_dir, config.output_suffix)
            if dst == src:
                dst = src.with_name(f"{src.stem}.translated{src.suffix}")
            print(f"Translating {src.name} ({src.suffix}) ...")
            try:
                translate_document(doc, src, dst, translator)
            except OSError as exc:
                failures.append((src, str(exc)))
                logger.error("could not write output for %s: %s", src.name, exc)
                print(f"  !! failed: {exc}", file=sys.stderr)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
    finally:
        cache.close()

    print_report(translator)
    if failures:
        print("Failed files:", file=sys.stderr)
        for path, reason in failures:
            print(f"  {path}: {reason}", file=sys.stderr)
        return 1
    return 0


def print_report(translator: Translator) -> None:
    stats = translator.stats
    print(
        f"\nDone. API requests: {stats.requests}"
        + (f" (+{stats.failed_requests} failed)" if stats.failed_requests else "")
        + f", tokens: {stats.prompt_tokens} in / {stats.completion_tokens} out."
    )
    if stats.cost > 0:
        print(f"Cost: ${stats.cost:.4f} (USD)")
    else:
        print("Cost: unavailable (provider did not report usage.cost)")

    saved = stats.lines_cached + stats.lines_deduped + stats.lines_skipped
    print(
        f"Lines: {stats.lines_total} total, {stats.lines_sent} sent to the model"
        + (f", {saved} not sent" if saved else "")
    )
    if saved:
        print(
            f"  not sent: {stats.lines_cached} from cache, "
            f"{stats.lines_deduped} duplicates, "
            f"{stats.lines_skipped} markup-only/skipped styles"
        )
    if stats.lines_rejected or stats.repairs:
        print(
            f"Quality: {stats.lines_rejected} line(s) failed validation, "
            f"{stats.repairs} follow-up request(s) for individual lines"
        )
    if stats.lines_failed:
        print(
            f"WARNING: {stats.lines_failed} line(s) kept their source text "
            "(see warnings above)",
            file=sys.stderr,
        )


if __name__ == "__main__":
    raise SystemExit(main())
