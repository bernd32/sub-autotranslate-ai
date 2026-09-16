"""Configuration handling: loads/creates a TOML config file.

Default location: $XDG_CONFIG_HOME/sub-autotranslate-tool/config.toml
(usually ~/.config/sub-autotranslate-tool/config.toml)

The OpenRouter API key is read (in priority order) from:
  1. --api-key CLI flag
  2. OPENROUTER_API_KEY environment variable
  3. api_key in the config file
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path

DEFAULT_CONFIG_DIR = Path(
    os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")
) / "sub-autotranslate-tool"
DEFAULT_CONFIG_PATH = DEFAULT_CONFIG_DIR / "config.toml"

DEFAULT_CONFIG_TEMPLATE = '''\
# sub-autotranslate-tool configuration
# OpenRouter API key. Prefer the OPENROUTER_API_KEY environment variable
# over storing the key here. Get a key at https://openrouter.ai/keys
api_key = ""

# LLM model to use for translation (any model id from https://openrouter.ai/models)
model = "deepseek/deepseek-v4.1-flash"

# Translation language pair
source_language = "English"
target_language = "Russian"

# Suffix inserted into the output file name, e.g. "movie.ru.srt"
output_suffix = "ru"

# Number of subtitle lines sent to the LLM in a single request.
# Larger batches give the model more context and are cheaper,
# smaller batches are more robust against formatting errors.
batch_size = 40

# Send the ASS "Name" (speaker) field along with each line. Costs a few
# tokens per line and tells the model who is talking, which decides
# grammatical gender and register in many target languages.
send_speaker = true

# Number of already translated lines shown as context at the start of the
# next batch, so dialogue does not break at batch boundaries. 0 disables.
context_overlap = 5

# Translate each distinct line once and reuse the result for its duplicates
# (repeated signs and nameplates). Saves tokens and keeps wording identical.
# Set to false if you want every occurrence translated in its own right.
dedup = true

# Reuse translations across files and runs via an on-disk translation memory
# (~/.cache/sub-autotranslate-tool). Opening/ending lyrics and location signs
# repeat in every episode of a season. Entries are invalidated automatically
# when the model, the language pair, the prompt or the glossary changes.
cache = true

# Before translating, ask the model once for canonical renderings of the
# names and recurring terms found in all input files, then pin them into
# every request. Costs one small extra request per run and is the main
# defence against a character being named differently in every scene.
glossary_auto = true

# Provider prompt caching for the static instructions:
#   "auto" - mark the prefix explicitly for models that need it (anthropic/*),
#            rely on automatic prefix caching elsewhere
#   "on"   - always mark the prefix (cache_control)
#   "off"  - never mark it
prompt_cache = "auto"

# Styles that must not be translated at all (the line is copied as is).
# Names are matched case-insensitively and accept globs, e.g. "sign*".
skip_styles = []

# Force a style to be treated as on-screen text or as song lyrics. Detection
# from the style name usually suffices; use these for non-obvious names.
sign_styles = []
song_styles = []

# Sampling temperature (lower = more deterministic)
temperature = 0.3

# Reasoning (thinking) mode for reasoning-capable models (e.g. DeepSeek).
# Disabled by default: it wastes tokens and some reasoning models return
# null content unless explicitly disabled (see OpenRouter reasoning docs).
# Set to true if you deliberately want reasoning enabled.
enable_reasoning = false

# Maximum tokens in a single LLM response
max_tokens = 8192

# Max retries per batch on API or formatting errors
max_retries = 4

# Seconds to wait between retry attempts (doubles on each retry)
retry_delay = 2.0

# Optional proxy for API requests, e.g.:
#   "http://127.0.0.1:8080"
#   "https://user:pass@proxy.example.com:8443"
#   "socks5://127.0.0.1:1080"
# Leave empty to connect directly. The standard HTTPS_PROXY/HTTP_PROXY
# environment variables are also honored automatically.
proxy = ""

# Log level: debug / info / warning / error / critical
# Use "debug" to log the exact text sent to and received from the LLM
# (useful to verify prompt behavior and token usage).
log_level = "info"

# The static instructions, sent as the system message of every request.
# Keeping them constant across batches is what lets providers cache them,
# so avoid putting anything batch-specific here.
# Available placeholders:
#   {source_language}  - source language name
#   {target_language}  - target language name
#   {glossary}         - the glossary block (empty when there is none)
#   {context}          - the --context series notes (empty when not given)
# The subtitle lines are sent separately, numbered as [1], [2], ...
prompt = """You are a professional subtitle translator for animated series.
Translate from {source_language} to {target_language}.

Input format: one subtitle per line, as "[N] text". The text may be preceded
by a marker in angle brackets:
  <Name>  - the character speaking the line
  <sign>  - on-screen text (signboard, caption, letter), not speech
  <song>  - song lyrics
Markers are information for you, never content: do not translate or repeat them.

Answer format: exactly one line per item, as "[N] translation", same numbers,
same order, nothing else. No comments, no notes, no blank lines, and never
repeat the source text.

Translation rules:
- Translate meaning, not words: write fluent {target_language} the way a native
  subtitle translator would, keeping register, tone and humour.
- Use the surrounding lines as context; keep names and terminology consistent.
- <sign> items are on-screen text: keep them short and label-like, with no
  added words and no sentence-final period.
- <song> items are lyrics: keep the imagery consistent across adjacent lines.
- Keep the markers \\\\N (line break) and \\\\h (hard space) exactly as they appear,
  in the same number and the same places.
- Copy every <0/>, <1/> marker unchanged and in place: each stands for styling
  that was removed before translation.
- Keep inline tags such as <i> and </i> around the same words.
- Use punctuation and quotation marks native to {target_language}; keep numbers.
- Never merge two items, never split one, never add or drop an item.{glossary}{context}"""
'''


@dataclass
class Config:
    api_key: str = ""
    model: str = "deepseek/deepseek-v4.1-flash"
    source_language: str = "English"
    target_language: str = "Russian"
    output_suffix: str = "ru"
    batch_size: int = 40
    send_speaker: bool = True
    context_overlap: int = 5
    dedup: bool = True
    cache: bool = True
    glossary_auto: bool = True
    prompt_cache: str = "auto"
    skip_styles: list[str] = field(default_factory=list)
    sign_styles: list[str] = field(default_factory=list)
    song_styles: list[str] = field(default_factory=list)
    temperature: float = 0.3
    enable_reasoning: bool = False
    max_tokens: int = 8192
    max_retries: int = 4
    retry_delay: float = 2.0
    proxy: str = ""
    log_level: str = "info"
    prompt: str = ""

    def effective_api_key(self, cli_key: str | None = None) -> str:
        key = cli_key or os.environ.get("OPENROUTER_API_KEY") or self.api_key
        return key.strip()


def ensure_default_config(path: Path = DEFAULT_CONFIG_PATH) -> Path:
    """Create the default config file (with comments) if it doesn't exist."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(DEFAULT_CONFIG_TEMPLATE, encoding="utf-8")
    return path


def load_config(path: Path | None = None) -> Config:
    """Load config from *path* (default config path if None).

    Creates the default config file on first run. Unknown keys are ignored
    so the config stays forward/backward compatible.
    """
    path = path or DEFAULT_CONFIG_PATH
    if path == DEFAULT_CONFIG_PATH:
        ensure_default_config(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("rb") as fh:
        data = tomllib.load(fh)

    known = {f.name for f in fields(Config)}
    unknown = set(data) - known
    if unknown:
        import sys

        print(
            f"Warning: ignoring unknown config keys: {', '.join(sorted(unknown))}",
            file=sys.stderr,
        )

    cfg = Config(**{k: v for k, v in data.items() if k in known})
    if not cfg.prompt:
        # Fall back to the prompt from the template.
        cfg.prompt = tomllib.loads(DEFAULT_CONFIG_TEMPLATE)["prompt"]
    return cfg
