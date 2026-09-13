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
model = "google/gemini-3.8-flash"

# Translation language pair
source_language = "English"
target_language = "Russian"

# Suffix inserted into the output file name, e.g. "movie.ru.srt"
output_suffix = "ru"

# Number of subtitle lines sent to the LLM in a single request.
# Larger batches give the model more context and are cheaper,
# smaller batches are more robust against formatting errors.
batch_size = 40

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

# The prompt template sent to the model for every batch.
# Available placeholders:
#   {source_language}  - source language name
#   {target_language}  - target language name
#   {count}            - number of lines in this batch
# The subtitle lines themselves are appended after this prompt,
# numbered as [1], [2], ...
prompt = """You are a professional subtitle translator.
Translate the following {count} subtitle lines from {source_language} to {target_language}.

Rules:
- Use the surrounding lines as context so the translation reads naturally and
  consistently (characters, tone, register, jokes, idioms).
- Do NOT translate literally word-by-word; produce fluent, natural {target_language}
  as a native speaker would write it for film/TV subtitles.
- Keep each numbered item a single subtitle: preserve internal line breaks inside an item.
- Keep the numbering intact: answer with exactly {count} items, numbered [1]..[{count}],
  one item per number, nothing else (no comments, no explanations).
- Preserve any markup such as HTML-like tags (<i>, </i>), ASS override tags
  ({{\\\\i1}}, {{\\\\pos(...)}} etc.) and line-break markers (\\\\N, \\\\h) exactly.
- Do not translate proper names phonetically unless that is the established
  convention in {target_language}; keep numbers, and punctuation style appropriate
  for {target_language}."""
'''


@dataclass
class Config:
    api_key: str = ""
    model: str = "google/gemini-3.8-flash"
    source_language: str = "English"
    target_language: str = "Russian"
    output_suffix: str = "ru"
    batch_size: int = 40
    temperature: float = 0.3
    enable_reasoning: bool = False
    max_tokens: int = 8192
    max_retries: int = 4
    retry_delay: float = 2.0
    proxy: str = ""
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
