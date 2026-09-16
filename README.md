# sub-autotranslate-tool

A Linux CLI tool for high-quality, context-aware AI translation of **SRT** and
**ASS/SSA** subtitle files using LLMs via [OpenRouter](https://openrouter.ai).

Default translation pair: **English → Russian**
Default model: `deepseek/deepseek-v4.1-flash`

## How it works

1. Parses the subtitle file into blocks (index, timestamp, text). For ASS files
   only the `Text` field of `Dialogue` events is touched — headers, styles and
   timing are preserved byte-for-byte. Field positions come from the `Format:`
   line, so SSA v4.00 and ASS v4.00+ both work.
2. Classifies every line as dialogue, on-screen sign or song lyrics (from its
   `Style`/`Name`), and splits it into a markup skeleton plus bare text.
   Override tags, vector drawings and lines with no words never reach the
   model; interior tags become `<0/>` placeholders.
3. Collapses identical lines into one translation unit and fills whatever the
   on-disk translation memory already knows.
4. Sends the rest in numbered batches, each line labelled with its speaker or
   type, together with the glossary, the series notes, and a few
   already-translated lines so dialogue doesn't break at batch boundaries.
5. Validates every answer (markup preserved, target script, plausible length)
   and re-requests only the lines that failed.
6. Reinserts the translated text into the original markup and timestamps.

The static instructions travel in a separate `system` message, which lets
providers cache that prefix across all batches of a run.

If a line can't be translated after all retries it keeps its source text, and
the final report says how many lines that happened to — untranslated leftovers
are never silent.

## What it costs, and how to spend less

The run ends with a report like:

```
Done. API requests: 3, tokens: 12400 in / 5900 out.
Cost: $0.0021 (USD)
Lines: 171 total, 108 sent to the model, 63 not sent
  not sent: 0 from cache, 54 duplicates, 9 markup-only/skipped styles
```

The knobs that matter most, roughly in order of effect:

| Setting | Effect |
| --- | --- |
| `cache` | Reuses translations across episodes. A season of sign/OP-heavy releases pays for that material once. |
| `dedup` | Repeated signs and nameplates are translated once per file. |
| `batch_size` | Fewer, larger batches mean the instructions are amortised over more lines. |
| `glossary_auto` | One extra small request per run; usually pays for itself in consistency. |
| `context_overlap` | Costs a few lines of input per batch, buys continuity across boundaries. |
| `enable_reasoning` | Leave `false`: reasoning tokens are billed and add nothing here. |

Measure before tuning: run with `--log-level debug` to see the exact text of
every request, and compare the reported token counts.

## Installation

### System-wide (recommended): pipx

[pipx](https://pipx.pypa.io/) installs CLI tools into isolated virtual
environments while making the command available globally — exactly what's
needed here:

```bash
# install pipx itself once (Debian/Ubuntu: sudo apt install pipx)
# or: python3 -m pip install --user pipx && python3 -m pipx ensurepath

cd /path/to/sub-autotranslate-ai
pipx install .
```

`sub-autotranslate-tool` is now on your `PATH` (usually via
`~/.local/bin`) and works from any directory:

```bash
sub-autotranslate-tool --version
```

Upgrade after pulling new code:

```bash
pipx reinstall sub-autotranslate-tool
```

Uninstall:

```bash
pipx uninstall sub-autotranslate-tool
```

### Alternative: pip --user

```bash
python3 -m pip install --user .
```

This installs into `~/.local`; ensure `~/.local/bin` is on your `PATH`.

### Development install (venv)

```bash
python3 -m venv .venv
.venv/bin/pip install -e .
.venv/bin/sub-autotranslate-tool --help
```

## Configuration

Configuration is done by editing a plain-text TOML config file. Create it with:

```bash
sub-autotranslate-tool --init-config
```

Default location: `~/.config/sub-autotranslate-tool/config.toml`

```toml
api_key = ""                          # or use the OPENROUTER_API_KEY env var
model = "deepseek/deepseek-v4.1-flash"  # any model id from openrouter.ai/models
source_language = "English"
target_language = "Russian"
output_suffix = "ru"                  # movie.srt -> movie.ru.srt
batch_size = 40                       # lines per LLM request

send_speaker = true                   # send the ASS Name field with each line
context_overlap = 5                   # translated lines carried into the next batch
dedup = true                          # translate each distinct line once
cache = true                          # reuse translations across files and runs
glossary_auto = true                  # one pre-pass for names and terms
prompt_cache = "auto"                 # auto / on / off
skip_styles = []                      # styles to copy verbatim, globs allowed
sign_styles = []                      # force "on-screen text" treatment
song_styles = []                      # force "lyrics" treatment

temperature = 0.3
max_tokens = 8192
max_retries = 4
retry_delay = 2.0
proxy = ""                            # e.g. "http://127.0.0.1:8080" or "socks5://127.0.0.1:1080"
log_level = "info"                    # debug / info / warning / error / critical
prompt = """..."""                    # fully customizable translation prompt
```

The prompt is sent as the `system` message of every request. It supports the
placeholders `{source_language}`, `{target_language}`, `{glossary}` and
`{context}`; keep it free of anything batch-specific so providers can cache
it. ASS override tags may be written literally (`{\i1}`), single braces and
all.

The API key is resolved in this order: `--api-key` flag →
`OPENROUTER_API_KEY` environment variable → `api_key` in the config file.

### Glossary

Names and recurring terms are the usual source of inconsistency, because each
batch is translated on its own. Pin them once:

```bash
# first run generates the file, later runs read it
sub-autotranslate-tool season/ --glossary season/glossary.toml
```

The file is plain TOML and meant to be hand-corrected:

```toml
[terms]
"Ayanokoji" = "Аянокодзи"
"White Room" = "Белая комната"

[genders]
"Ayanokoji" = "m"
```

Use `--refresh-glossary` to rebuild it, or `--no-glossary` to skip the
pre-pass entirely.

### Series notes

A few sentences about the show do more for quality than any other setting:
who the characters are, their gender, and who addresses whom formally.

```bash
sub-autotranslate-tool season/ --context season/notes.txt
```

A `.subctx.txt` or `series.txt` file next to the input is picked up
automatically.

### Translation memory

Translations are stored in `~/.cache/sub-autotranslate-tool/`, keyed by model,
language pair, line type and a fingerprint of the prompt and glossary — so
editing the prompt invalidates old entries instead of silently reusing them.

```bash
sub-autotranslate-tool --cache-clear   # wipe it
sub-autotranslate-tool movie.ass --no-cache
```

### Proxy

If your network blocks direct access to OpenRouter (e.g. HTTP 403
"Access denied by security policy"), route API traffic through a proxy:

```toml
# in the config file
proxy = "http://127.0.0.1:8080"        # http/https proxy
proxy = "socks5://127.0.0.1:1080"      # SOCKS5 proxy
```

or via the command line:

```bash
sub-autotranslate-tool movie.srt --proxy socks5://127.0.0.1:1080
```

The standard `HTTPS_PROXY`/`HTTP_PROXY` environment variables are also
honored automatically.

## Usage

```bash
# Translate a single file (output: movie.ru.srt next to the input)
sub-autotranslate-tool movie.srt

# Translate into a specific output directory
sub-autotranslate-tool movie.ass -o ./translated

# Translate ALL subtitles in a directory
sub-autotranslate-tool /path/to/subtitles -o /path/to/output

# Override settings on the command line
sub-autotranslate-tool movie.srt --model anthropic/claude-sonnet-4 \
    --source-language English --target-language German

# Route requests through a proxy
sub-autotranslate-tool movie.srt --proxy http://127.0.0.1:8080

# A whole season with a shared glossary and series notes
sub-autotranslate-tool season/ --glossary season/glossary.toml \
    --context season/notes.txt

# Leave the fansub typesetting styles alone
sub-autotranslate-tool movie.ass --log-level debug   # see the styles in use
# then, in the config: skip_styles = ["signs", "OP*"]
```

## Logging

The tool logs to stderr with configurable levels
(`debug` / `info` / `warning` / `error` / `critical`), set via
`log_level` in the config file or the `--log-level` CLI flag (both tools).

- **`info`** (default) — high-level progress: files, model, configuration.
- **`debug`** — full diagnostics, including the **exact text sent to the LLM**
  (prompt + numbered subtitle lines) and the **raw model response** plus
  per-request token usage. Use this to verify the prompt behaves correctly
  and no extra content is wasting tokens:
  ```bash
  sub-autotranslate-tool movie.srt --log-level debug 2>debug.log
  ```
- **`warning`** — retries, malformed LLM responses, batches that had to be split.
- **`error`** — per-file failures that don't abort the run.
- **`critical`** — unrecoverable errors (e.g. missing API key).

## Extracting subtitles from MKV: `mkv-sub-extract`

The package ships a companion tool that extracts subtitle streams from MKV
files using **ffmpeg/ffprobe** (must be installed: `sudo apt install ffmpeg`).

It takes a directory, probes the **first** mkv to detect subtitle tracks
(SRT and ASS/SSA are supported), and extracts the chosen stream from every
mkv in the directory:

```bash
# List subtitle streams of the first mkv in the directory
mkv-sub-extract /path/to/mkv --list
# Subtitle streams in movie.mkv:
#   0:s:0  codec=subrip  (English)
#   0:s:1  codec=ass     (Русские)

# Extract the first subtitle stream (default, equals ffmpeg -map 0:s:0)
mkv-sub-extract /path/to/mkv

# Extract a specific stream (equals ffmpeg -map 0:s:1)
mkv-sub-extract /path/to/mkv --stream 1

# Custom output directory
mkv-sub-extract /path/to/mkv -o ./subtitles
```

The output extension is chosen by the codec of the selected stream
(`subrip` → `.srt`, `ass`/`ssa` → `.ass`), so extracted files can be fed
directly to `sub-autotranslate-tool`:

```bash
mkv-sub-extract ./videos -o ./subs
sub-autotranslate-tool ./subs -o ./subs-ru
```

## Project layout

```
sub_autotranslate_tool/
├── cli.py          # sub-autotranslate-tool: argument parsing, orchestration
├── config.py       # TOML config loading/creation
├── subtitles.py    # SRT and ASS/SSA parsing & re-rendering
├── markup.py       # markup/text separation, line classification, validation
├── glossary.py     # name & term glossary: collection, pre-pass, TOML file
├── cache.py        # on-disk translation memory
├── translator.py   # OpenRouter client, batching, repair, response parsing
├── log.py          # logging setup
└── mkvextract.py   # mkv-sub-extract: ffmpeg-based subtitle extraction from MKV

tests/              # pytest suite; run with: pytest
```

## Tests

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/pytest
```

The suite runs entirely offline — the OpenRouter transport is stubbed. Its
backbone is a round-trip guarantee: parsing a subtitle file and rendering it
back must reproduce the input exactly, which is what makes the markup
handling safe to change. If the real subtitle files in `test_files/` are
present, they are round-tripped too; otherwise those checks skip.
