# sub-autotranslate-tool

A Linux CLI tool for high-quality, context-aware AI translation of **SRT** and
**ASS/SSA** subtitle files using LLMs via [OpenRouter](https://openrouter.ai).

Default translation pair: **English → Russian**
Default model: `google/gemini-3.8-flash`

## How it works

1. Parses the subtitle file into blocks (index, timestamp, text). For ASS files
   only the `Text` field of `Dialogue` events is touched — headers, styles and
   timing are preserved byte-for-byte.
2. Sends text lines to the LLM in numbered batches, so the model sees many
   lines at once and can use the surrounding dialogue as context.
3. Reinserts the translated text into the original timestamp structure.

If a batch response can't be parsed, it is retried with exponential backoff
and, as a last resort, recursively split into smaller batches down to single
lines — so one malformed response never loses a subtitle.

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
model = "google/gemini-3.8-flash"     # any model id from openrouter.ai/models
source_language = "English"
target_language = "Russian"
output_suffix = "ru"                  # movie.srt -> movie.ru.srt
batch_size = 40                       # lines per LLM request
temperature = 0.3
max_tokens = 8192
max_retries = 4
retry_delay = 2.0
proxy = ""                            # e.g. "http://127.0.0.1:8080" or "socks5://127.0.0.1:1080"
prompt = """..."""                    # fully customizable translation prompt
```

The prompt template supports the placeholders `{source_language}`,
`{target_language}` and `{count}`.

The API key is resolved in this order: `--api-key` flag →
`OPENROUTER_API_KEY` environment variable → `api_key` in the config file.

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
```

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
├── translator.py   # OpenRouter client, batching, retries, response parsing
└── mkvextract.py   # mkv-sub-extract: ffmpeg-based subtitle extraction from MKV
```
