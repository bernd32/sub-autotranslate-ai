"""LLM translation backend (OpenRouter) with batching and retries.

Subtitles are sent to the model in numbered batches so that many lines can
be translated in a single request with full mutual context. The model is
required to answer with the same numbering; if the response cannot be
parsed into exactly the right number of items, the batch is retried and,
as a last resort, recursively split into smaller halves down to single
lines so one malformed response never loses a subtitle.
"""

from __future__ import annotations

import logging
import re
import sys
import time
from dataclasses import dataclass

import requests

from .config import Config

API_URL = "https://openrouter.ai/api/v1/chat/completions"

logger = logging.getLogger(__name__)


class TranslationError(RuntimeError):
    pass


@dataclass
class UsageStats:
    requests: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost: float = 0.0  # USD, taken from OpenRouter's usage.cost field


class Translator:
    def __init__(self, config: Config, api_key: str) -> None:
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

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def translate_all(
        self, texts: list[str], progress_label: str = ""
    ) -> list[str]:
        """Translate a list of subtitle texts, preserving order and count."""
        batch_size = max(1, self.config.batch_size)
        results: list[str | None] = [None] * len(texts)
        total = len(texts)

        for start in range(0, total, batch_size):
            chunk = texts[start : start + batch_size]
            translated = self._translate_chunk(chunk)
            results[start : start + len(chunk)] = translated
            done = min(start + len(chunk), total)
            print(
                f"\r{progress_label}{done}/{total} lines translated",
                end="",
                file=sys.stderr,
                flush=True,
            )
        if total:
            print(file=sys.stderr)

        return [r if r is not None else t for r, t in zip(results, texts)]

    # ------------------------------------------------------------------
    # Batching / retries
    # ------------------------------------------------------------------

    def _translate_chunk(self, chunk: list[str]) -> list[str]:
        """Translate one chunk, splitting recursively on persistent failure."""
        if len(chunk) == 1:
            # Single line: retry, then give up keeping the original text.
            for attempt in range(self.config.max_retries + 1):
                try:
                    return self._translate_batch(chunk)
                except TranslationError as exc:
                    self._wait(attempt, str(exc))
            logger.warning(
                "giving up on a line; keeping original text: %r", chunk[0][:80]
            )
            return list(chunk)

        for attempt in range(self.config.max_retries + 1):
            try:
                return self._translate_batch(chunk)
            except TranslationError as exc:
                self._wait(attempt, str(exc))

        # Persistent failure: split the batch and translate halves separately.
        mid = len(chunk) // 2
        logger.warning(
            "batch of %d failed repeatedly; splitting into smaller batches",
            len(chunk),
        )
        return self._translate_chunk(chunk[:mid]) + self._translate_chunk(chunk[mid:])

    def _translate_batch(self, lines: list[str]) -> list[str]:
        """One API round-trip for a batch of lines."""
        user_content = self._build_user_message(lines)
        # The exact text sent to the LLM: inspect prompt behavior and
        # make sure no extra content is wasting tokens.
        logger.debug(
            "LLM request (model=%s, %d lines):\n%s",
            self.config.model,
            len(lines),
            user_content,
        )
        payload = {
            "model": self.config.model,
            "messages": [{"role": "user", "content": user_content}],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            # OpenRouter unified reasoning control: explicitly disable
            # reasoning unless enabled in the config. Some reasoning models
            # (e.g. DeepSeek) return null content otherwise.
            "reasoning": {"enabled": self.config.enable_reasoning},
        }

        try:
            resp = self.session.post(API_URL, json=payload, timeout=180)
        except requests.RequestException as exc:
            raise TranslationError(f"network error: {exc}") from exc

        if resp.status_code != 200:
            detail = resp.text[:300]
            raise TranslationError(f"HTTP {resp.status_code}: {detail}")

        try:
            data = resp.json()
            choice = data["choices"][0]
            message = choice.get("message") or {}
            content = message.get("content")
        except (ValueError, KeyError, IndexError) as exc:
            raise TranslationError(f"malformed API response: {exc}") from exc

        if not isinstance(content, str) or not content.strip():
            # Some models return null content (e.g. reasoning-only output or a
            # provider-side refusal); treat it as a retryable translation error.
            finish_reason = choice.get("finish_reason", "?")
            raise TranslationError(
                f"empty response content (finish_reason={finish_reason})"
            )

        self.stats.requests += 1
        usage = data.get("usage") or {}
        self.stats.prompt_tokens += usage.get("prompt_tokens", 0)
        self.stats.completion_tokens += usage.get("completion_tokens", 0)
        cost = usage.get("cost")
        if isinstance(cost, (int, float)):
            self.stats.cost += float(cost)
        logger.debug("LLM response usage: %s", usage)
        logger.debug("LLM response content:\n%s", content)

        parsed = parse_numbered_response(content, len(lines))
        if parsed is None:
            logger.warning(
                "could not parse %d numbered items from the LLM response",
                len(lines),
            )
            raise TranslationError(
                f"could not parse {len(lines)} numbered items from the response"
            )
        return parsed

    def _build_user_message(self, lines: list[str]) -> str:
        prompt = self.config.prompt.format(
            source_language=self.config.source_language,
            target_language=self.config.target_language,
            count=len(lines),
        )
        numbered = "\n".join(f"[{i}] {line}" for i, line in enumerate(lines, 1))
        return f"{prompt}\n\n{numbered}"

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

_ITEM_START_RE = re.compile(r"^\s*\[?\s*(\d{1,4})\s*\]?\s*[.:)\-]?\s?(.*)$")


def parse_numbered_response(content: str, expected: int) -> list[str] | None:
    """Parse a numbered model response into exactly *expected* items.

    A new item starts at a line like "[3] text" / "3. text" / "3) text".
    Lines that don't start a new number are treated as continuation of the
    current item (preserving in-subtitle line breaks). Returns None if the
    numbering is unusable.
    """
    items: dict[int, list[str]] = {}
    current: int | None = None

    for raw_line in content.split("\n"):
        m = _ITEM_START_RE.match(raw_line)
        if m:
            n = int(m.group(1))
            if 1 <= n <= expected and n not in items:
                current = n
                items[n] = [m.group(2)]
                continue
        if current is not None:
            items[current].append(raw_line)

    if len(items) != expected or sorted(items) != list(range(1, expected + 1)):
        return None

    result = []
    for n in range(1, expected + 1):
        text = "\n".join(items[n]).strip("\n")
        if not text.strip():
            return None
        result.append(text)
    return result
