from __future__ import annotations

import ipaddress
import re
import socket
from html.parser import HTMLParser
from urllib.parse import urljoin, urlsplit

import httpx

from tools.base import BaseTool, ToolResult

_USER_AGENT = "Mozilla/5.0 (compatible; ai-agent/2.0; +https://github.com/samandar-hamrayev/ai-agent)"
_TIMEOUT = 15
_MAX_REDIRECTS = 5
_MAX_BYTES = 2 * 1024 * 1024
_DEFAULT_MAX_CHARS = 8_000
_TEXT_TYPES = ("text/html", "text/plain", "application/json", "application/xhtml+xml", "text/markdown")

# Content that never carries readable page text.
_SKIP_TAGS = {"script", "style", "noscript", "svg", "template", "iframe", "head"}
_BLOCK_TAGS = {
    "p", "div", "section", "article", "main", "header", "footer", "li", "ul", "ol", "tr", "table",
    "h1", "h2", "h3", "h4", "h5", "h6", "br", "hr", "pre", "blockquote", "nav", "aside",
}  # fmt: skip


class UnsafeURLError(ValueError):
    """The URL points somewhere the agent must not reach (internal network, bad scheme...)."""


def ensure_public_url(url: str) -> None:
    """Reject non-HTTP(S) URLs and hosts that resolve to private, loopback or reserved addresses.

    This is the SSRF guard: without it the model could be talked into fetching
    http://169.254.169.254/ (cloud metadata), http://localhost:6379/ or other internal services.
    Every redirect hop is checked again, because a public URL can redirect inward.
    """
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https"):
        raise UnsafeURLError(f"Only http/https URLs are allowed, got '{parts.scheme or 'none'}'.")
    if not parts.hostname:
        raise UnsafeURLError("URL has no host.")
    if parts.username or parts.password:
        raise UnsafeURLError("URLs with embedded credentials are not allowed.")
    try:
        infos = socket.getaddrinfo(parts.hostname, parts.port or (443 if parts.scheme == "https" else 80))
    except socket.gaierror as e:
        raise UnsafeURLError(f"Cannot resolve host '{parts.hostname}'.") from e
    for info in infos:
        address = ipaddress.ip_address(info[4][0])
        if not address.is_global or address.is_multicast:
            raise UnsafeURLError(f"Host '{parts.hostname}' resolves to a non-public address.")


class _TextExtractor(HTMLParser):
    """Turns HTML into readable plain text: drops scripts/styles, keeps block structure."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.title = ""
        self._parts: list[str] = []
        self._skip_depth = 0
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "title":
            self._in_title = True
        elif tag in _SKIP_TAGS:
            self._skip_depth += 1
        elif tag in _BLOCK_TAGS:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self._in_title = False
        elif tag in _SKIP_TAGS and self._skip_depth:
            self._skip_depth -= 1
        elif tag in _BLOCK_TAGS:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title += data
        elif not self._skip_depth:
            self._parts.append(data)

    def text(self) -> str:
        raw = "".join(self._parts)
        lines = (re.sub(r"[ \t\r\f\v]+", " ", line).strip() for line in raw.split("\n"))
        return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def html_to_text(html: str) -> tuple[str, str]:
    """Return (title, readable text) for an HTML document."""
    parser = _TextExtractor()
    parser.feed(html)
    parser.close()
    return parser.title.strip(), parser.text()


class WebFetchTool(BaseTool):
    """Fetches a public web page and returns its readable text, in pages of `max_chars`."""

    name: str = "web_fetch"
    description: str = (
        "Fetches a public web page (http/https) and returns its title and readable text. "
        "Use it after web_search to read a result in full. Long pages are returned in chunks: "
        "if 'next_start_index' is present, call again with start_index set to it to continue."
    )

    def __init__(self, transport: httpx.BaseTransport | None = None, resolve_check: bool = True) -> None:
        self._transport = transport  # injectable for tests
        self._resolve_check = resolve_check

    def run(self, url: str, start_index: int = 0, max_chars: int = _DEFAULT_MAX_CHARS) -> ToolResult:
        max_chars = max(500, min(max_chars, 20_000))
        start_index = max(0, start_index)
        try:
            final_url, content_type, body = self._fetch(url)
        except UnsafeURLError as e:
            return ToolResult(success=False, output=None, error=f"Blocked: {e}")
        except httpx.TimeoutException:
            return ToolResult(success=False, output=None, error=f"Timed out after {_TIMEOUT}s.")
        except httpx.HTTPStatusError as e:
            return ToolResult(success=False, output=None, error=f"HTTP {e.response.status_code} for {url}")
        except (httpx.HTTPError, ValueError) as e:
            return ToolResult(success=False, output=None, error=f"Fetch failed: {e}")

        if "html" in content_type:
            title, text = html_to_text(body)
        else:
            title, text = "", body.strip()

        chunk = text[start_index : start_index + max_chars]
        output: dict = {"url": final_url, "title": title, "content": chunk, "total_chars": len(text)}
        if start_index + max_chars < len(text):
            output["next_start_index"] = start_index + max_chars
        return ToolResult(success=True, output=output)

    def _fetch(self, url: str) -> tuple[str, str, str]:
        """GET with manual redirects so each hop passes the SSRF check; enforce type and size limits."""
        with httpx.Client(
            headers={"User-Agent": _USER_AGENT, "Accept": "text/html,text/plain;q=0.9,*/*;q=0.5"},
            timeout=_TIMEOUT,
            follow_redirects=False,
            transport=self._transport,
        ) as client:
            current = url
            for _ in range(_MAX_REDIRECTS + 1):
                if self._resolve_check:
                    ensure_public_url(current)
                else:
                    urlsplit(current)  # still require a parseable URL in tests
                with client.stream("GET", current) as response:
                    if response.is_redirect:
                        location = response.headers.get("location")
                        if not location:
                            raise ValueError("Redirect without a Location header.")
                        current = urljoin(current, location)
                        continue
                    response.raise_for_status()
                    content_type = response.headers.get("content-type", "").split(";")[0].strip().lower()
                    if content_type and not content_type.startswith(_TEXT_TYPES):
                        raise ValueError(f"Unsupported content type '{content_type}' (text pages only).")
                    body = bytearray()
                    for chunk in response.iter_bytes():
                        body.extend(chunk)
                        if len(body) > _MAX_BYTES:
                            raise ValueError(f"Page is larger than {_MAX_BYTES // (1024 * 1024)} MB.")
                    encoding = response.encoding or "utf-8"
                    return str(response.url), content_type or "text/plain", body.decode(encoding, errors="replace")
            raise ValueError(f"Too many redirects (>{_MAX_REDIRECTS}).")

    def get_input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "Absolute http(s) URL of the page to read."},
                "start_index": {
                    "type": "integer",
                    "minimum": 0,
                    "description": "Character offset to continue from (use next_start_index from a previous call).",
                },
                "max_chars": {
                    "type": "integer",
                    "minimum": 500,
                    "maximum": 20000,
                    "description": "Maximum characters to return (default 8000).",
                },
            },
            "required": ["url"],
            "additionalProperties": False,
        }
