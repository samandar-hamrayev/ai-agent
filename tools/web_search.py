from __future__ import annotations

import re
from html import unescape
from urllib.parse import unquote

import httpx

from tools.base import BaseTool, ToolResult

_DDG_URL = "https://html.duckduckgo.com/html/"
_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
_MAX_RESULTS = 5
_TIMEOUT = 15


def _extract_url(href: str) -> str:
    """Pull the real URL out of DuckDuckGo's redirect wrapper."""
    match = re.search(r"uddg=([^&]+)", href)
    return unquote(match.group(1)) if match else href


def _clean_html(text: str) -> str:
    """Strip tags and unescape entities."""
    return unescape(re.sub(r"<[^>]+>", "", text)).strip()


def _parse_results(html: str, max_results: int = _MAX_RESULTS) -> list[dict]:
    """Extract title/url/snippet from DuckDuckGo HTML using regex."""
    results: list[dict] = []

    blocks = re.findall(
        r'<div[^>]*class="[^"]*result[^"]*"[^>]*>(.*?)</div>\s*(?=<div[^>]*class="[^"]*result|$)',
        html,
        re.DOTALL,
    )

    for block in blocks:
        if len(results) >= max_results:
            break

        title_match = re.search(
            r'<a[^>]*class="result__a"[^>]*href="([^"]*)"[^>]*>(.*?)</a>',
            block,
            re.DOTALL,
        )
        if not title_match:
            continue

        title = _clean_html(title_match.group(2))
        url = _extract_url(title_match.group(1))

        snippet_match = re.search(
            r'<a[^>]*class="result__snippet"[^>]*>(.*?)</a>',
            block,
            re.DOTALL,
        )
        snippet = _clean_html(snippet_match.group(1)) if snippet_match else ""

        if title and url:
            results.append({"title": title, "url": url, "snippet": snippet})

    return results


class WebSearchTool(BaseTool):
    """Searches the web via DuckDuckGo (no API key needed)."""

    name: str = "web_search"
    description: str = (
        "Searches the web for a given query and returns the top results "
        "with title, URL, and snippet. Use this to find current information, "
        "facts, prices, news, etc."
    )

    def run(self, query: str) -> ToolResult:
        if not query or not query.strip():
            return ToolResult(success=False, output=None, error="Empty search query provided.")

        try:
            with httpx.Client(
                headers={"User-Agent": _USER_AGENT},
                timeout=_TIMEOUT,
                follow_redirects=True,
            ) as client:
                response = client.post(_DDG_URL, data={"q": query.strip()})
                response.raise_for_status()

            results = _parse_results(response.text)

            if not results:
                return ToolResult(
                    success=True,
                    output={"query": query, "results": [], "message": "No results found."},
                )

            return ToolResult(
                success=True,
                output={"query": query, "results": results, "total": len(results)},
            )

        except httpx.TimeoutException:
            return ToolResult(success=False, output=None, error=f"Search timed out after {_TIMEOUT}s.")
        except httpx.HTTPStatusError as e:
            return ToolResult(success=False, output=None, error=f"HTTP error: {e.response.status_code}")
        except httpx.HTTPError as e:
            return ToolResult(success=False, output=None, error=f"Network error: {e}")

    def get_input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to look up on the web.",
                }
            },
            "required": ["query"],
        }
