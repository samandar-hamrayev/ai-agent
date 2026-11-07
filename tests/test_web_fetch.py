"""Tests for the web_fetch tool: SSRF guard, redirects, limits and HTML-to-text."""

from __future__ import annotations

import socket

import httpx
import pytest

from tools.web_fetch import UnsafeURLError, WebFetchTool, ensure_public_url, html_to_text

PAGE = """<html><head><title> Test Page </title><style>.x{color:red}</style></head>
<body><nav>Menu</nav><h1>Hello</h1><p>First &amp; second.</p><script>alert(1)</script>
<ul><li>one</li><li>two</li></ul></body></html>"""


def tool_for(handler) -> WebFetchTool:
    return WebFetchTool(transport=httpx.MockTransport(handler), resolve_check=False)


def test_html_to_text_strips_scripts_and_keeps_structure():
    title, text = html_to_text(PAGE)
    assert title == "Test Page"
    assert "alert" not in text and "color" not in text
    assert "Hello" in text and "First & second." in text
    assert text.splitlines().count("one") == 1


def test_fetch_html_page():
    tool = tool_for(lambda req: httpx.Response(200, html=PAGE))
    result = tool.run(url="https://example.com/")
    assert result.success
    assert result.output["title"] == "Test Page"
    assert "Hello" in result.output["content"]
    assert "next_start_index" not in result.output


def test_long_pages_are_chunked():
    body = "word " * 2000
    tool = tool_for(lambda req: httpx.Response(200, text=body, headers={"content-type": "text/plain"}))
    first = tool.run(url="https://example.com/long", max_chars=1000)
    assert len(first.output["content"]) == 1000
    second = tool.run(url="https://example.com/long", start_index=first.output["next_start_index"], max_chars=1000)
    assert second.output["content"] == body.strip()[1000:2000]


def test_redirects_are_followed_and_rechecked(monkeypatch):
    def handler(req):
        if req.url.path == "/start":
            return httpx.Response(302, headers={"location": "http://169.254.169.254/latest/meta-data"})
        return httpx.Response(200, text="secret")

    tool = WebFetchTool(transport=httpx.MockTransport(handler), resolve_check=True)
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda host, port: [(None, None, None, None, ("93.184.216.34" if host == "example.com" else host, port))],
    )
    result = tool.run(url="https://example.com/start")
    assert not result.success
    assert result.error.startswith("Blocked")


def test_too_many_redirects():
    tool = tool_for(lambda req: httpx.Response(302, headers={"location": "/again"}))
    result = tool.run(url="https://example.com/")
    assert not result.success and "redirects" in result.error


def test_binary_content_rejected():
    tool = tool_for(lambda req: httpx.Response(200, content=b"\x89PNG", headers={"content-type": "image/png"}))
    result = tool.run(url="https://example.com/a.png")
    assert not result.success and "Unsupported content type" in result.error


def test_oversized_page_rejected():
    big = b"a" * (2 * 1024 * 1024 + 10)
    tool = tool_for(lambda req: httpx.Response(200, content=big, headers={"content-type": "text/plain"}))
    result = tool.run(url="https://example.com/big")
    assert not result.success and "larger than" in result.error


def test_http_errors_reported():
    tool = tool_for(lambda req: httpx.Response(404))
    result = tool.run(url="https://example.com/missing")
    assert not result.success and "HTTP 404" in result.error


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "ftp://example.com/",
        "http://127.0.0.1:6379/",
        "http://localhost/",
        "http://10.0.0.5/",
        "http://192.168.1.1/admin",
        "http://169.254.169.254/latest/meta-data",
        "http://[::1]/",
        "http://user:pass@example.com/",
    ],
)
def test_ssrf_guard_blocks_internal_targets(url):
    with pytest.raises(UnsafeURLError):
        ensure_public_url(url)


def test_ssrf_guard_allows_public_address(monkeypatch):
    monkeypatch.setattr(socket, "getaddrinfo", lambda host, port: [(None, None, None, None, ("93.184.216.34", port))])
    ensure_public_url("https://example.com/page")


def test_schema_has_required_url():
    schema = WebFetchTool().to_schema()
    assert schema["name"] == "web_fetch"
    assert schema["input_schema"]["required"] == ["url"]
