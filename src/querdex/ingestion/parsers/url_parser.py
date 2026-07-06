from __future__ import annotations

import ipaddress
import socket
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener

from bs4 import BeautifulSoup

from querdex.schemas import Section

_MAX_RESPONSE_BYTES = 5 * 1024 * 1024
_USER_AGENT = "QuerdexBot/0.1"


def _validate_public_http_url(url: str) -> None:
    """Reject non-http(s) URLs and hosts that resolve to non-public addresses.

    Guards against SSRF: without this, indexing a URL could reach localhost
    services, private-network hosts, or cloud metadata endpoints.
    """
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        msg = f"URL parser expects http/https URL. Got: {url}"
        raise ValueError(msg)
    host = parsed.hostname
    if not host:
        msg = f"URL has no host: {url}"
        raise ValueError(msg)
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        infos = socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        msg = f"Cannot resolve host: {host}"
        raise ValueError(msg) from exc
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if not ip.is_global:
            msg = f"Refusing to fetch non-public address {ip} (host {host})"
            raise ValueError(msg)


class _SafeRedirectHandler(HTTPRedirectHandler):
    def redirect_request(self, req: Request, fp: Any, code: int, msg: str, headers: Any, newurl: str) -> Any:
        _validate_public_http_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _fetch_html(url: str, timeout: float = 10.0) -> str:
    _validate_public_http_url(url)
    opener = build_opener(_SafeRedirectHandler())
    req = Request(url, headers={"User-Agent": _USER_AGENT})
    with opener.open(req, timeout=timeout) as resp:  # noqa: S310
        body = resp.read(_MAX_RESPONSE_BYTES + 1)
    if len(body) > _MAX_RESPONSE_BYTES:
        msg = f"Response from {url} exceeds {_MAX_RESPONSE_BYTES} bytes; refusing to parse"
        raise ValueError(msg)
    return str(body.decode("utf-8", errors="ignore"))


class URLParser:
    source_format = "url"

    def parse(self, path: Path, doc_id: str) -> list[Section]:
        # `path` can be a plain text file containing a URL or a URL-like string path.
        raw = str(path)
        if path.exists() and path.is_file():
            raw = path.read_text(encoding="utf-8").strip()

        parsed = urlparse(raw)
        if parsed.scheme not in {"http", "https"}:
            msg = f"URL parser expects http/https URL. Got: {raw}"
            raise ValueError(msg)

        html = _fetch_html(raw)

        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()

        sections: list[Section] = []
        page_no = 1
        for block in soup.find_all(["h1", "h2", "h3", "h4", "p", "li"]):
            text = block.get_text(" ", strip=True)
            if not text:
                continue
            links = [a.get("href") for a in block.find_all("a") if a.get("href")]
            sections.append(
                Section(
                    section_id=f"sec_{len(sections) + 1:04d}",
                    doc_id=doc_id,
                    content=text,
                    page_number=page_no,
                    source_format=self.source_format,
                    metadata={
                        "url": raw,
                        "tag": block.name,
                        "links": links,
                    },
                )
            )
            page_no += 1

        if not sections:
            text = soup.get_text(" ", strip=True)
            if text:
                sections.append(
                    Section(
                        section_id="sec_0001",
                        doc_id=doc_id,
                        content=text,
                        page_number=1,
                        source_format=self.source_format,
                        metadata={"url": raw, "tag": "document"},
                    )
                )
        return sections
