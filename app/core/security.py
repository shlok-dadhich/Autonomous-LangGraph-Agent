"""Security helpers: SSRF guard, HTML sanitize, rate limit."""

from __future__ import annotations

import ipaddress
import re
from urllib.parse import urlparse

PRIVATE_NETS = [
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("::1/128"),
]

def is_private_url(url: str) -> bool:
    try:
        host = urlparse(url).hostname or ""
        ip = ipaddress.ip_address(host)
        return any(ip in net for net in PRIVATE_NETS)
    except ValueError:
        return False

def sanitize_html(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text)
