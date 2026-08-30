"""EmailProvider protocol."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass
class EmailMessage:
    to: str
    subject: str
    html: str
    text: str | None = None
    headers: dict | None = None


@dataclass
class EmailResult:
    message_id: str | None
    provider: str
    success: bool
    raw: dict | None = None


class EmailProvider(Protocol):
    provider_name: str

    def send(self, message: EmailMessage) -> EmailResult:
        ...

    async def send_async(self, message: EmailMessage) -> EmailResult:
        ...
