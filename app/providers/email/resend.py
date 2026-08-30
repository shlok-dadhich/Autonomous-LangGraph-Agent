"""Resend email provider + webhook event ingestion."""

from __future__ import annotations

import os

import requests

from app.providers.email.base import EmailMessage, EmailResult


class ResendEmailProvider:
    provider_name = "resend"

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.getenv("RESEND_API_KEY")

    def send(self, message: EmailMessage) -> EmailResult:
        if not self.api_key:
            raise ValueError("RESEND_API_KEY not configured")
        resp = requests.post(
            "https://api.resend.com/emails",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json={"from": "Your AI Brief <brief@example.com>", "to": [message.to], "subject": message.subject, "html": message.html},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        return EmailResult(message_id=data.get("id"), provider=self.provider_name, success=True, raw=data)

    async def send_async(self, message: EmailMessage) -> EmailResult:
        return self.send(message)


# Webhook event types from Resend (delivered/bounced/complained/opened/clicked etc.)
RESEND_EVENTS = {"email.sent", "email.delivered", "email.opened", "email.clicked", "email.bounced", "email.complained", "email.delivery_delayed"}


def ingest_resend_webhook(payload: dict) -> dict:
    """Normalize Resend webhook payload to DeliveryEvent."""
    event_type = payload.get("type", "unknown")
    data = payload.get("data", {})
    # map to internal event
    mapped = {
        "email.delivered": "delivered",
        "email.opened": "opened",
        "email.clicked": "clicked",
        "email.bounced": "bounced",
        "email.complained": "complained",
        "email.delivery_delayed": "delayed",
        "email.sent": "sent",
    }.get(event_type, event_type)
    return {"event_type": mapped, "payload": payload, "message_id": data.get("email_id") or data.get("id")}
