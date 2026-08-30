"""SMTP email provider — wraps src/services/email_service logic via EmailProvider protocol."""

from __future__ import annotations

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from app.core.config import get_settings
from app.providers.email.base import EmailMessage, EmailResult


class SmtpEmailProvider:
    provider_name = "smtp"

    def __init__(self, smtp_host: str | None = None, smtp_port: int | None = None):
        s = get_settings()
        self.smtp_host = smtp_host or s.smtp_host
        self.smtp_port = smtp_port or s.smtp_port
        self.smtp_user = s.smtp_user
        self.smtp_app_pass = s.smtp_app_pass.get_secret_value() if s.smtp_app_pass else None

    def send(self, message: EmailMessage) -> EmailResult:
        if not self.smtp_user or not self.smtp_app_pass:
            raise ValueError("SMTP_USER or SMTP_APP_PASS not configured")
        recipient = message.to
        mimemsg = MIMEMultipart("alternative")
        mimemsg["Subject"] = message.subject
        mimemsg["From"] = f"Your AI Research Agent <{self.smtp_user}>"
        mimemsg["To"] = recipient
        mimemsg.attach(MIMEText(message.html, "html", "utf-8"))
        if message.text:
            mimemsg.attach(MIMEText(message.text, "plain", "utf-8"))

        app_pass = self.smtp_app_pass.replace(" ", "") if self.smtp_app_pass else ""
        # SMTP send
        if self.smtp_port == 465:
            with smtplib.SMTP_SSL(self.smtp_host, self.smtp_port, timeout=15) as server:
                server.login(self.smtp_user, app_pass)
                server.sendmail(self.smtp_user, [recipient], mimemsg.as_string())
        else:
            with smtplib.SMTP(self.smtp_host, self.smtp_port, timeout=15) as server:
                server.ehlo()
                server.starttls()
                server.ehlo()
                server.login(self.smtp_user, app_pass)
                server.sendmail(self.smtp_user, [recipient], mimemsg.as_string())
        return EmailResult(message_id=None, provider=self.provider_name, success=True)

    async def send_async(self, message: EmailMessage) -> EmailResult:
        return self.send(message)
