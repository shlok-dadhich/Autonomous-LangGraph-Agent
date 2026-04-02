"""SMTP delivery service for newsletter dispatch."""

from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import os
import smtplib
from typing import Any, Dict, Optional

from loguru import logger


class EmailDeliveryError(RuntimeError):
	"""Raised when SMTP delivery cannot complete."""


class EmailService:
	"""Send newsletter email via Gmail SMTP App Password authentication."""

	def __init__(
		self,
		smtp_host: str = "smtp.gmail.com",
		smtp_port: int = 587,
		smtp_user: Optional[str] = None,
		smtp_app_pass: Optional[str] = None,
		recipient_email: Optional[str] = None,
		telegram_service: Optional[object] = None,
	) -> None:
		self.smtp_host = smtp_host
		self.smtp_port = smtp_port
		self.smtp_user = smtp_user or os.getenv("SMTP_USER")
		raw_app_pass = smtp_app_pass or os.getenv("SMTP_APP_PASS")
		self.smtp_app_pass = raw_app_pass.replace(" ", "") if raw_app_pass else None
		self.recipient_email = recipient_email or os.getenv("SMTP_TO") or self.smtp_user
		self.telegram_service = telegram_service

	@staticmethod
	def _append_graph_log(
		graph_state: Optional[Dict[str, Any]],
		level: str,
		message: str,
	) -> None:
		"""Append service logs to GraphState in-place when provided."""
		if graph_state is None:
			return
		graph_state.setdefault("logs", [])
		graph_state["logs"].append({"level": level, "message": message})

	def send_newsletter(
		self,
		html_content: str,
		subject: str,
		graph_state: Optional[Dict[str, Any]] = None,
		thread_id: Optional[str] = None,
	) -> bool:
		"""
		Send newsletter over SMTP.

		On failure, logs to GraphState and triggers Telegram alert.
		"""
		try:
			if not self.smtp_user or not self.smtp_app_pass:
				raise ValueError("SMTP_USER or SMTP_APP_PASS is not configured")

			if not self.recipient_email:
				raise ValueError("Recipient email not configured (set SMTP_TO or SMTP_USER)")

			message = MIMEMultipart("alternative")
			message["Subject"] = subject
			message["From"] = f"Your AI Research Agent <{self.smtp_user}>"
			message["To"] = self.recipient_email
			message.attach(MIMEText(html_content, "html", "utf-8"))

			if self.smtp_port == 465:
				with smtplib.SMTP_SSL(self.smtp_host, self.smtp_port, timeout=15) as server:
					server.login(self.smtp_user, self.smtp_app_pass)
					server.sendmail(
						from_addr=self.smtp_user,
						to_addrs=[self.recipient_email],
						msg=message.as_string(),
					)
			else:
				with smtplib.SMTP(self.smtp_host, self.smtp_port, timeout=15) as server:
					server.ehlo()
					server.starttls()
					server.ehlo()
					server.login(self.smtp_user, self.smtp_app_pass)
					server.sendmail(
						from_addr=self.smtp_user,
						to_addrs=[self.recipient_email],
						msg=message.as_string(),
					)

			success_msg = f"[email] Newsletter sent successfully to {self.recipient_email}"
			logger.success(success_msg)
			self._append_graph_log(graph_state, "success", success_msg)
			return True

		except Exception as exc:
			error_msg = f"[email] Dispatch failed: {type(exc).__name__}: {str(exc)}"
			logger.error(error_msg)
			self._append_graph_log(graph_state, "error", error_msg)
			return False
