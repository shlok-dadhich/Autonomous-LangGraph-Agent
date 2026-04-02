"""Telegram notification service for worker observability."""

import os
from typing import Optional

import requests
from loguru import logger


class TelegramAlertService:
	"""Send operational success/failure notifications to Telegram."""

	def __init__(
		self,
		bot_token: Optional[str] = None,
		chat_id: Optional[str] = None,
	) -> None:
		self.bot_token = bot_token or os.getenv("TELEGRAM_BOT_TOKEN")
		self.chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID")

	def send_status(self, message: str) -> bool:
		"""Send arbitrary alert message. Returns True on success, False otherwise."""
		if not self.bot_token or not self.chat_id:
			logger.warning("[telegram] Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID")
			return False

		url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
		payload = {
			"chat_id": self.chat_id,
			"text": message,
			"disable_web_page_preview": True,
		}

		try:
			response = requests.post(url, json=payload, timeout=15)
			response.raise_for_status()
			logger.info("[telegram] Alert sent successfully")
			return True
		except Exception as exc:
			logger.error(f"[telegram] Failed to send alert: {type(exc).__name__}: {str(exc)}")
			return False

	def send_alert(self, message: str) -> bool:
		"""Backward-compatible alias for status updates."""
		return self.send_status(message)

	def send_success_notification(self, delivered_items: int, thread_id: str) -> bool:
		"""Send weekly success heartbeat after email dispatch."""
		message = (
			f"🚀 AI Weekly Delivered: {delivered_items} items.\n"
			f"thread_id: {thread_id}"
		)
		return self.send_status(message)

	def send_error(
		self,
		error_message: str,
		thread_id: str,
		traceback_text: Optional[str] = None,
	) -> bool:
		"""Send immediate failure alert including debugging context."""
		message = (
			"Newsletter worker failure detected\n"
			f"thread_id: {thread_id}\n"
			f"error: {error_message}"
		)
		if traceback_text:
			message = f"{message}\ntraceback: {traceback_text[:2500]}"

		return self.send_status(message)

	def send_failure_notification(
		self,
		error_message: str,
		thread_id: str,
		traceback_text: Optional[str] = None,
	) -> bool:
		"""Backward-compatible alias for failure notifications."""
		return self.send_error(error_message, thread_id, traceback_text)


class AlertService(TelegramAlertService):
	"""Backward-compatible alias for existing imports."""


class TelegramBotService(TelegramAlertService):
	"""Backward-compatible alias for existing imports."""
