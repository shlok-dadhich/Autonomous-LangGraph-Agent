# Fly.io Secrets

Set the required secrets before deploying:

```bash
fly secrets set TAVILY_API_KEY=your_tavily_key
fly secrets set GROQ_API_KEY=your_groq_key
fly secrets set SMTP_USER=your_email@gmail.com
fly secrets set SMTP_APP_PASS=your_16_character_gmail_app_password
fly secrets set TELEGRAM_BOT_TOKEN=your_telegram_bot_token
fly secrets set TELEGRAM_CHAT_ID=your_telegram_chat_id
```

If you want to set them in one command:

```bash
fly secrets set \
  TAVILY_API_KEY=your_tavily_key \
  GROQ_API_KEY=your_groq_key \
  SMTP_USER=your_email@gmail.com \
  SMTP_APP_PASS=your_16_character_gmail_app_password \
  TELEGRAM_BOT_TOKEN=your_telegram_bot_token \
  TELEGRAM_CHAT_ID=your_telegram_chat_id
```

Deploy after the secrets are set:

```bash
fly deploy
```
