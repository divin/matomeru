"""SMTP mail building and sending for Matomeru digests.

Message construction is deterministic: headers come from the supplied
values, including an explicit ``Date`` header built from the caller's
``sent_at`` timestamp (no clock reads here). Sending runs the blocking
stdlib :mod:`smtplib` inside :func:`asyncio.to_thread`.

Retry logic belongs to orchestration (phase 5b), not here.
"""

from __future__ import annotations

import asyncio
import smtplib
from datetime import datetime
from email.message import EmailMessage
from email.utils import format_datetime


class MailError(Exception):
    """Raised when building or sending mail fails.

    Messages deliberately never contain the password.
    """


def digest_subject(source: str, report_date: str) -> str:
    """Build the deterministic digest email subject line."""
    return f"Matomeru {source} digest {report_date}"


def build_digest_message(
    *,
    sender: str,
    recipients: list[str],
    subject: str,
    body: str,
    sent_at: datetime,
) -> EmailMessage:
    """Build a plain-text digest email message.

    Args:
        sender: From address.
        recipients: To addresses (comma-joined into one To header).
        subject: Subject line (see :func:`digest_subject`).
        body: Plain-text body (the rendered digest).
        sent_at: Timestamp for the Date header (deterministic; use an
            aware datetime for an unambiguous header).

    Returns:
        The fully-formed :class:`email.message.EmailMessage` (no attachments).
    """
    message = EmailMessage()
    message["From"] = sender
    message["To"] = ", ".join(recipients)
    message["Subject"] = subject
    message["Date"] = format_datetime(sent_at)
    message.set_content(body)
    return message


def _send_sync(
    host: str,
    port: int,
    username: str | None,
    password: str | None,
    sender: str,
    recipients: list[str],
    message: EmailMessage,
) -> None:
    use_ssl = port == 465
    try:
        if use_ssl:
            server = smtplib.SMTP_SSL(host, port)
        else:
            server = smtplib.SMTP(host, port)
        server.ehlo()
        if not use_ssl and server.has_extn("starttls"):
            server.starttls()
            server.ehlo()
        if username is not None and username != "":
            server.login(username, password or "")
        server.send_message(message)
        server.quit()
    except Exception as e:
        raise MailError(f"SMTP send to {host}:{port} failed: {e}") from e


async def send_mail(
    host: str,
    port: int,
    username: str | None,
    password: str | None,
    sender: str,
    recipients: list[str],
    message: EmailMessage,
) -> None:
    """Send one email message over SMTP.

    Port 465 uses implicit TLS (:class:`smtplib.SMTP_SSL`); any other port
    uses plain :class:`smtplib.SMTP` with STARTTLS when the server supports
    it. Login happens only when a username is provided.

    Raises:
        MailError: On any SMTP failure (never leaking the password).
    """
    await asyncio.to_thread(_send_sync, host, port, username, password, sender, recipients, message)
