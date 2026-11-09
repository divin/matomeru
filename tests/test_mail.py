"""Tests for deterministic mail building and SMTP sending (no network)."""

from __future__ import annotations

from datetime import datetime, timezone
from email.message import EmailMessage
from typing import Any

import pytest

from matomeru.mail import MailError, build_digest_message, digest_subject, send_mail

SENT_AT = datetime(2026, 11, 9, 12, 0, 0, tzinfo=timezone.utc)
BODY = "# Reddit digest — 2026-11-09\n\nNo posts matched the configured subreddits in this window.\n"


def make_message() -> EmailMessage:
    return build_digest_message(
        sender="matomeru@example.com",
        recipients=["a@example.com", "b@example.com"],
        subject=digest_subject("reddit", "2026-11-09"),
        body=BODY,
        sent_at=SENT_AT,
    )


class FakeSMTP:
    """Recording fake standing in for smtplib.SMTP / SMTP_SSL."""

    instances: list["FakeSMTP"] = []

    def __init__(self, host: str, port: int, *, starttls_supported: bool = True, fail_on: str | None = None) -> None:
        self.host = host
        self.port = port
        self.calls: list[tuple[str, tuple[Any, ...]]] = []
        self._starttls_supported = starttls_supported
        self._fail_on = fail_on
        FakeSMTP.instances.append(self)

    def _record(self, name: str, *args: Any) -> None:
        self.calls.append((name, args))
        if self._fail_on == name:
            raise RuntimeError("relay denied")

    def ehlo(self) -> None:
        self._record("ehlo")

    def has_extn(self, ext: str) -> bool:
        return self._starttls_supported if ext == "starttls" else False

    def starttls(self) -> None:
        self._record("starttls")

    def login(self, username: str, password: str) -> None:
        self._record("login", username, password)

    def send_message(self, message: EmailMessage) -> None:
        self._record("send_message", message)

    def quit(self) -> None:
        self._record("quit")


class TestBuildDigestMessage:
    """Tests for build_digest_message and digest_subject."""

    def test_headers(self) -> None:
        message = make_message()
        assert message["From"] == "matomeru@example.com"
        assert message["To"] == "a@example.com, b@example.com"
        assert message["Subject"] == "Matomeru reddit digest 2026-11-09"

    def test_deterministic_subject(self) -> None:
        assert digest_subject("reddit", "2026-11-09") == "Matomeru reddit digest 2026-11-09"
        assert digest_subject("arxiv", "2026-11-09") == "Matomeru arxiv digest 2026-11-09"

    def test_deterministic_date_header(self) -> None:
        first = make_message()
        second = build_digest_message(
            sender="matomeru@example.com",
            recipients=["a@example.com"],
            subject="s",
            body=BODY,
            sent_at=SENT_AT,
        )
        assert first["Date"] == second["Date"] == "Mon, 09 Nov 2026 12:00:00 +0000"

    def test_plain_text_body_verbatim(self) -> None:
        message = make_message()
        assert message.get_content_type() == "text/plain"
        assert message.get_content().strip() == BODY.strip()

    def test_no_attachments(self) -> None:
        message = make_message()
        assert not list(message.iter_attachments())


class TestSendMail:
    """Tests for send_mail against faked smtplib classes."""

    pytestmark = pytest.mark.asyncio

    async def _run(self, monkeypatch: pytest.MonkeyPatch, target: type[FakeSMTP], port: int, **kwargs: Any) -> FakeSMTP:
        monkeypatch.setattr("smtplib.SMTP", target, raising=True)
        monkeypatch.setattr("smtplib.SMTP_SSL", target, raising=True)
        await send_mail(
            host="smtp.example.com",
            port=port,
            username=kwargs.get("username"),
            password=kwargs.get("password"),
            sender="matomeru@example.com",
            recipients=["a@example.com"],
            message=make_message(),
        )
        assert len(target.instances) == 1
        return target.instances[0]

    async def test_port_465_uses_ssl_without_starttls(self, monkeypatch: pytest.MonkeyPatch) -> None:
        FakeSMTP.instances = []
        server = await self._run(monkeypatch, FakeSMTP, 465, username="user", password="pw")
        assert server.port == 465
        names = [c[0] for c in server.calls]
        assert names == ["ehlo", "login", "send_message", "quit"]
        assert "starttls" not in names

    async def test_port_587_uses_starttls(self, monkeypatch: pytest.MonkeyPatch) -> None:
        FakeSMTP.instances = []
        server = await self._run(monkeypatch, FakeSMTP, 587, username="user", password="pw")
        names = [c[0] for c in server.calls]
        assert names == ["ehlo", "starttls", "ehlo", "login", "send_message", "quit"]

    async def test_starttls_skipped_when_unsupported(self, monkeypatch: pytest.MonkeyPatch) -> None:
        FakeSMTP.instances = []

        class NoStarttls(FakeSMTP):
            def __init__(self, host: str, port: int) -> None:
                super().__init__(host, port, starttls_supported=False)

        server = await self._run(monkeypatch, NoStarttls, 25, username=None, password=None)
        names = [c[0] for c in server.calls]
        assert names == ["ehlo", "send_message", "quit"]

    async def test_login_only_with_username(self, monkeypatch: pytest.MonkeyPatch) -> None:
        FakeSMTP.instances = []
        server = await self._run(monkeypatch, FakeSMTP, 587, username=None, password="pw")
        assert "login" not in [c[0] for c in server.calls]

    async def test_login_skipped_for_empty_username(self, monkeypatch: pytest.MonkeyPatch) -> None:
        FakeSMTP.instances = []
        server = await self._run(monkeypatch, FakeSMTP, 587, username="", password="pw")
        assert "login" not in [c[0] for c in server.calls]

    async def test_login_receives_credentials(self, monkeypatch: pytest.MonkeyPatch) -> None:
        FakeSMTP.instances = []
        server = await self._run(monkeypatch, FakeSMTP, 587, username="user", password="pw")
        assert ("login", ("user", "pw")) in server.calls

    async def test_message_passed_to_send_message(self, monkeypatch: pytest.MonkeyPatch) -> None:
        FakeSMTP.instances = []
        server = await self._run(monkeypatch, FakeSMTP, 465, username="user", password="pw")
        sent = server.calls[[c[0] for c in server.calls].index("send_message")][1][0]
        assert sent["Subject"] == "Matomeru reddit digest 2026-11-09"

    async def test_exception_wrapped_as_mail_error_without_password(self, monkeypatch: pytest.MonkeyPatch) -> None:
        FakeSMTP.instances = []

        class Failing(FakeSMTP):
            def __init__(self, host: str, port: int) -> None:
                super().__init__(host, port, fail_on="send_message")

        monkeypatch.setattr("smtplib.SMTP", Failing, raising=True)
        monkeypatch.setattr("smtplib.SMTP_SSL", Failing, raising=True)
        with pytest.raises(MailError) as exc_info:
            await send_mail(
                host="smtp.example.com",
                port=587,
                username="user",
                password="super-secret-password",
                sender="matomeru@example.com",
                recipients=["a@example.com"],
                message=make_message(),
            )
        assert "super-secret-password" not in str(exc_info.value)
        assert "relay denied" in str(exc_info.value)


HTML_BODY = "<html><body><h1>Reddit digest</h1><p>Hello &amp; goodbye</p></body></html>"


class TestBuildDigestMessageHtml:
    """Tests for the optional html_body alternative part."""

    def test_html_body_produces_multipart_alternative(self) -> None:
        message = build_digest_message(
            sender="matomeru@example.com",
            recipients=["a@example.com"],
            subject="s",
            body=BODY,
            sent_at=SENT_AT,
            html_body=HTML_BODY,
        )
        assert message.get_content_type() == "multipart/alternative"
        parts = list(message.iter_parts())
        assert [part.get_content_type() for part in parts] == ["text/plain", "text/html"]
        assert parts[0].get_content().strip() == BODY.strip()
        assert parts[1].get_content().strip() == HTML_BODY.strip()

    def test_without_html_body_unchanged(self) -> None:
        message = make_message()
        assert message.get_content_type() == "text/plain"
        assert message.get_content().strip() == BODY.strip()
        assert list(message.iter_parts()) == []
        assert not list(message.iter_attachments())
