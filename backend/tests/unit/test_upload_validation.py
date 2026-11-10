from unittest.mock import patch

import pytest
from fastapi import HTTPException

from app.config import settings
from app.services.ingestion.upload_validation import _clamav_scan


class _FakeSocket:
    def __init__(self, response: bytes):
        self.response = response
        self.sent = b""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def sendall(self, payload: bytes):
        self.sent += payload

    def recv(self, _size: int) -> bytes:
        return self.response


def test_clamav_disabled_does_not_connect(monkeypatch):
    monkeypatch.setattr(settings, "CLAMAV_HOST", None)
    with patch("app.services.ingestion.upload_validation.socket.create_connection") as connect:
        _clamav_scan(b"clean")
    connect.assert_not_called()


def test_clamav_clean_stream_uses_instream(monkeypatch):
    monkeypatch.setattr(settings, "CLAMAV_HOST", "clamav")
    fake = _FakeSocket(b"stream: OK\0")
    with patch("app.services.ingestion.upload_validation.socket.create_connection", return_value=fake):
        _clamav_scan(b"clean")
    assert fake.sent.startswith(b"zINSTREAM\0")
    assert fake.sent.endswith(b"\x00\x00\x00\x00")


def test_clamav_detection_is_rejected(monkeypatch):
    monkeypatch.setattr(settings, "CLAMAV_HOST", "clamav")
    fake = _FakeSocket(b"stream: Eicar-Test-Signature FOUND\0")
    with patch("app.services.ingestion.upload_validation.socket.create_connection", return_value=fake):
        with pytest.raises(HTTPException) as error:
            _clamav_scan(b"malware")
    assert error.value.status_code == 400


def test_clamav_unavailable_fails_closed(monkeypatch):
    monkeypatch.setattr(settings, "CLAMAV_HOST", "clamav")
    with patch("app.services.ingestion.upload_validation.socket.create_connection", side_effect=OSError):
        with pytest.raises(HTTPException) as error:
            _clamav_scan(b"clean")
    assert error.value.status_code == 503
