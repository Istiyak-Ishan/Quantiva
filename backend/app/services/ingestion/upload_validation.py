"""Bounded reads, structural checks, and optional ClamAV malware scanning."""
import asyncio
from io import BytesIO
from pathlib import PurePath
import socket
import struct
from zipfile import BadZipFile, ZipFile

from fastapi import HTTPException, UploadFile

from app.config import settings

TYPES = {
    ".pdf": "application/pdf",
    ".csv": "text/csv",
    ".txt": "text/plain",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}


def _clamav_scan(data: bytes) -> None:
    """Scan bytes with clamd's INSTREAM protocol and fail closed on errors."""
    if not settings.CLAMAV_HOST:
        return
    try:
        with socket.create_connection(
            (settings.CLAMAV_HOST, settings.CLAMAV_PORT),
            timeout=settings.CLAMAV_TIMEOUT_SECONDS,
        ) as connection:
            connection.sendall(b"zINSTREAM\0")
            for offset in range(0, len(data), 64 * 1024):
                chunk = data[offset:offset + 64 * 1024]
                connection.sendall(struct.pack("!I", len(chunk)) + chunk)
            connection.sendall(struct.pack("!I", 0))
            response = connection.recv(4096).decode("utf-8", errors="replace")
    except OSError:
        raise HTTPException(status_code=503, detail="Malware scanner unavailable")
    if "FOUND" in response:
        raise HTTPException(status_code=400, detail="Document failed malware scanning")
    if "OK" not in response:
        raise HTTPException(status_code=503, detail="Malware scanner returned an invalid response")


async def read_validated_upload(file: UploadFile) -> tuple[bytes, str, str]:
    filename = (file.filename or "").replace("\\", "/").split("/")[-1]
    suffix = PurePath(filename).suffix.lower()
    mime = TYPES.get(suffix)
    if not mime or mime != file.content_type:
        raise HTTPException(status_code=400, detail="Filename and supported content type must match")
    data = bytearray()
    while block := await file.read(64 * 1024):
        if len(data) + len(block) > settings.MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="Document exceeds the server upload limit")
        data.extend(block)
    if not data:
        raise HTTPException(status_code=400, detail="Empty documents are not supported")
    if suffix == ".pdf" and not data.startswith(b"%PDF-"):
        raise HTTPException(status_code=400, detail="Invalid PDF signature")
    if suffix in {".csv", ".txt"}:
        try:
            text = data.decode("utf-8-sig")
            if "\x00" in text:
                raise ValueError("Binary content")
        except (UnicodeDecodeError, ValueError):
            raise HTTPException(status_code=400, detail="Text documents must contain valid UTF-8 text")
    if suffix in {".xlsx", ".docx"}:
        try:
            with ZipFile(BytesIO(data)) as archive:
                entries = archive.infolist()
                expected = "word/document.xml" if suffix == ".docx" else "xl/workbook.xml"
                if "[Content_Types].xml" not in archive.namelist() or expected not in archive.namelist():
                    raise ValueError("Wrong Office format")
                if len(entries) > 10000 or sum(i.file_size for i in entries) > settings.MAX_UPLOAD_BYTES * 8:
                    raise ValueError("Expanded archive too large")
                if any(i.flag_bits & 1 for i in entries):
                    raise ValueError("Encrypted archive")
        except (BadZipFile, ValueError):
            raise HTTPException(status_code=400, detail="Invalid or oversized Office document archive")
    await asyncio.to_thread(_clamav_scan, bytes(data))
    return bytes(data), filename, mime
