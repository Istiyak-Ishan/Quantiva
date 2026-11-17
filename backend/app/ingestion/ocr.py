"""Optional OCR fallback for scanned PDF pages.

Design constraints for the Windows desktop build:

* **No Poppler dependency.**  PDF pages are rasterised with ``pypdfium2``,
  which ships prebuilt wheels for Windows.
* **Bundled Tesseract first.**  The executable is looked up inside the
  application vendor directories before the system ``PATH``.
* **Graceful degradation.**  Missing binaries, disabled OCR, or per-page
  failures never abort ingestion.  A warning is attached to the extraction
  metadata so the UI/job manager can explain why a scanned page is empty.
"""
from __future__ import annotations

import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Iterable

from app.ingestion.types import ExtractedBlock, ExtractionResult

__all__ = ["OCRService", "render_pdf_page_to_image"]

logger = logging.getLogger(__name__)

_DEFAULT_MIN_TEXT_CHARS = 30
_DEFAULT_DPI = 200


def _bundled_tesseract_candidates() -> list[Path]:
    """Return candidate bundled executable paths, highest priority first."""
    package_dir = Path(__file__).resolve().parent
    backend_dir = package_dir.parents[1]
    candidates = [
        package_dir / "vendor" / "tesseract" / "tesseract.exe",
        package_dir / "vendor" / "tesseract" / "tesseract",
        backend_dir / "vendor" / "tesseract" / "tesseract.exe",
        backend_dir / "vendor" / "tesseract" / "tesseract",
        backend_dir / "bin" / "tesseract" / "tesseract.exe",
        backend_dir / "bin" / "tesseract.exe",
        Path(sys.executable).resolve().parent / "tesseract.exe",
        Path(sys.executable).resolve().parent / "tesseract" / "tesseract.exe",
    ]
    for variable in ("QUANTIVA_TESSERACT_CMD", "TESSERACT_CMD"):
        configured = os.environ.get(variable)
        if configured:
            candidates.insert(0, Path(configured).expanduser())
    return candidates


def render_pdf_page_to_image(
    file_bytes: bytes,
    page_number: int,
    *,
    dpi: int = _DEFAULT_DPI,
):
    """Render one 1-based PDF page to a PIL RGB image using pypdfium2.

    Raises ``RuntimeError`` when pypdfium2 is unavailable and ``ValueError``
    when the page number is out of range.
    """
    if page_number < 1:
        raise ValueError("page_number is 1-indexed and must be >= 1")
    try:
        import pypdfium2 as pdfium
    except ImportError as exc:  # pragma: no cover - dependency error path
        raise RuntimeError("pypdfium2 is required for PDF page rasterisation") from exc

    document = pdfium.PdfDocument(file_bytes)
    try:
        if page_number > len(document):
            raise ValueError(
                f"Page {page_number} is out of range for a {len(document)}-page PDF"
            )
        page = document[page_number - 1]
        try:
            bitmap = page.render(scale=dpi / 72.0)
            try:
                return bitmap.to_pil()
            finally:
                close_bitmap = getattr(bitmap, "close", None)
                if callable(close_bitmap):
                    close_bitmap()
        finally:
            close_page = getattr(page, "close", None)
            if callable(close_page):
                close_page()
    finally:
        close_document = getattr(document, "close", None)
        if callable(close_document):
            close_document()


class OCRService:
    """Tesseract-backed OCR with graceful fallback metadata.

    Parameters
    ----------
    enabled:
        Hard switch from settings.  When false, OCR is never attempted.
    tesseract_cmd:
        Explicit executable path.  When omitted the bundled locations and
        then ``PATH`` are searched.
    min_text_chars:
        Pages marked by the PDF parser with fewer non-whitespace characters
        than this threshold are OCR candidates (only when they contain images).
    dpi:
        Rasterisation resolution.  200–300 DPI is a good accuracy/speed
        trade-off for document scans.
    languages:
        Tesseract language pack, e.g. ``"eng"`` or ``"eng+deu"``.
    """

    def __init__(
        self,
        *,
        enabled: bool = True,
        tesseract_cmd: str | Path | None = None,
        min_text_chars: int = _DEFAULT_MIN_TEXT_CHARS,
        dpi: int = _DEFAULT_DPI,
        languages: str = "eng",
        bundled_candidates: Iterable[str | Path] | None = None,
    ) -> None:
        self.enabled = bool(enabled)
        self.tesseract_cmd = Path(tesseract_cmd).expanduser() if tesseract_cmd else None
        self.min_text_chars = max(1, int(min_text_chars))
        self.dpi = max(72, int(dpi))
        self.languages = languages or "eng"
        self._bundled_candidates = (
            [Path(path).expanduser() for path in bundled_candidates]
            if bundled_candidates is not None
            else None
        )

    # ── Tesseract discovery ───────────────────────────────────────────────

    def _locate_tesseract(self) -> Path | str | None:
        """Bundled path first, then environment overrides, then system PATH."""
        if self.tesseract_cmd is not None:
            if self.tesseract_cmd.is_file():
                return self.tesseract_cmd
            if self.tesseract_cmd.is_dir():
                for executable_name in ("tesseract.exe", "tesseract"):
                    executable = self.tesseract_cmd / executable_name
                    if executable.is_file():
                        return executable
            # An explicitly configured path is still passed through so the
            # caller gets a clear OCR failure instead of a silent system fallback.
            return None
        candidates = (
            self._bundled_candidates
            if self._bundled_candidates is not None
            else _bundled_tesseract_candidates()
        )
        for candidate in candidates:
            if candidate.is_file():
                return candidate
            if candidate.is_dir():
                for executable_name in ("tesseract.exe", "tesseract"):
                    executable = candidate / executable_name
                    if executable.is_file():
                        return executable
        system_path = shutil.which("tesseract")
        return system_path or None

    @property
    def available(self) -> bool:
        if not self.enabled:
            return False
        try:
            import pytesseract  # noqa: F401
        except ImportError:
            return False
        return self._locate_tesseract() is not None

    def _configure_pytesseract(self) -> Any | None:
        try:
            import pytesseract
        except ImportError:
            return None
        executable = self._locate_tesseract()
        if executable is None:
            return None
        pytesseract.pytesseract.tesseract_cmd = str(executable)
        return pytesseract

    # ── OCR application ───────────────────────────────────────────────────

    def ocr_image(self, image: Any) -> str:
        """Run Tesseract on a PIL image and return stripped text."""
        pytesseract = self._configure_pytesseract()
        if pytesseract is None:
            raise RuntimeError("Tesseract OCR is unavailable")
        text = pytesseract.image_to_string(
            image,
            lang=self.languages,
            config="--psm 3",
        )
        return (text or "").strip()

    def apply(
        self,
        file_bytes: bytes,
        result: ExtractionResult,
    ) -> ExtractionResult:
        """OCR the pages marked in ``result.ocr_candidates`` in place.

        Returns the same result object for convenient chaining.  All failure
        modes are reported through ``result.warnings`` and ``result.metadata``.
        """
        candidates = sorted({page for page in result.ocr_candidates if page >= 1})
        ocr_metadata: dict[str, Any] = {
            "enabled": self.enabled,
            "tesseract_path": None,
            "pages_candidates": candidates,
            "pages_attempted": [],
            "pages_ocr": [],
            "pages_failed": [],
        }
        result.metadata["ocr"] = ocr_metadata

        if not candidates:
            ocr_metadata["status"] = "not_required"
            return result

        if not self.enabled:
            message = (
                "OCR is disabled; scanned page(s) "
                f"{candidates} may have little or no searchable text."
            )
            result.warnings.append(message)
            ocr_metadata["status"] = "disabled"
            return result

        executable = self._locate_tesseract()
        ocr_metadata["tesseract_path"] = str(executable) if executable else None
        pytesseract = self._configure_pytesseract()
        if executable is None or pytesseract is None:
            reason = (
                "Tesseract executable was not found"
                if executable is None
                else "pytesseract is not installed"
            )
            message = (
                f"OCR unavailable ({reason}); scanned page(s) {candidates} "
                "were left without text. Install/bundle Tesseract or set "
                "TESSERACT_CMD to enable OCR."
            )
            result.warnings.append(message)
            ocr_metadata["status"] = "unavailable"
            ocr_metadata["reason"] = reason
            logger.warning(message)
            return result

        existing_by_page: dict[int, list[ExtractedBlock]] = {}
        for block in result.blocks:
            if block.page_number is not None:
                existing_by_page.setdefault(block.page_number, []).append(block)

        extracted_pages: list[int] = []
        for page_number in candidates:
            ocr_metadata["pages_attempted"].append(page_number)
            try:
                image = render_pdf_page_to_image(
                    file_bytes,
                    page_number,
                    dpi=self.dpi,
                )
                try:
                    text = self.ocr_image(image)
                finally:
                    close_image = getattr(image, "close", None)
                    if callable(close_image):
                        close_image()
            except Exception as exc:
                reason = f"Page {page_number}: {exc}"
                result.warnings.append(f"OCR failed for scanned page {page_number}: {exc}")
                ocr_metadata["pages_failed"].append({"page": page_number, "error": str(exc)})
                logger.warning("OCR failed for page %s: %s", page_number, reason)
                continue

            if not text:
                ocr_metadata["pages_failed"].append(
                    {"page": page_number, "error": "Tesseract returned no text"}
                )
                continue

            page_blocks = existing_by_page.get(page_number, [])
            existing_characters = sum(
                len("".join(block.text.split())) for block in page_blocks
            )
            if existing_characters < self.min_text_chars:
                # Replace the unusably short text layer with OCR output, but
                # never discard an already-structured table block.
                stale = {
                    id(block)
                    for block in page_blocks
                    if block.block_type != "table"
                }
                result.blocks = [block for block in result.blocks if id(block) not in stale]

            result.blocks.append(
                ExtractedBlock(
                    text=text,
                    block_type="text",
                    page_number=page_number,
                    metadata={
                        "ocr": True,
                        "ocr_engine": "tesseract",
                        "ocr_language": self.languages,
                        "ocr_dpi": self.dpi,
                    },
                )
            )
            extracted_pages.append(page_number)

        ocr_metadata["pages_ocr"] = extracted_pages
        if extracted_pages:
            result.ocr_applied = True
            ocr_metadata["status"] = "applied"
            result.ocr_candidates = []
        else:
            ocr_metadata["status"] = "failed"
        return result
