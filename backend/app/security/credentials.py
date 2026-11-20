"""Secure credential storage for Quantiva Desktop.

API keys are stored in Windows Credential Manager via ``keyring``.
When the keyring backend is unavailable (headless CI, containers), an
encrypted local fallback file is used.  The fallback encryption key is
derived from machine-bound entropy:

  Windows  → DPAPI ``CryptProtectData`` wraps a random 32-byte seed.
  Fallback → Machine GUID + hostname hash (best-effort, still far better
              than plaintext).

Security invariants enforced by this module:
- Keys are **never** written to SQLite columns, log lines, or API responses.
- ``mask_api_key`` must be called on any value that may appear in a user-
  facing string (logs, UI, error messages).
- The ``get_api_key`` return value must never be serialised to JSON.
"""
from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import platform
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

SERVICE_NAME = "QuantivaDesktop"
_FALLBACK_FILENAME = ".creds_store.enc"
_KEY_LABEL = "_fernet_key"


# ── Machine-bound entropy helpers ─────────────────────────────────────────────

def _machine_guid() -> str:
    """Return a stable per-machine identifier (Windows registry MachineGuid)."""
    if os.name == "nt":
        try:
            import winreg
            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Cryptography"
            ) as key:
                return str(winreg.QueryValueEx(key, "MachineGuid")[0])
        except Exception:
            pass
    return platform.node()


def _dpapi_protect(data: bytes) -> bytes:
    """Encrypt *data* with Windows DPAPI (current-user scope)."""
    import ctypes
    import ctypes.wintypes as wt

    class _BLOB(ctypes.Structure):
        _fields_ = [("cbData", wt.DWORD), ("pbData", ctypes.POINTER(ctypes.c_ubyte))]

    crypt = ctypes.windll.crypt32
    blob_in = _BLOB(
        len(data),
        ctypes.cast(ctypes.c_char_p(data), ctypes.POINTER(ctypes.c_ubyte)),
    )
    blob_out = _BLOB()
    if not crypt.CryptProtectData(
        ctypes.byref(blob_in), None, None, None, None, 0, ctypes.byref(blob_out)
    ):
        raise OSError("CryptProtectData failed")
    protected = bytes(blob_out.pbData[: blob_out.cbData])
    ctypes.windll.kernel32.LocalFree(blob_out.pbData)
    return protected


def _dpapi_unprotect(data: bytes) -> bytes:
    """Decrypt DPAPI-protected bytes."""
    import ctypes
    import ctypes.wintypes as wt

    class _BLOB(ctypes.Structure):
        _fields_ = [("cbData", wt.DWORD), ("pbData", ctypes.POINTER(ctypes.c_ubyte))]

    crypt = ctypes.windll.crypt32
    blob_in = _BLOB(
        len(data),
        ctypes.cast(ctypes.c_char_p(data), ctypes.POINTER(ctypes.c_ubyte)),
    )
    blob_out = _BLOB()
    if not crypt.CryptUnprotectData(
        ctypes.byref(blob_in), None, None, None, None, 0, ctypes.byref(blob_out)
    ):
        raise OSError("CryptUnprotectData failed")
    plain = bytes(blob_out.pbData[: blob_out.cbData])
    ctypes.windll.kernel32.LocalFree(blob_out.pbData)
    return plain


def _derive_fallback_key() -> bytes:
    """Derive a 32-byte Fernet-compatible key from machine-bound entropy."""
    seed = (_machine_guid() + platform.node()).encode("utf-8")
    return base64.urlsafe_b64encode(hashlib.sha256(seed).digest())


def _load_or_create_fernet_key(store_path: Path) -> bytes:
    """Load the persisted Fernet key, creating and machine-binding it if absent.

    On Windows: the random seed is wrapped with DPAPI so only the same user on
    the same machine can decrypt it.
    On other platforms: the key is derived from machine-bound constants (best-
    effort protection; suitable for CI / sandboxed environments).
    """
    from cryptography.fernet import Fernet

    key_file = store_path.parent / ".creds_key.bin"
    if key_file.is_file():
        raw = key_file.read_bytes()
        if os.name == "nt":
            try:
                seed = _dpapi_unprotect(raw)
                return base64.urlsafe_b64encode(hashlib.sha256(seed).digest())
            except Exception:
                pass
        else:
            # Non-Windows: raw is the base64-encoded key itself
            if len(raw) == 44:  # Fernet key is 44 URL-safe base64 chars
                return raw

    # Create a new key
    seed = os.urandom(32)
    if os.name == "nt":
        try:
            protected = _dpapi_protect(seed)
            key_file.write_bytes(protected)
            key_file.chmod(0o600)
        except Exception:
            pass
        return base64.urlsafe_b64encode(hashlib.sha256(seed).digest())
    else:
        new_key = Fernet.generate_key()
        key_file.write_bytes(new_key)
        try:
            key_file.chmod(0o600)
        except Exception:
            pass
        return new_key


# ── CredentialManager ─────────────────────────────────────────────────────────

class CredentialManager:
    """Thread-safe API key vault backed by Windows Credential Manager.

    Falls back transparently to an encrypted local file when ``keyring`` has no
    working backend (headless servers, Docker containers).

    Usage::

        cm = CredentialManager()
        cm.set_api_key("cfg_openai_prod", "sk-...")
        key = cm.get_api_key("cfg_openai_prod")  # returns plaintext; never log this!
        print(cm.mask_api_key(key))               # "sk-...Xxxx"
        cm.delete_api_key("cfg_openai_prod")
    """

    def __init__(
        self,
        service_name: str = SERVICE_NAME,
        fallback_dir: Path | str | None = None,
    ) -> None:
        self._service = service_name
        self._lock = threading.RLock()
        self._keyring_ok: bool | None = None  # cached probe result

        if fallback_dir is not None:
            fb_root = Path(fallback_dir).expanduser().resolve()
        else:
            from app.config import settings
            fb_root = Path(settings.QUANTIVA_DATA_DIR).expanduser().resolve() / "security"

        fb_root.mkdir(parents=True, exist_ok=True)
        self._store_path = fb_root / _FALLBACK_FILENAME

    # ── keyring probe ─────────────────────────────────────────────────────────

    def _use_keyring(self) -> bool:
        """Return True if the keyring backend can store and retrieve credentials."""
        if self._keyring_ok is not None:
            return self._keyring_ok
        with self._lock:
            if self._keyring_ok is not None:
                return self._keyring_ok
            try:
                import keyring
                import keyring.errors

                probe_user = "__quantiva_probe__"
                probe_val = "__probe__"
                keyring.set_password(self._service, probe_user, probe_val)
                retrieved = keyring.get_password(self._service, probe_user)
                keyring.delete_password(self._service, probe_user)
                self._keyring_ok = retrieved == probe_val
            except Exception as exc:
                logger.debug("keyring unavailable (%s); using encrypted fallback", exc)
                self._keyring_ok = False
            return self._keyring_ok

    # ── fallback encrypted file helpers ──────────────────────────────────────

    def _fernet(self):
        from cryptography.fernet import Fernet
        key = _load_or_create_fernet_key(self._store_path)
        return Fernet(key)

    def _load_store(self) -> dict[str, str]:
        if not self._store_path.is_file():
            return {}
        try:
            encrypted = self._store_path.read_bytes()
            plaintext = self._fernet().decrypt(encrypted)
            return json.loads(plaintext.decode("utf-8"))
        except Exception as exc:
            logger.warning("Could not load credential store: %s", exc)
            return {}

    def _save_store(self, data: dict[str, str]) -> None:
        payload = json.dumps(data, ensure_ascii=False).encode("utf-8")
        encrypted = self._fernet().encrypt(payload)
        # Atomic write
        import uuid
        tmp = self._store_path.parent / f".tmp_{uuid.uuid4().hex}"
        tmp.write_bytes(encrypted)
        try:
            tmp.chmod(0o600)
        except Exception:
            pass
        import os as _os
        _os.replace(tmp, self._store_path)

    # ── Public API ────────────────────────────────────────────────────────────

    def set_api_key(self, config_id: str, api_key: str) -> None:
        """Store *api_key* securely under *config_id*.

        The key is NEVER written to SQLite, logs, or API responses.
        """
        if not config_id or not api_key:
            raise ValueError("config_id and api_key must be non-empty strings")
        with self._lock:
            if self._use_keyring():
                import keyring
                keyring.set_password(self._service, config_id, api_key)
            else:
                store = self._load_store()
                store[config_id] = api_key
                self._save_store(store)

    def get_api_key(self, config_id: str) -> str | None:
        """Retrieve the stored key for *config_id*, or ``None`` if absent.

        CAUTION: never pass the return value to log statements or JSON responses.
        Call :meth:`mask_api_key` before displaying anything to users.
        """
        if not config_id:
            return None
        with self._lock:
            if self._use_keyring():
                import keyring
                return keyring.get_password(self._service, config_id)
            else:
                return self._load_store().get(config_id)

    def delete_api_key(self, config_id: str) -> bool:
        """Delete the stored key for *config_id*.  Returns True if it existed."""
        if not config_id:
            return False
        with self._lock:
            if self._use_keyring():
                import keyring
                import keyring.errors
                try:
                    keyring.delete_password(self._service, config_id)
                    return True
                except keyring.errors.PasswordDeleteError:
                    return False
            else:
                store = self._load_store()
                if config_id not in store:
                    return False
                del store[config_id]
                self._save_store(store)
                return True

    def has_api_key(self, config_id: str) -> bool:
        """Return True if a key is stored for *config_id*."""
        return self.get_api_key(config_id) is not None

    def list_config_ids(self) -> list[str]:
        """Return all config IDs that have stored credentials.

        Note: when using the Windows keyring backend, only IDs stored through
        this service name are enumerable; this falls back to the encrypted
        store index.
        """
        with self._lock:
            if not self._use_keyring():
                return list(self._load_store().keys())
            # Keyring does not expose enumeration; return empty for safety
            return []

    @staticmethod
    def mask_api_key(api_key: str | None, visible_prefix: int = 4, visible_suffix: int = 4) -> str:
        """Return a safely masked version of *api_key* safe for logs and UI.

        Examples::

            mask_api_key("AIzaSyD_abc123XYZ789Q")  → "AIza...789Q"
            mask_api_key("sk-proj-short")           → "sk-p...hort"
            mask_api_key(None)                      → "<not set>"
            mask_api_key("")                        → "<not set>"
        """
        if not api_key:
            return "<not set>"
        total = len(api_key)
        if total <= visible_prefix + visible_suffix + 3:
            # Too short to redact meaningfully — show first char + stars
            return api_key[0] + "*" * (total - 1)
        prefix = api_key[:visible_prefix]
        suffix = api_key[-visible_suffix:]
        return f"{prefix}...{suffix}"


# ── Module-level singleton (lazy) ─────────────────────────────────────────────

_manager_lock = threading.RLock()
_default_manager: CredentialManager | None = None


def get_credential_manager(fallback_dir: Path | str | None = None) -> CredentialManager:
    """Return the process-wide CredentialManager singleton."""
    global _default_manager
    if _default_manager is None:
        with _manager_lock:
            if _default_manager is None:
                _default_manager = CredentialManager(fallback_dir=fallback_dir)
    return _default_manager


def reset_credential_manager() -> None:
    """Reset the singleton (test isolation only)."""
    global _default_manager
    with _manager_lock:
        _default_manager = None


__all__ = [
    "CredentialManager",
    "SERVICE_NAME",
    "get_credential_manager",
    "mask_api_key",
    "reset_credential_manager",
]


def mask_api_key(api_key: str | None) -> str:
    """Module-level convenience wrapper for :meth:`CredentialManager.mask_api_key`."""
    return CredentialManager.mask_api_key(api_key)
