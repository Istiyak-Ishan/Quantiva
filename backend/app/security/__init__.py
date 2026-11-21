"""Local security boundaries used by the Quantiva desktop backend."""
from app.security.sandbox import (
    LocalPythonSandbox,
    SandboxResult,
    SandboxValidationError,
)
from app.security.credentials import (
    CredentialManager,
    get_credential_manager,
    reset_credential_manager,
    mask_api_key,
)
from app.security.local_api import (
    DESKTOP_CORS_ORIGINS,
    IPCTokenAuthMiddleware,
    LoopbackGuardMiddleware,
    configure_desktop_security,
)

__all__ = [
    "CredentialManager",
    "DESKTOP_CORS_ORIGINS",
    "IPCTokenAuthMiddleware",
    "LocalPythonSandbox",
    "LoopbackGuardMiddleware",
    "SandboxResult",
    "SandboxValidationError",
    "configure_desktop_security",
    "get_credential_manager",
    "mask_api_key",
    "reset_credential_manager",
]
