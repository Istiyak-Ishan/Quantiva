# -*- mode: python ; coding: utf-8 -*-
"""
backend/quantiva-backend.spec
────────────────────────────────────────────────────────────────────────────────
PyInstaller spec for the Quantiva Desktop Backend (--onedir, Windows).

Usage:
    pyinstaller quantiva-backend.spec

Output:
    dist/quantiva-backend/quantiva-backend.exe   (Tauri sidecar)

Design Notes:
    • --onedir  → instant startup (< 1.5 s) instead of unpacking hundreds of
      megabytes of Torch/ONNX DLLs from %TEMP% on every launch.
    • console=False for release so no black terminal flashes behind Tauri UI.
      Override via  `QUANTIVA_BUILD_CONSOLE=1` env var for debugging.
    • CPU-only PyTorch is expected in the build venv; CUDA DLLs (> 2 GB)
      are explicitly excluded below as a safety net.
"""

import os
import sys
from pathlib import Path

from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_submodules,
    copy_metadata,
)

# ── Build configuration knobs ─────────────────────────────────────────────────

CONSOLE = bool(os.environ.get("QUANTIVA_BUILD_CONSOLE", ""))
DEBUG = bool(os.environ.get("QUANTIVA_BUILD_DEBUG", ""))

BACKEND_DIR = Path(SPECPATH)   # directory containing this .spec file
APP_DIR = BACKEND_DIR / "app"

# ── Hidden imports ────────────────────────────────────────────────────────────
# Packages that PyInstaller's static analysis cannot trace because they are
# imported lazily, behind try/except, or via entry-points / plugins.

hidden_imports = [
    # ── Uvicorn internals (plugin-style imports) ──────────────────────────
    "uvicorn.logging",
    "uvicorn.loops",
    "uvicorn.loops.auto",
    "uvicorn.loops.asyncio",
    "uvicorn.protocols",
    "uvicorn.protocols.http",
    "uvicorn.protocols.http.auto",
    "uvicorn.protocols.http.h11_impl",
    "uvicorn.protocols.http.httptools_impl",
    "uvicorn.protocols.websockets",
    "uvicorn.protocols.websockets.auto",
    "uvicorn.protocols.websockets.wsproto_impl",
    "uvicorn.lifespan",
    "uvicorn.lifespan.on",
    "uvicorn.lifespan.off",

    # ── SQLAlchemy / SQLite ───────────────────────────────────────────────
    "sqlalchemy.sql.default_comparator",
    "sqlalchemy.dialects.sqlite",
    "sqlite3",

    # ── Async SQLite (SQLAlchemy asyncio dialect) ─────────────────────────
    "aiosqlite",

    # ── Tokeniser extension registry (tiktoken) ──────────────────────────
    "tiktoken_ext.openai_public",
    "tiktoken_ext",

    # ── ML / Embedding models ────────────────────────────────────────────
    "sentence_transformers",
    "transformers",
    "scipy.special._cdflib",
    "torch",
    "torch.utils",

    # ── Document parsers ─────────────────────────────────────────────────
    "pypdfium2",
    "pypdfium2._helpers",
    "docx",
    "openpyxl",
    "fitz",                     # PyMuPDF

    # ── Web / API ────────────────────────────────────────────────────────
    "multipart",
    "python_multipart",
    "email.mime.multipart",

    # ── Windows credential store ─────────────────────────────────────────
    "keyring.backends",
    "keyring.backends.Windows",

    # ── Pydantic / settings ──────────────────────────────────────────────
    "pydantic",
    "pydantic_settings",
    "platformdirs",

    # ── SSE streaming ────────────────────────────────────────────────────
    "sse_starlette",

    # ── Misc runtime deps ────────────────────────────────────────────────
    "psutil",
    "cryptography",
    "PIL",
    "pandas",
    "rank_bm25",
    "tabulate",

    # ── Restricted sandbox ───────────────────────────────────────────────
    "RestrictedPython",

    # ── PII detection ────────────────────────────────────────────────────
    "presidio_analyzer",
    "presidio_anonymizer",

    # ── Observability ────────────────────────────────────────────────────
    "prometheus_client",

    # ── Provider SDKs (lazy-loaded behind interfaces) ────────────────────
    "google.genai",
    "openai",
    "anthropic",
    "httpx",
    "httpcore",
    "h11",
    "anyio",
    "sniffio",

    # ── Langchain text splitters ─────────────────────────────────────────
    "langchain_text_splitters",

    # ── Alembic (may be needed for auto-migration at startup) ────────────
    "alembic",

    # ── Standard library modules sometimes missed ────────────────────────
    "multiprocessing",
    "multiprocessing.freeze_support",
]

# Collect all submodules of packages that rely heavily on lazy/plugin loading.
for pkg in [
    "uvicorn",
    "starlette",
    "fastapi",
    "sqlalchemy",
    "pydantic",
    "keyring",
    "transformers",
]:
    hidden_imports += collect_submodules(pkg)


# ── Data files ────────────────────────────────────────────────────────────────
# Non-Python files that must be bundled (model configs, tokeniser vocabs, etc.)

datas = []

# Collect package metadata needed at runtime by importlib.metadata / pkg_resources.
for pkg_name in [
    "uvicorn",
    "fastapi",
    "starlette",
    "pydantic",
    "pydantic_settings",
    "sqlalchemy",
    "sentence_transformers",
    "transformers",
    "keyring",
    "pypdfium2",
    "tiktoken",
    "sse_starlette",
    "platformdirs",
    "presidio_analyzer",
    "presidio_anonymizer",
    "prometheus_client",
    "tabulate",
    "h11",
    "httpx",
    "httpcore",
    "anyio",
    "sniffio",
]:
    try:
        datas += copy_metadata(pkg_name)
    except Exception:
        pass  # Package not installed in build venv — skip.

# Collect data files for packages with embedded resources.
for pkg_name in [
    "sentence_transformers",
    "transformers",
    "pypdfium2",
    "tiktoken",
    "tiktoken_ext",
    "presidio_analyzer",
]:
    try:
        datas += collect_data_files(pkg_name)
    except Exception:
        pass

# Include the entire app package tree as data for any dynamic imports.
datas += [(str(APP_DIR), "app")]


# ── Excludes ──────────────────────────────────────────────────────────────────
# Strip bloat that never runs in the packaged desktop app.

excludes = [
    # GUI / plotting (not needed in a headless API server)
    "tkinter",
    "_tkinter",
    "Tkinter",
    "matplotlib",
    "matplotlib.tests",
    "mpl_toolkits",
    "pylab",

    # Interactive / notebook
    "IPython",
    "ipykernel",
    "ipywidgets",
    "jupyter",
    "jupyter_client",
    "jupyter_core",
    "notebook",
    "nbconvert",
    "nbformat",

    # Testing frameworks
    "pytest",
    "pytest_asyncio",
    "pytest_cov",
    "coverage",
    "unittest",
    "doctest",

    # Development tools
    "setuptools",
    "pip",
    "wheel",
    "distutils",

    # CUDA libraries (CPU-only build; saves > 2 GB)
    "torch.cuda",
    "torch.distributed",
    "torch.backends.cudnn",
    "triton",
    "apex",

    # Migration-era cloud services not needed in desktop
    "celery",
    "redis",
    "asyncpg",
    "boto3",
    "botocore",
    "s3transfer",
]


# ── Analysis ──────────────────────────────────────────────────────────────────

a = Analysis(
    [str(BACKEND_DIR / "run_desktop_backend.py")],
    pathex=[str(BACKEND_DIR)],
    binaries=[],
    datas=datas,
    hiddenimports=hidden_imports,
    hookspath=[str(BACKEND_DIR / "pyinstaller_hooks")],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
    optimize=0,
)

# ── Filter out CUDA / cuDNN / NCCL binaries that may still leak through ──────

CUDA_PATTERNS = (
    "cublas",
    "cudart",
    "cudnn",
    "cufft",
    "curand",
    "cusolver",
    "cusparse",
    "nccl",
    "nvrtc",
    "nvjitlink",
    "nvToolsExt",
)

a.binaries = [
    (name, path, typ)
    for name, path, typ in a.binaries
    if not any(pat.lower() in name.lower() for pat in CUDA_PATTERNS)
]


# ── PYZ (compressed Python archive) ──────────────────────────────────────────

pyz = PYZ(a.pure, a.zipped_data)


# ── EXE ───────────────────────────────────────────────────────────────────────

exe = EXE(
    pyz,
    a.scripts,
    [],                 # --onedir: no bundled binaries in EXE itself
    exclude_binaries=True,
    name="quantiva-backend",
    debug=DEBUG,
    bootloader_ignore_signals=False,
    strip=False,        # Windows PE doesn't benefit from strip
    upx=False,          # UPX on Windows can trigger AV false positives
    console=CONSOLE,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(BACKEND_DIR.parent / "frontend" / "src-tauri" / "icons" / "icon.ico")
         if (BACKEND_DIR.parent / "frontend" / "src-tauri" / "icons" / "icon.ico").exists()
         else None,
    version_info=None,
)


# ── COLLECT (--onedir output) ────────────────────────────────────────────────

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="quantiva-backend",
)
