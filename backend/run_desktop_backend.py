#!/usr/bin/env python
"""
backend/run_desktop_backend.py
────────────────────────────────────────────────────────────────────────────────
PyInstaller-safe entry point for the Quantiva desktop backend.

Usage (Tauri sidecar):
    quantiva-backend.exe --port 0 --token <ipc-secret> --parent-pid <tauri-pid>

Usage (developer CLI):
    python run_desktop_backend.py --port 8765 --token testsecret
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing
import os
import signal
import sys
import threading
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path


def _ensure_stdio() -> None:
    """Redirect None stdout/stderr to devnull (PyInstaller --noconsole)."""
    if sys.stdout is None:
        sys.stdout = open(os.devnull, "w", encoding="utf-8")  # noqa: SIM115
    if sys.stderr is None:
        sys.stderr = open(os.devnull, "w", encoding="utf-8")  # noqa: SIM115


def _setup_logging(log_dir: Path) -> None:
    """Configure rotating file logger under QUANTIVA_DATA_DIR/logs/."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "backend.log"

    handler = RotatingFileHandler(
        log_file,
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5,
        encoding="utf-8",
    )
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(handler)

    # Also log to stderr if it's a real stream.
    if sys.stderr and hasattr(sys.stderr, "fileno"):
        try:
            sys.stderr.fileno()
            stream_handler = logging.StreamHandler(sys.stderr)
            stream_handler.setFormatter(formatter)
            root.addHandler(stream_handler)
        except (OSError, ValueError):
            pass  # Redirected to devnull — skip.


def _parent_pid_watchdog(parent_pid: int, check_interval: float = 3.0) -> None:
    """Daemon thread: shut down when the Tauri parent process exits.

    Uses ``psutil.pid_exists`` for reliable cross-platform checks.
    Falls back to ``os.kill(pid, 0)`` if psutil is unavailable.
    """
    logger = logging.getLogger("quantiva.watchdog")
    logger.info("Parent-PID watchdog started (parent=%d)", parent_pid)

    try:
        import psutil  # noqa: PLC0415
        pid_alive = psutil.pid_exists
    except ImportError:
        def pid_alive(pid: int) -> bool:
            try:
                os.kill(pid, 0)
                return True
            except OSError:
                return False

    while True:
        time.sleep(check_interval)
        if not pid_alive(parent_pid):
            logger.warning(
                "Parent process %d exited — shutting down backend.", parent_pid
            )
            # Send SIGINT to ourselves so Uvicorn shuts down gracefully.
            if sys.platform == "win32":
                # On Windows os.kill with SIGINT sends CTRL_C_EVENT.
                os.kill(os.getpid(), signal.CTRL_C_EVENT)
            else:
                os.kill(os.getpid(), signal.SIGINT)
            break


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quantiva Desktop Backend",
        prog="quantiva-backend",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=0,
        help="TCP port to bind (0 = ephemeral OS-assigned port).",
    )
    parser.add_argument(
        "--token",
        type=str,
        required=True,
        help="IPC secret shared with the Tauri frontend.",
    )
    parser.add_argument(
        "--parent-pid",
        type=int,
        default=None,
        dest="parent_pid",
        help="PID of the Tauri host process for orphan shutdown.",
    )
    return parser.parse_args()


def main() -> None:
    _ensure_stdio()
    args = _parse_args()

    # Inject IPC token BEFORE importing app (settings read env at import).
    os.environ["QUANTIVA_IPC_TOKEN"] = args.token

    # Resolve data directory for log location.
    from app.config import settings  # noqa: PLC0415

    _setup_logging(settings.local_data_paths.logs)

    logger = logging.getLogger("quantiva.entrypoint")
    logger.info("Starting Quantiva desktop backend (port=%d)", args.port)

    # Start parent-PID watchdog daemon thread.
    if args.parent_pid is not None:
        watcher = threading.Thread(
            target=_parent_pid_watchdog,
            args=(args.parent_pid,),
            daemon=True,
            name="parent-pid-watchdog",
        )
        watcher.start()

    # Import app AFTER env setup.
    from app.main import app  # noqa: PLC0415

    import uvicorn  # noqa: PLC0415

    # Resolve ephemeral port: Uvicorn 0.30+ supports port=0 natively, but
    # older versions may not.  We bind a temporary socket to discover the
    # actual port and pass it explicitly for maximum compatibility.
    port = args.port
    if port == 0:
        import socket  # noqa: PLC0415

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]

    # Print the connection info line that Tauri's `Command` sidecar reads.
    # Flush immediately so the Tauri process sees it before any log noise.
    connection_line = f"QUANTIVA_BACKEND_READY port={port}"
    print(connection_line, flush=True)
    logger.info(connection_line)

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=port,
        workers=1,
        log_config=None,  # We manage our own logging.
    )


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
