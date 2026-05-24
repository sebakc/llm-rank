"""Self-update + version-check support.

Source of truth for the latest version is the `__version__` constant in
`llm_rank/__init__.py` on the default branch of the GitHub repo. We pin the
URL here rather than hitting the PyPI/index API so the tool stays installable
straight from git.
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Optional

import requests

from . import __version__, cache

log = logging.getLogger(__name__)

REPO = "sebakc/llm-rank"
VERSION_URL = f"https://raw.githubusercontent.com/{REPO}/main/llm_rank/__init__.py"
GIT_INSTALL_URL = f"git+https://github.com/{REPO}"
VERSION_CACHE_KEY = "self_version_check"
VERSION_CHECK_TTL_HOURS = 24.0
VERSION_RE = re.compile(r'__version__\s*=\s*["\']([^"\']+)["\']')


def _version_tuple(v: str) -> tuple[int, ...]:
    parts = []
    for chunk in v.split("."):
        m = re.match(r"(\d+)", chunk)
        parts.append(int(m.group(1)) if m else 0)
    return tuple(parts)


def fetch_latest_version() -> Optional[str]:
    """Pull the version string from the repo's __init__.py."""
    try:
        r = requests.get(VERSION_URL, timeout=8)
        r.raise_for_status()
    except requests.RequestException as e:
        log.debug("version check failed: %s", e)
        return None
    m = VERSION_RE.search(r.text)
    return m.group(1) if m else None


def check_for_update(force: bool = False) -> tuple[str | None, bool]:
    """Returns (latest_version_or_None, is_newer_than_installed).

    Cached for 24h via the shared cache module so we don't hammer GitHub.
    """
    def _loader():
        return {"version": fetch_latest_version(), "fetched_at": time.time()}

    payload = cache.get(VERSION_CACHE_KEY, _loader, VERSION_CHECK_TTL_HOURS, refresh=force)
    latest = (payload or {}).get("version")
    if latest is None:
        return None, False
    return latest, _version_tuple(latest) > _version_tuple(__version__)


def _detect_install_kind() -> str:
    """Best-effort: is this binary installed as a uv tool, pipx, or pip?"""
    home = str(Path.home())
    # `uv tool` installs land under ~/.local/share/uv/tools/
    exec_path = shutil.which("llm-rank") or ""
    if "uv/tools" in exec_path or exec_path.endswith("/.local/share/uv/tools/llm-rank/bin/llm-rank"):
        return "uv-tool"
    if "pipx" in exec_path or f"{home}/.local/pipx" in exec_path:
        return "pipx"
    return "pip"


def self_update_command() -> list[str]:
    """Build the command appropriate for the current install kind."""
    kind = _detect_install_kind()
    if kind == "uv-tool":
        return ["uv", "tool", "install", "--reinstall", GIT_INSTALL_URL]
    if kind == "pipx":
        return ["pipx", "install", "--force", GIT_INSTALL_URL]
    return ["pip", "install", "--upgrade", GIT_INSTALL_URL]


def run_self_update() -> tuple[bool, str]:
    """Shell out to the appropriate package manager. Returns (ok, output)."""
    cmd = self_update_command()
    if shutil.which(cmd[0]) is None:
        return False, f"{cmd[0]!r} not found on PATH"
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    except (subprocess.SubprocessError, OSError) as e:
        return False, str(e)
    ok = proc.returncode == 0
    return ok, (proc.stdout + proc.stderr).strip()
