"""Runtime path helpers derived from the active config context."""

from __future__ import annotations

from pathlib import Path

from researchbot.config.loader import get_config_path
from researchbot.utils.helpers import ensure_dir

# New default paths
_NEW_CONFIG_DIR = Path.home() / ".researchbot"
_NEW_WORKSPACE = _NEW_CONFIG_DIR / "workspace"
_NEW_BRIDGE = _NEW_CONFIG_DIR / "bridge"


def get_data_dir() -> Path:
    """Return the instance-level runtime data directory."""
    return ensure_dir(get_config_path().parent)


def get_runtime_subdir(name: str) -> Path:
    """Return a named runtime subdirectory under the instance data dir."""
    return ensure_dir(get_data_dir() / name)


def get_media_dir(channel: str | None = None) -> Path:
    """Return the media directory, optionally namespaced per channel."""
    base = get_runtime_subdir("media")
    return ensure_dir(base / channel) if channel else base


def get_cron_dir() -> Path:
    """Return the cron storage directory."""
    return get_runtime_subdir("cron")


def get_logs_dir() -> Path:
    """Return the logs directory."""
    return get_runtime_subdir("logs")


def get_workspace_path(workspace: str | None = None) -> Path:
    """Resolve and ensure the agent workspace path."""
    if workspace:
        return ensure_dir(Path(workspace).expanduser())
    # Use workspace from config data dir if available, otherwise default
    config_parent = get_config_path().parent
    if config_parent.name == ".researchbot" and (config_parent / "workspace").exists():
        return ensure_dir(config_parent / "workspace")
    return ensure_dir(_NEW_WORKSPACE)


def is_default_workspace(workspace: str | Path | None) -> bool:
    """Return whether a workspace resolves to researchbot's default workspace path."""
    if workspace is not None:
        current = Path(workspace).expanduser().resolve(strict=False)
    else:
        current = get_workspace_path().resolve(strict=False)
    return current == _NEW_WORKSPACE.resolve(strict=False)


def get_cli_history_path() -> Path:
    """Return the shared CLI history file path."""
    return get_data_dir() / "history" / "cli_history"


def get_bridge_install_dir() -> Path:
    """Return the shared WhatsApp bridge installation directory."""
    return ensure_dir(get_data_dir() / "bridge")


def get_legacy_sessions_dir() -> Path:
    """Return the legacy global session directory used for migration fallback."""
    return _NEW_CONFIG_DIR / "sessions"
