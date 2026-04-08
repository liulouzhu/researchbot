"""Configuration loading utilities."""

import json
import shutil
from pathlib import Path

import pydantic
from loguru import logger

from researchbot.config.schema import Config

# Global variable to store current config path (for multi-instance support)
_current_config_path: Path | None = None

# New default paths
_NEW_CONFIG_DIR = Path.home() / ".researchbot"
_NEW_CONFIG_FILE = _NEW_CONFIG_DIR / "config.json"


def set_config_path(path: Path) -> None:
    """Set the current config path (used to derive data directory)."""
    global _current_config_path
    _current_config_path = path


def _migrate_from_old_location() -> bool:
    """Migrate config from ~/.nanobot/ to ~/.researchbot/ if needed.

    Returns True if migration was performed.
    """
    _OLD_CONFIG_DIR = Path.home() / ".nanobot"
    if _NEW_CONFIG_DIR.exists():
        # New location already exists, no migration needed
        return False

    if not _OLD_CONFIG_DIR.exists():
        # Old location doesn't exist either, no migration
        return False

    # Perform migration
    try:
        shutil.copytree(_OLD_CONFIG_DIR, _NEW_CONFIG_DIR, dirs_exist_ok=True)
        logger.info(
            f"Migrated config from {_OLD_CONFIG_DIR} to {_NEW_CONFIG_DIR}. "
            f"The old location is still preserved as a backup."
        )
        return True
    except Exception as e:
        logger.warning(f"Failed to migrate config from {_OLD_CONFIG_DIR}: {e}")
        # Fall through - we'll use the old location as fallback
        return False


def get_config_path() -> Path:
    """Get the configuration file path.

    Uses ~/.researchbot/config.json by default.
    If the new location doesn't exist but old ~/.nanobot/config.json does,
    migrates automatically from the old location.
    """
    if _current_config_path:
        return _current_config_path

    # Check if new path exists
    if _NEW_CONFIG_FILE.exists():
        return _NEW_CONFIG_FILE

    # If new doesn't exist but old does, migrate
    if _migrate_from_old_location():
        return _NEW_CONFIG_FILE

    # Neither exists, return new default
    return _NEW_CONFIG_FILE


def load_config(config_path: Path | None = None) -> Config:
    """
    Load configuration from file or create default.

    Args:
        config_path: Optional path to config file. Uses default if not provided.

    Returns:
        Loaded configuration object.
    """
    path = config_path or get_config_path()

    if path.exists():
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            data = _migrate_config(data)
            return Config.model_validate(data)
        except (json.JSONDecodeError, ValueError, pydantic.ValidationError) as e:
            logger.warning(f"Failed to load config from {path}: {e}")
            logger.warning("Using default configuration.")

    return Config()


def save_config(config: Config, config_path: Path | None = None) -> None:
    """
    Save configuration to file.

    Args:
        config: Configuration to save.
        config_path: Optional path to save to. Uses default if not provided.
    """
    path = config_path or get_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    data = config.model_dump(mode="json", by_alias=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _migrate_config(data: dict) -> dict:
    """Migrate old config formats to current."""
    # Move tools.exec.restrictToWorkspace → tools.restrictToWorkspace
    tools = data.get("tools", {})
    exec_cfg = tools.get("exec", {})
    if "restrictToWorkspace" in exec_cfg and "restrictToWorkspace" not in tools:
        tools["restrictToWorkspace"] = exec_cfg.pop("restrictToWorkspace")
    return data
