"""Configuration loading and path resolution utilities.

Supports YAML config files with ``${VAR:default}`` placeholder syntax.
Environment variables always take precedence over config file values.

Example:
    >>> config = Config.load("config/data_config.yaml")
    >>> data_root = config["data"]["root"]
    >>> clip_model = config["clip"]["model_name"]
"""

import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml

logger = logging.getLogger(__name__)

# Regex to match ${VAR:default} or ${VAR} placeholders
_PLACEHOLDER_RE = re.compile(r"\$\{(\w+)(?::([^}]*))?\}")


class Config:
    """YAML configuration loader with environment variable resolution.

    Supports ``${ENV_VAR:default_value}`` placeholder syntax in YAML values.
    Environment variables always override config file values.

    Attributes:
        _data: The raw parsed configuration dictionary.

    Example:
        >>> config = Config.load("config/data_config.yaml")
        >>> root = config["data"]["root"]   # resolved from $DATA_ROOT or default
        >>> config.get("clip.model_name", "openai/clip-vit-base-patch32")
    """

    def __init__(self, data: Dict[str, Any]) -> None:
        """Initialise Config with a pre-resolved dictionary.

        Args:
            data: Resolved configuration dictionary.
        """
        self._data = data

    # ------------------------------------------------------------------
    # Class-level constructors
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, config_path: Union[str, Path]) -> "Config":
        """Load and resolve a YAML configuration file.

        Reads the YAML file, resolves all ``${VAR:default}`` placeholders
        against current environment variables, and returns a ``Config``
        instance.

        Args:
            config_path: Path to the YAML configuration file.

        Returns:
            Resolved ``Config`` instance.

        Raises:
            FileNotFoundError: If ``config_path`` does not exist.
            yaml.YAMLError: If the YAML is malformed.

        Example:
            >>> cfg = Config.load("config/data_config.yaml")
        """
        config_path = Path(config_path).resolve()
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)

        logger.debug(f"Loaded raw config from {config_path}")
        resolved = cls._resolve(raw)
        logger.info(f"Config loaded and resolved: {config_path.name}")
        return cls(resolved)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        """Create a Config from an already-resolved dictionary.

        Args:
            data: Configuration dictionary (placeholders already resolved).

        Returns:
            ``Config`` instance wrapping ``data``.
        """
        return cls(data)

    # ------------------------------------------------------------------
    # Accessor helpers
    # ------------------------------------------------------------------

    def __getitem__(self, key: str) -> Any:
        """Dictionary-style access to top-level keys.

        Args:
            key: Top-level config key.

        Returns:
            Value for ``key``.

        Raises:
            KeyError: If ``key`` is absent.
        """
        return self._data[key]

    def __contains__(self, key: str) -> bool:
        """Check membership of a top-level key."""
        return key in self._data

    def get(self, dotted_key: str, default: Any = None) -> Any:
        """Dot-notation access to nested config values.

        Args:
            dotted_key: Dot-separated path, e.g. ``"clip.model_name"``.
            default: Value returned when the key is absent.

        Returns:
            Resolved value or ``default``.

        Example:
            >>> cfg.get("clip.batch_size", 32)
            32
        """
        keys = dotted_key.split(".")
        node: Any = self._data
        for k in keys:
            if not isinstance(node, dict) or k not in node:
                return default
            node = node[k]
        return node

    def as_dict(self) -> Dict[str, Any]:
        """Return the underlying configuration dictionary.

        Returns:
            Raw resolved dictionary.
        """
        return self._data

    # ------------------------------------------------------------------
    # Internal resolution logic
    # ------------------------------------------------------------------

    @classmethod
    def _resolve(cls, obj: Any) -> Any:
        """Recursively resolve ``${VAR:default}`` placeholders.

        Args:
            obj: Python object (dict, list, str, or scalar).

        Returns:
            Object with all string placeholders resolved.
        """
        if isinstance(obj, dict):
            return {k: cls._resolve(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [cls._resolve(item) for item in obj]
        if isinstance(obj, str):
            return cls._resolve_string(obj)
        return obj

    @classmethod
    def _resolve_string(cls, value: str) -> str:
        """Resolve all placeholders within a single string value.

        Args:
            value: String potentially containing ``${VAR:default}`` tokens.

        Returns:
            String with placeholders replaced by env-var or default values.
        """

        def replacer(match: re.Match) -> str:
            var_name = match.group(1)
            default = match.group(2) if match.group(2) is not None else ""
            resolved = os.environ.get(var_name, default)
            if resolved != default:
                logger.debug(f"Env override: ${{{var_name}}} = {resolved!r}")
            return resolved

        return _PLACEHOLDER_RE.sub(replacer, value)


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------


def get_data_root(config: Optional[Config] = None) -> Path:
    """Resolve the data root directory from environment or config.

    Priority: ``DATA_ROOT`` env var > ``config["data"]["root"]`` > ``./data``.

    Args:
        config: Optional loaded ``Config`` instance.

    Returns:
        Resolved ``Path`` to the data root directory.
    """
    env_root = os.environ.get("DATA_ROOT")
    if env_root:
        return Path(env_root).resolve()
    if config is not None and "data" in config:
        return Path(config["data"]["root"]).resolve()
    default = Path("./data").resolve()
    logger.warning(f"DATA_ROOT not set; defaulting to {default}")
    return default


def load_config(config_path: Union[str, Path]) -> Config:
    """Shorthand for ``Config.load()``.

    Args:
        config_path: Path to YAML config file.

    Returns:
        Resolved ``Config`` instance.
    """
    return Config.load(config_path)
