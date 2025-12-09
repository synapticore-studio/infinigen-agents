# Core Tools - Essential functionality only
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

# Simple dependency injection


@dataclass
class FileManager:
    """Essential file operations"""

    base_path: Path = Path(".")

    def save_json(self, data: Dict, path: Path) -> bool:
        try:
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
            return True
        except Exception:
            return False

    def load_json(self, path: Path) -> Optional[Dict]:
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception:
            return None


@dataclass
class Logger:
    """Essential logging"""

    name: str = "infinigen"

    def __post_init__(self):
        self.logger = logging.getLogger(self.name)

    def info(self, message: str):
        self.logger.info(message)

    def error(self, message: str):
        self.logger.error(message)


def get_file_manager() -> FileManager:
    return FileManager()


def get_logger() -> Logger:
    return Logger()


FileManagerDep = get_file_manager
LoggerDep = get_logger
