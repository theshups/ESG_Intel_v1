"""src/exception.py — Custom exception with script + line capture."""
import sys, traceback
from src.logger import get_logger

log = get_logger(__name__)

def _ctx(e: Exception) -> tuple[str, int]:
    _, _, tb = sys.exc_info()
    if tb is None:
        tb = getattr(e, "__traceback__", None)
    if tb:
        frames = traceback.extract_tb(tb)
        if frames:
            return frames[-1].filename, frames[-1].lineno
    return "<unknown>", -1

class ESGException(Exception):
    def __init__(self, error: Exception) -> None:
        script, line = _ctx(error)
        self.error_message = (
            f"[ESGException] File:'{script}' | Line:{line} | "
            f"[{type(error).__name__}] {error}"
        )
        super().__init__(self.error_message)
        log.error(self.error_message)

    def __str__(self): return self.error_message
