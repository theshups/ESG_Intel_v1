"""src/logger.py — Centralised Rich-enhanced logging."""
import logging, os
from datetime import datetime
from rich.logging import RichHandler

_ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR  = os.path.join(_ROOT, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE = os.path.join(LOG_DIR, f"esg_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log")

_fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
_fh.setFormatter(logging.Formatter(
    "%(asctime)s [%(levelname)-8s] %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"))

_rh = RichHandler(rich_tracebacks=True, show_path=False, markup=False)

logging.basicConfig(level=logging.DEBUG, handlers=[_fh, _rh])
for _lib in ("urllib3","httpx","tensorflow","absl","h5py","matplotlib",
             "PIL","pdfminer","watchdog","numba"):
    logging.getLogger(_lib).setLevel(logging.WARNING)

def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)

get_logger(__name__).info("Logger ready → %s", LOG_FILE)
