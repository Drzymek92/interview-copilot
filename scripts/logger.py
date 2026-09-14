import logging
from pathlib import Path
from datetime import datetime

def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured — avoid duplicate handlers on repeated calls
    # Anchored to the project root via __file__, NOT the CWD: a relative Path("logs")
    # resolves against wherever the process was launched from, so any run whose CWD is
    # not the project root (scheduler, service, Startup-folder shortcut, systemd unit)
    # writes to a different — often unwritable — directory. That cost us a machine_bridge
    # watch daemon crash-looping silently for 5 days (2026-07-19).
    log_dir = Path(__file__).resolve().parents[1] / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{name}_{datetime.now():%Y%m%d}.log"
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    # encoding is explicit: the Windows default is cp1252, which raises UnicodeEncodeError
    # on any non-Latin-1 char in a log message.
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.setLevel(logging.INFO)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    return logger
