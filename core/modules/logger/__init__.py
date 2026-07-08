from contextlib import contextmanager
from datetime import timedelta, timezone
from pathlib import Path
import sys
import time
from typing import Any, Iterator

from loguru import logger


def get_runtime_project_root() -> Path:
    cwd = Path.cwd()
    search_list = [cwd, *cwd.parents]

    for path in search_list:
        if (path / "pyproject.toml").exists() or (path / ".git").exists() or (path / ".gitignore").exists():
            return path

    return cwd


BEIJING_TIMEZONE = timezone(timedelta(hours=8))


def beijing_time_patcher(record: dict[str, Any]) -> None:
    record["time"] = record["time"].astimezone(BEIJING_TIMEZONE)


def setup_logger(log_file_path: str | Path = "logs/app.log") -> Path:
    project_root = get_runtime_project_root()
    full_log_path = project_root / log_file_path
    full_log_path.parent.mkdir(parents=True, exist_ok=True)

    logger.remove()
    logger.configure(patcher=beijing_time_patcher)

    console_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>"
    )
    file_format = "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}"

    logger.add(sys.stderr, level="INFO", format=console_format, enqueue=True)
    logger.add(
        str(full_log_path),
        rotation="00:00",
        retention="7 days",
        encoding="utf-8",
        level="INFO",
        format=file_format,
        enqueue=True,
    )
    return full_log_path


def get_logger():
    return logger


@contextmanager
def log_step(name: str, **context: Any) -> Iterator[None]:
    started = time.perf_counter()
    context_text = _format_context(context)
    logger.info("start {}{}", name, context_text)

    try:
        yield
    except Exception:
        elapsed = time.perf_counter() - started
        logger.exception("failed {}{} elapsed_seconds={:.3f}", name, context_text, elapsed)
        raise

    elapsed = time.perf_counter() - started
    logger.info("done {}{} elapsed_seconds={:.3f}", name, context_text, elapsed)


def _format_context(context: dict[str, Any]) -> str:
    if not context:
        return ""

    parts = [f"{key}={value}" for key, value in context.items()]
    return " " + " ".join(parts)


setup_logger()


__all__ = [
    "logger",
    "setup_logger",
    "get_logger",
    "log_step",
    "get_runtime_project_root",
]
