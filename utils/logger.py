"""
logger.py
=========
Structured logging using structlog. Writes both to console and rotating file.
"""
from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path

import structlog

from utils.config_loader import Config


def setup_logger(name: str = "pumpfun_agent") -> structlog.BoundLogger:
    cfg = Config.get()
    log_cfg = cfg.get_nested("logging", default={})
    level = getattr(logging, log_cfg.get("level", "INFO").upper(), logging.INFO)
    log_file = Path(log_cfg.get("file", "./data/agent.log"))
    log_file.parent.mkdir(parents=True, exist_ok=True)

    # Rotating file handler
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=log_cfg.get("max_size_mb", 50) * 1024 * 1024,
        backupCount=log_cfg.get("backup_count", 5),
        encoding="utf-8",
    )
    file_handler.setLevel(level)

    console = logging.StreamHandler()
    console.setLevel(level)

    use_json = log_cfg.get("json_format", False)
    if use_json:
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.PrintLoggerFactory(file=None),
        cache_logger_on_first_use=True,
    )

    # Bridge stdlib logging
    root = logging.getLogger()
    root.handlers = [file_handler, console]
    root.setLevel(level)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("web3").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)

    return structlog.get_logger(name)
