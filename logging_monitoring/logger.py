"""Authored by: Ibrahim Noor."""

import logging
import sys
from pathlib import Path
from pythonjsonlogger import jsonlogger

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

LOG_FILE = LOG_DIR / "pipeline.log"


class CustomJsonFormatter(jsonlogger.JsonFormatter):
    """JSON log formatter that injects timestamp, level, logger name, module, function, and line."""

    def add_fields(self, log_record, record, message_dict):
        super(CustomJsonFormatter, self).add_fields(log_record, record, message_dict)

        if not log_record.get("timestamp"):
            log_record["timestamp"] = record.created

        if log_record.get("level"):
            log_record["level"] = log_record["level"].upper()
        else:
            log_record["level"] = record.levelname

        log_record["logger"] = record.name
        log_record["module"] = record.module
        log_record["function"] = record.funcName
        log_record["line"] = record.lineno


def get_logger(name):
    """Return a logger that emits JSON to stdout and to logs/pipeline.log."""
    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.setLevel(logging.INFO)

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)

        console_formatter = CustomJsonFormatter(
            "%(timestamp)s %(level)s %(name)s %(message)s"
        )
        console_handler.setFormatter(console_formatter)

        file_handler = logging.FileHandler(LOG_FILE)
        file_handler.setLevel(logging.INFO)

        file_formatter = CustomJsonFormatter(
            "%(timestamp)s %(level)s %(name)s %(message)s"
        )
        file_handler.setFormatter(file_formatter)

        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

        logger.propagate = False

    return logger


if __name__ == "__main__":
    logger = get_logger(__name__)

    logger.info("Logger initialized successfully")
    logger.info("Processing data", extra={
        "records_processed": 100,
        "errors": 0,
        "duration_seconds": 2.5,
    })
    logger.warning("High latency detected", extra={
        "latency_ms": 1500,
        "threshold_ms": 1000,
    })
    logger.error("Database connection failed", extra={
        "host": "localhost",
        "port": 5432,
        "error": "Connection timeout",
    })

    print(f"\nLogs written to: {LOG_FILE}")
    print("\nExample log entry:")
    print(open(LOG_FILE).readlines()[-1])
