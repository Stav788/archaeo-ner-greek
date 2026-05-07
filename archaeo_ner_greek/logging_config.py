import logging
import logging.config

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s [%(levelname)s]: %(message)s (%(name)s:%(lineno)d)"

        },
    },
    "handlers": {
        "default": {
            "level": "INFO",
            "formatter": "standard",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
        },
        "file": {
            "level": "INFO",
            "formatter": "standard",
            "class": "logging.FileHandler",
            "filename": "training.log",
            "mode": "a",
        },
    },
    "loggers": {
        "": {
            "handlers": ["default", "file"],
            "level": "INFO",
            "propagate": True
        },
        "argilla": {
            "handlers": ["default"],
            "level": "WARNING",
            "propagate": False
        },
        "argilla.client": {
            "handlers": ["default"],
            "level": "WARNING",
            "propagate": False
        },
        "argilla.sdk": {
            "handlers": ["default"],
            "level": "WARNING",
            "propagate": False
        },
        "httpx": {
            "handlers": ["default"],
            "level": "WARNING",
            "propagate": False
        },
    }
}

def setup_logging(level: int = logging.INFO, log_file: str = None):
    """
    Applies the default LOGGING_CONFIG to the current environment and
    suppresses deprecation warnings from third-party libraries like Argilla.
    """
    import warnings
    from datetime import datetime
    
    # Suppress the datetime.utcnow() warning from argilla
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="argilla")
    
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        log_file = f"/tmp/training_{timestamp}.log"
    
    config = LOGGING_CONFIG.copy()
    config["loggers"][""]["level"] = level
    config["handlers"]["file"]["filename"] = log_file
    
    logging.config.dictConfig(config)
    
    # Ensure we return the filename so it can be logged or displayed
    return log_file
