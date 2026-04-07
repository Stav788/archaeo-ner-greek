import logging
import logging.config

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        },
    },
    "handlers": {
        "default": {
            "level": "INFO",
            "formatter": "standard",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
        },
    },
    "loggers": {
        "": {
            "handlers": ["default"],
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

def setup_logging(level: int = logging.INFO):
    """
    Applies the default LOGGING_CONFIG to the current environment and
    suppresses deprecation warnings from third-party libraries like Argilla.
    """
    import warnings
    # Suppress the datetime.utcnow() warning from argilla
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="argilla")
    
    config = LOGGING_CONFIG.copy()
    config["loggers"][""]["level"] = level
    logging.config.dictConfig(config)
