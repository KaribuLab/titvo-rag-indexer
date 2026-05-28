import os

config = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "standard",
        },
    },
    "loggers": {
        "botocore": {
            "level": "WARNING",
            "handlers": ["console"],
            "propagate": False,
        },
        "boto3": {
            "level": "WARNING",
            "handlers": ["console"],
            "propagate": False,
        },
        "urllib3": {
            "level": "WARNING",
            "handlers": ["console"],
            "propagate": False,
        },
    },
    "root": {
        "handlers": ["console"],
        "level": os.getenv("TITVO_LOG_LEVEL", "INFO"),
    },
}
