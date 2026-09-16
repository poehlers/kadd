import logging
import os
from importlib import reload


def separator() -> None:
    logging.info("-----------------------------------------------")


def init_logger(*, fn: str, verbose: bool = True) -> None:
    """Initialize the logger with the specified log file path and verbosity level.

    Args:
        fn: The path to the log file directory.
        verbose: Whether to enable console logging in addition to the file.
    """
    os.makedirs(fn, exist_ok=True)
    reload(logging)
    logger = logging.getLogger()
    logger.handlers.clear()

    formatter = logging.Formatter('%(asctime)s - %(message)s')
    file_handler = logging.FileHandler(f"{fn}/log", mode='w')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    if verbose:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    logger.setLevel(logging.INFO)
