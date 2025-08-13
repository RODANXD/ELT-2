import logging
import os
import io

_log_stream = io.StringIO()

def get_log_stream():
    return _log_stream

def setup_logger(name, log_file=None, level=logging.INFO, to_memory=True):
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)

        # Console / terminal logs
        ch = logging.StreamHandler()
        ch.setLevel(level)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        # Optional: Memory log for UI
        if to_memory:
            mh = logging.StreamHandler(_log_stream)
            mh.setLevel(level)
            mh.setFormatter(formatter)
            logger.addHandler(mh)

    return logger
