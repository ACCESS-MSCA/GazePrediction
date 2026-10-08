import logging
import sys
import os
from datetime import datetime

def setup_logger(script_name, log_dir=None):
    """
    Configures a comprehensive logger for Python scripts.
    Returns a logger instance.
    """
    if log_dir is None:
        # Use environment variable if set by master script, otherwise default to "logs"
        log_dir = os.environ.get("RUN_LOG_DIR", "logs")
        
    try:
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"{script_name}_{timestamp}.log")

        # Create logger
        logger = logging.getLogger(script_name)
        logger.setLevel(logging.INFO)

        # Create formatters
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )

        # File Handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(file_formatter)

        # Console Handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(console_formatter)

        # Add handlers
        if not logger.handlers:
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)

        logger.info(f"Logging initialized for {script_name}. Log file: {log_file}")
        return logger

    except Exception as e:
        print(f"CRITICAL: Failed to setup logging: {e}", file=sys.stderr)
        sys.exit(1)
