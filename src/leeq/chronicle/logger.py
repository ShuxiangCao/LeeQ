import logging


def setup_logging(name, level=logging.INFO):
    """
    Setup the logging system

    Parameters:
        name (str): The name of the logger
        level (int): The logging level

    Returns:
        logger (logging.Logger): The logger
    """
    # Create the logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Check if handlers already exist to prevent duplicates
    if not logger.handlers:
        # Create the console handler with a recommended format
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        console_handler.setFormatter(formatter)

        # Add the console handler to the logger
        logger.addHandler(console_handler)

    # Prevent propagation to parent loggers to avoid duplicate messages
    # This is especially important for chronicle loggers since the parent
    # leeq.chronicle namespace also has a handler
    if name.startswith('leeq.chronicle.'):
        logger.propagate = False

    # Optionally, add a file handler as well
    # file_handler = logger.FileHandler('app.log')
    # file_handler.setLevel(logger.DEBUG)
    # file_handler.setFormatter(formatter)
    # logger.addHandler(file_handler)

    # Return the logger
    return logger
