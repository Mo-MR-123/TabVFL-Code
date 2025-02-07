import logging

class TabNetLogger:
    """
    General logging class for TabNet VFL and experiments that utilize TabNet.
    """
    def __init__(self, name, level=logging.DEBUG, file_name=None, truncation_chars=6000):
        """Constructs the logger object. 

        Args:
            name (str): Name of the logger.
            level (int, optional): Level of logging (check python logging docs). Defaults to logging.DEBUG.
            file_name (str, optional): If provided, a log file is created with the specified name. Defaults to None.
            truncation_chars (int, optional): Amount of chars to be logged before truncation. Defaults to 150.
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)

        formatter = logging.Formatter(f'[%(name)s] - [%(levelname)s] - %(message).{truncation_chars}s')

        if file_name:
            file_handler = logging.FileHandler(file_name)
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

    def get_logger(self):
        return self.logger