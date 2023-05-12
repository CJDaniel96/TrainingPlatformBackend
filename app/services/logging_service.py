import logging


class Logger:
    def __init__(self, name='main', level=logging.INFO, format='[%(asctime)s] [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S') -> None:
        self.logger = logging.getLogger(name=name)
        logging.basicConfig(level=level, format=format, datefmt=datefmt)

    @classmethod
    def debug(cls, message):
        cls().logger.debug(message)

    @classmethod
    def info(cls, message):
        cls().logger.info(message)

    @classmethod
    def warn(cls, message):
        cls().logger.warn(message)

    @classmethod
    def error(cls, message):
        cls().logger.error(message)

    @classmethod
    def critical(cls, message):
        cls().logger.critical(message)
        