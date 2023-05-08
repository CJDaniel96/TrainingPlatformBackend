import logging


class Logger:
    def __init__(self, path='main', level=logging.INFO) -> None:
        self.logger = logging.getLogger(path)
        self.logger.setLevel(logging.DEBUG)
        self.formater = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s', '%Y-%m-%d %H:%M:%S')
        self.stream_handler = logging.StreamHandler()
        self.stream_handler.setFormatter(self.formater)
        self.stream_handler.setLevel(level)
        self.logger.addHandler(self.stream_handler)

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
        