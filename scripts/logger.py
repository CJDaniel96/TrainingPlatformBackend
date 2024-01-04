import logging


class Logger:
    def __init__(self, debug_file='debug.log', name='main', level='DEBUG', format='[%(asctime)s] [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S') -> None:
        self.logger = logging.getLogger(name=name)
        self.logger.setLevel(self.getLevel(level))

        if not self.logger.handlers:
            self.file_logger = logging.FileHandler(debug_file, mode='a')
            self.file_logger.setLevel(self.getLevel(level))
            self.file_logger.setFormatter(logging.Formatter(format))

            self.stream_logger = logging.StreamHandler()
            self.stream_logger.setLevel(self.getLevel(level))
            self.stream_logger.setFormatter(logging.Formatter(format))

            self.logger.addHandler(self.file_logger)
            self.logger.addHandler(self.stream_logger)

    def getLevel(self, level_string):
        if level_string == 'DEBUG':
            return logging.DEBUG
        elif level_string == 'INFO':
            return logging.INFO
        elif level_string == 'WARN':
            return logging.WARN
        elif level_string == 'WARNING':
            return logging.WARNING
        elif level_string == 'ERROR':
            return logging.ERROR
        elif level_string == 'CRITICAL':
            return logging.CRITICAL
        else:
            return logging.NOTSET

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