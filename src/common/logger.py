
import logging


class Logger:

    def __init__(self, logger: logging.Logger):
        """
        コンストラクタ

        Parameters
        ----------
        logger : logging.Logger
            ロガーオブジェクト
        """
        self._logger = logger

    def debug(self, message: str):
        self._logger.debug(message)

    def info(self, message: str):
        self._logger.info(message)

    def warning(self, message: str):
        self._logger.warning(message)

    def errror(self, message: str):
        self._logger.error(message)

    def exception(self, message: str):
        self._logger.exception(message)
