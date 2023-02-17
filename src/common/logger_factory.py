
import logging
import logging.config
from pathlib import Path

from common.logger import Logger


class LoggerFactory:

    @staticmethod
    def create_logger() -> Logger:
        """
        ロガーを作成する

        Returns
        -------
        Logger
            ロガーオブジェクト
        """

        # プロジェクトのルートディレクトリのパスを取得する
        project_root_dir_path = Path(__file__).resolve().parent.parent.parent

        # ログの設定ファイルのパスを取得する
        logging_config_path = project_root_dir_path / "src" / "config" / "logging.ini"

        # ログの出力先ディレクトリを作成する
        logs_dir_path = project_root_dir_path / "logs"
        logs_dir_path.mkdir(parents=True, exist_ok=True)

        # ログファイル名を取得する
        log_file_name = logs_dir_path / "application.log"

        # Windows環境ではファイルパス文字列を正規化する
        logging_config_path = logging_config_path.as_posix()
        log_file_name = log_file_name.as_posix()

        # ログ設定ファイルを読み込んで、ロガーを初期化する
        logging.config.fileConfig(
            logging_config_path,
            defaults={"log_file_name": log_file_name}
        )

        # ロガーオブジェクトを取得する
        logger = Logger(logging.getLogger())

        return logger
