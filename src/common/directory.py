import os
from typing import List


class Directory:

    @staticmethod
    def get_files(path: str, extension: str = ".txt") -> List[str]:
        """
        ディレクトリ内のファイルのリストを取得する

        Parameters
        ----------
        path : str
            検索するディレクトリ
        extension : str
            検索する拡張子(.txt)

        Returns
        -------
        List[str]
            ファイルのリスト
        """

        file_paths = []
        for root, dirs, files in os.walk(path):
            for file in files:
                if os.path.splitext(file)[1] == extension:
                    file_path = os.path.join(root, file)
                    file_paths.append(file_path)

        return file_paths
