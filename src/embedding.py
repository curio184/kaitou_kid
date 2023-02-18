import os
import re
from time import sleep
from typing import List

import openai
import pandas as pd
import tiktoken
from tiktoken import Encoding

from common.directory import Directory
from common.logger_factory import LoggerFactory


class BuildEmbeddingUseCase:
    """
    テキストを分散表現に変換するクラス。
    """

    def __init__(self) -> None:
        self._logger = LoggerFactory.create_logger()

    def execute(
        self,
        text_dir_path: str,
        text_embeddings_file_path: str,
        embedding_model: str = "text-embedding-ada-002",
        encoding: str = "cl100k_base",
        max_sentence_token_count: int = 150
    ) -> None:
        """
        テキストを分散表現に変換する。

        Parameters
        ----------
        text_dir_path : str
            テキストファイルのディレクトリパス
        text_embeddings_file_path : str
            テキストと分散表現を保存するファイルパス
        embedding_model : str, optional
            使用する分散表現モデル名
        encoding : str, optional
            使用するトークナイザー名
        max_sentence_token_count : int, optional
            1文あたりの最大トークン数
        """

        self._logger.info("Text embeddings started.")

        # トークナイザーを取得する
        tokenizer = tiktoken.get_encoding(encoding)

        # テキストファイルを読み込む
        texts = self._load_texts(text_dir_path)

        # テキストから不要文字を取り除く
        texts = self._remove_unnecessary_chars(texts)

        # 1文あたりの最大トークン数を超えないように、テキストを文に分割する。
        sentences = self._split_texts_by_token_count(
            tokenizer, texts, max_sentence_token_count
        )

        # 分割したテキストのトークン数を取得する
        sentence_token_counts = list(map(lambda x: len(tokenizer.encode(x)), sentences))

        # テキストを分散表現に変換する
        embeddings = self._text_to_embeddings(embedding_model, sentences, sentence_token_counts)

        # テキストと分散表現をCSVファイルに保存する
        self._save_text_embeddings_to_csv(
            sentences, embeddings, sentence_token_counts, text_embeddings_file_path
        )

        self._logger.info("Text embeddings finished.")

    def _load_texts(self, text_dir_path: str) -> List[str]:
        """
        テキストファイルを読み込む

        Parameters
        ----------
        text_dir_path : str
            テキストファイルのディレクトリパス

        Returns
        -------
        List[str]
            テキストのリスト
        """

        texts = []
        for file_path in Directory.get_files(text_dir_path):
            with open(file_path, "r", encoding="UTF-8") as f:
                text = f.read()
                texts.append(text)

        return texts

    def _remove_unnecessary_chars(self, texts: List[str]) -> List[str]:
        """
        テキストから不要文字を取り除く

        Parameters
        ----------
        List[str]
            テキストのリスト

        Returns
        -------
        List[str]
            不要文字を取り除いたテキストのリスト
        """

        # 改行文字をスペースに置換する
        texts = list(map(lambda x: re.sub(r"\n", " ", x), texts))

        # 2つ以上連続するスペースを1つに置換する
        texts = list(map(lambda x: re.sub(r"\s+", " ", x), texts))

        # 2つ以上連続する全角スペースを1つに置換する
        texts = list(map(lambda x: re.sub(r"　+", "　", x), texts))

        return texts

    def _split_texts_by_token_count(
        self,
        tokenizer: Encoding,
        texts: List[str],
        max_sentence_token_count: int
    ) -> List[str]:
        """
        1文あたりの最大トークン数を超えないように、テキストを文に分割する。

        Parameters
        ----------
        tokenizer: Encoding
            トークナイザー
        texts : List[str]
            テキストのリスト
        max_sentence_token_count : int
            1文あたりの最大トークン数

        Returns
        -------
        List[str]
            分割された文のリスト
        """

        # 分割された文のリスト
        grouped_sentences = []

        for text in texts:

            # テキストを文に分割する
            sentences = re.split(
                r"[.。]",
                text[:-1] if text[-1] in [".", "。"] else text
            )

            # 各文のトークン数を取得する
            sentence_token_counts = [len(tokenizer.encode(" " + sentence))
                                     for sentence in sentences]

            # 1文あたりの最大トークン数を超えないように、文をグルーピングする。
            sentence_groups = self._group_sentences_by_token_count(
                sentences, sentence_token_counts, max_sentence_token_count
            )

            # 分割された文をリストに追加する
            for sentence_group in sentence_groups:
                grouped_sentences.append(". ".join(sentence_group) + ".")

        return grouped_sentences

    def _text_to_embeddings(
        self,
        embedding_model: str,
        sentences: List[str],
        sentence_token_counts: List[int]
    ) -> List[float]:
        """
        テキストを分散表現に変換する

        Parameters
        ----------
        embedding_model : str
            使用する分散表現モデル名
        sentences : List[str]
            文のリスト
        sentence_token_counts : List[int]
            各文のトークン数のリスト

        Returns
        -------
        List[List[float]]
            分散表現のリスト
        """

        # 1リクエストで照会可能な最大トークン数を超えないように、文をグルーピングする。
        sentence_groups = self._group_sentences_by_token_count(sentences, sentence_token_counts, 8000)

        embeddings = []
        for i, sentence_group in enumerate(sentence_groups):
            self._logger.info(f"Now embedding...({i+1}/{len(sentence_groups)})")

            # テキストを分散表現に変換する
            embedding_response = openai.Embedding.create(
                input=sentence_group,
                engine=embedding_model
            )
            embeddings.extend([x["embedding"] for x in embedding_response["data"]])

            sleep(3)    # Rate limit対策

        return embeddings

    def _save_text_embeddings_to_csv(
        self,
        sentences: List[str],
        embeddings: List[List[float]],
        sentence_token_count: List[int],
        text_embeddings_file_path: str
    ) -> None:
        """
        テキストと分散表現をCSVファイルに保存する

        Parameters
        ----------
        sentences : List[str]
            文のリスト
        embeddings : List[List[float]]
            分散表現のリスト
        sentence_token_count : List[int]
            文のトークン数のリスト
        text_embeddings_file_path : str
            テキストと分散表現を保存するファイル名
        """

        # CSVファイルに出力する
        df_text = pd.DataFrame(
            {"sentence": sentences, "embeddings": embeddings, "sentence_token_count": sentence_token_count}
        )
        df_text.index.name = "sentence_id"
        dir_path = os.path.dirname(text_embeddings_file_path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        df_text.to_csv(f"{text_embeddings_file_path}", header=True)

    def _group_sentences_by_token_count(
        self,
        sentences: List[str],
        sentence_token_counts: List[int],
        max_token_count_per_group: int
    ) -> List[List[str]]:
        """
        1グループあたりの最大トークン数を超えないように、文をグルーピングする。

        Parameters
        ----------
        sentences : List[str]
            文のリスト
        sentence_token_counts : List[int]
            文のトークン数のリスト
        max_group_token_count : int
            1グループあたりの最大トークン数

        Returns
        -------
        List[List[str]]
            グルーピングした文のリスト
        """

        groups = []
        current_group_token_count = 0
        current_group = []

        for sentence, token_count in zip(sentences, sentence_token_counts):
            if current_group_token_count + token_count <= max_token_count_per_group:
                current_group_token_count += token_count
                current_group.append(sentence)
            else:
                if current_group:
                    groups.append(current_group)
                current_group = [sentence]
                current_group_token_count = token_count

        if current_group:
            groups.append(current_group)

        return groups
