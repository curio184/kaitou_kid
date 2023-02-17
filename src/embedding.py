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
        embeddings_file_path: str,
        embedding_model: str = "text-embedding-ada-002",
        encoding: str = "cl100k_base",
        max_tokens_in_sentence: int = 500
    ) -> None:
        """
        テキストを分散表現に変換する。

        Parameters
        ----------
        text_dir_path : str
            テキストファイルのディレクトリパス
        embeddings_file_path : str
            テキストと分散表現を保存するファイルパス
        embedding_model : str, optional
            分散表現に変換する際に使用するモデル名
        encoding : str, optional
            使用するトークナイザー名
        max_tokens_in_sentence : int, optional
            1文あたりの最大トークン数
        """

        self._logger.info("Text embeddings started.")

        # トークナイザーを取得する
        tokenizer = tiktoken.get_encoding(encoding)

        # テキストファイルを読み込む
        texts = self._load_texts(text_dir_path)

        # テキストの改行を削除する
        texts = self._remove_newline(texts)

        # 最大トークン数を下回るように、テキストを分割する。
        texts = self._split_text_by_max_tokens(tokenizer, texts, max_tokens_in_sentence)

        # 分割したテキストのトークン数を取得する
        n_tokens = list(map(lambda x: len(tokenizer.encode(x)), texts))

        # テキストを分散表現に変換する
        embeddings = []
        for i, text in enumerate(texts):
            embedding = self._text_to_embedding(embedding_model, text)
            embeddings.append(embedding)
            self._logger.info(f"Now embedding...({i+1}/{len(texts)})")
            sleep(3)    # Rate limit対策

        # テキストと分散表現をCSVファイルに保存する
        self._save_embeddings_to_csv(
            texts, embeddings, n_tokens, embeddings_file_path
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

    def _remove_newline(self, texts: List[str]) -> List[str]:
        """
        テキストの改行を削除する

        Parameters
        ----------
        List[str]
            テキストのリスト

        Returns
        -------
        List[str]
            改行を削除したテキストのリスト
        """

        # 改行文字をスペースに置換する
        texts = list(map(lambda x: re.sub(r"\n", " ", x), texts))

        # 2つ以上連続するスペースを1つに置換する
        texts = list(map(lambda x: re.sub(r"\s+", " ", x), texts))

        # 2つ以上連続する全角スペースを1つに置換する
        texts = list(map(lambda x: re.sub(r"　+", "　", x), texts))

        return texts

    def _split_text_by_max_tokens(
        self,
        tokenizer: Encoding,
        texts: List[str],
        max_tokens_in_sentence: int
    ) -> List[str]:
        """
        最大トークン数を下回るように、テキストを分割する。

        Parameters
        ----------
        tokenizer: Encoding
            トークナイザー
        texts : List[str]
            テキストのリスト
        max_tokens_in_sentence : int
            1文あたりの最大トークン数

        Returns
        -------
        List[str]
            分割されたテキストのリスト
        """

        chunks = []                 # 分割されたテキストを格納するリスト
        current_chunk = []          # 現在のchunkに追加する文
        current_chunk_tokens = 0    # 現在のchunkに含まれるトークン数

        for text in texts:

            # テキストを文に分割する
            sentences = re.split(r"[.。]", text)

            # 各文のトークン数を取得する
            n_tokens = [len(tokenizer.encode(" " + sentence))
                        for sentence in sentences]

            # 各文に対してループ処理
            for sentence, n_token in zip(sentences, n_tokens):

                # 現在の文を追加すると最大トークン数を超える場合
                if current_chunk_tokens + n_token > max_tokens_in_sentence:
                    chunks.append(". ".join(current_chunk) + ".")
                    current_chunk = []
                    current_chunk_tokens = 0

                # 現在の文のトークン数が最大トークン数よりも大きい場合
                if n_token > max_tokens_in_sentence:
                    # 処理できないので諦める
                    self._logger.exception(
                        f"Embedding skipped due to excessively long sentence. Please divide the sentence into shorter.({sentence})"
                    )
                    continue

                # それ以外の場合
                current_chunk.append(sentence)
                current_chunk_tokens += n_token + 1

        # 残りの文を追加する
        if current_chunk:
            chunks.append(". ".join(current_chunk) + ".")

        return chunks

    def _text_to_embedding(self, embedding_model: str, text: str) -> List[float]:
        """
        テキストを分散表現に変換する

        Parameters
        ----------
        embedding_model : str
            使用するモデル名
        text : str
            テキスト

        Returns
        -------
        List[float]
            分散表現
        """

        # テキストを分散表現に変換する
        oprenai_object = openai.Embedding.create(
            input=text,
            engine=embedding_model
        )
        t_embeddings = oprenai_object["data"][0]["embedding"]

        return t_embeddings

    def _save_embeddings_to_csv(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        n_tokens: List[int],
        embeddings_file_path: str
    ) -> None:
        """
        テキストと分散表現をCSVファイルに保存する

        Parameters
        ----------
        texts : List[str]
            テキストのリスト
        embeddings : List[List[float]]
            分散表現のリスト
        n_tokens : List[int]
            トークン数のリスト
        embeddings_file_path : str
            テキストと分散表現を保存するファイル名
        """

        # CSVファイルに出力する
        df_text = pd.DataFrame(
            {"text": texts, "embeddings": embeddings, "n_tokens": n_tokens}
        )
        df_text.index.name = "id"
        dir_path = os.path.dirname(embeddings_file_path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        df_text.to_csv(f"{embeddings_file_path}", header=True)
