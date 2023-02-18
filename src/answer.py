import numpy as np
import openai
import pandas as pd
from openai.embeddings_utils import distances_from_embeddings

from common.logger_factory import LoggerFactory


class AnswerQuestionUseCase:
    """
    テキストに対して質問を投げかけ、回答を得るクラス。
    """

    def __init__(self) -> None:
        self._logger = LoggerFactory.create_logger()

    def execute(
        self,
        text_embeddings_file_path: str,
        question: str,
        embedding_model: str = "text-embedding-ada-002",
        completion_model: str = "text-davinci-003",
        max_context_tokens: int = 1800,
        max_generate_tokens: int = 150,
        stop_sequence=None
    ) -> str:
        """
        テキストに対して質問を投げかけ、回答を得る。

        Parameters
        ----------
        text_embeddings_file_path : str
            テキストと分散表現が保存されたファイルパス
        question : str
            質問
        embedding_model : str, optional
            使用する分散表現モデル名
        completion_model : str, optional
            使用する自然言語モデル名
        max_context_tokens : int, optional
            コンテキストの最大トークン数
        max_generate_tokens : int, optional
            生成するトークンの最大数
        stop_sequence : str, optional
            生成を停止する特定の文字列

        Returns
        -------
        str
            回答
        """

        self._logger.info("Answering question started.")

        # テキストと分散表現を読み込む
        df_text = self._load_text_embeddings_from_csv(text_embeddings_file_path)

        # 質問と類似度の近いテキストを検索し、コンテキストを作成する
        context = self._create_context(
            df_text, question, embedding_model, max_context_tokens
        )

        # コンテキストと質問を与えて回答を得る
        answer = self._answer_question(
            context, question, completion_model, max_generate_tokens, stop_sequence
        )

        self._logger.info("Answering question finished.")

        return answer

    def _load_text_embeddings_from_csv(self, text_embeddings_file_path: str) -> pd.DataFrame:
        """
        CSVファイルからテキストと分散表現を読み込む

        Parameters
        ----------
        text_embeddings_file_path : str
            テキストと分散表現が保存されたファイルパス

        Returns
        -------
        pd.DataFrame
            テキストと分散表現のリスト
        """
        df_text = pd.read_csv(f"{text_embeddings_file_path}", index_col=0)
        df_text["embeddings"] = df_text["embeddings"].apply(eval).apply(np.array)
        return df_text

    def _create_context(
        self,
        df_text: pd.DataFrame,
        question: str,
        embedding_model: str = "text-embedding-ada-002",
        max_context_tokens: int = 1800,
    ) -> str:
        """
        質問と類似度の近いテキストを検索し、コンテキストを作成する

        Parameters
        ----------
        df_text : pd.DataFrame
            テキストと分散表現
        question : str
            質問
        embedding_model: str, optional
            使用する分散表現モデル名
        max_context_tokens : int, optional
            コンテキストの最大トークン数

        Returns
        -------
        str
            コンテキスト
        """

        # 質問を分散表現に変換する
        embedding_response = openai.Embedding.create(
            input=question,
            engine=embedding_model
        )
        question_embeddings = embedding_response["data"][0]["embedding"]

        # テキストと質問の分散表現の類似度を計算し、類似度が高い順にソートする
        df_text["distances"] = distances_from_embeddings(
            question_embeddings, df_text["embeddings"].values, distance_metric="cosine"
        )
        df_text_sorted = df_text.sort_values("distances", ascending=True)

        # 類似度が高い文の前後の優先度を上げるようにソートする
        candidate_sentence_ids = []
        for sentence_id in df_text_sorted.index:
            left_boundary = sentence_id - 1 if sentence_id > 0 else sentence_id
            right_boundary = sentence_id + 1 if sentence_id < df_text_sorted.index.max() else sentence_id
            for neighbor_id in range(left_boundary, right_boundary + 1):
                if neighbor_id not in candidate_sentence_ids:
                    candidate_sentence_ids.append(neighbor_id)

        # コンテキストの最大トークン数を超えないように文を絞り込む
        selected_sentence_ids = []
        total_tokens = 0
        for sentence_id in candidate_sentence_ids:
            sentence_tokens = df_text_sorted.loc[sentence_id, "sentence_token_count"]
            if total_tokens + sentence_tokens <= max_context_tokens:
                selected_sentence_ids.append(sentence_id)
                total_tokens += sentence_tokens
            else:
                break

        # 連続する文をコンテキストの断片としてまとめる
        context_fragments = []
        for sentence_id in sorted(selected_sentence_ids):
            if len(context_fragments) == 0:
                context_fragments.append([sentence_id])
            else:
                if sentence_id - context_fragments[-1][-1] == 1:
                    context_fragments[-1].append(sentence_id)
                else:
                    context_fragments.append([sentence_id])

        # コンテキストの断片を結合してコンテキストを作成する
        contexts = []
        for context_fragment in context_fragments:
            text_fragments = []
            for sentence_id in context_fragment:
                text_fragments.append(df_text_sorted.loc[sentence_id, "sentence"])
            context = "".join(text_fragments)
            contexts.append(context)

        # コンテキストを返す
        return "\n\n###\n\n".join(contexts)

    def _answer_question(
        self,
        context: str,
        question: str,
        completion_model: str = "text-davinci-003",
        max_generate_tokens: int = 150,
        stop_sequence: str = None
    ) -> str:
        """
        コンテキストと質問を与えて回答を得る

        Parameters
        ----------
        context : str
            コンテキスト
        question : str
            質問
        completion_model : str, optional
            使用する自然言語モデル名
        max_generate_tokens : int, optional
            生成するトークンの最大数
        stop_sequence : str, optional
            生成を停止する特定の文字列
            特定の文字列にヒットすると、トークン生成を中止してテキストを返す。
            文法的な完全性を保証するための役割を果たす。

        Returns
        -------
        str
            回答
        """

        # コンテキストを与えて質問の回答を得る
        completion_response = openai.Completion.create(
            prompt=f"Answer the question based on the context below, and if the question can't be answered based on the context, say \"I don't know\"\n\nContext: {context}\n\n---\n\nQuestion: {question}\nAnswer:",
            temperature=0,
            max_tokens=max_generate_tokens,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=stop_sequence,
            model=completion_model,
        )

        self._logger.info(f"Context: {context}")
        self._logger.info(f"Question: {question}")
        self._logger.info(f"Answer: {completion_response['choices'][0]['text']}")

        # 答えを返す
        return completion_response["choices"][0]["text"]
