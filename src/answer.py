import numpy as np
import openai
import pandas as pd
from openai.embeddings_utils import distances_from_embeddings


class AnswerQuestionUseCase:
    """
    テキストに対して質問を投げかけ、回答を得るクラス。
    """

    def execute(
        self,
        embeddings_file_path: str,
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
        embeddings_file_path : str
            テキストと分散表現が保存されたファイルパス
        question : str
            質問
        embedding_model : str, optional
            分散表現に変換する際に使用するモデル名
        completion_model : str, optional
            自然言語を生成する際に使用するモデル名
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

        print("Answering question started.")

        # テキストと分散表現を読み込む
        df_text = self._load_embeddings_from_csv(embeddings_file_path)

        # 質問と類似度の近いテキストを検索し、コンテキストを作成する
        context = self._create_context(
            df_text, question, embedding_model, max_context_tokens
        )

        # コンテキストを与えて質問の回答を得る
        answer = self._answer_question(
            context, question, completion_model, max_generate_tokens, stop_sequence
        )

        print("Answering question finished.")

        return answer

    def _load_embeddings_from_csv(self, embeddings_file_path: str) -> pd.DataFrame:
        """
        CSVファイルからテキストと分散表現を読み込む

        Parameters
        ----------
        embeddings_file_path : str
            テキストと分散表現が保存されたファイルパス

        Returns
        -------
        pd.DataFrame
            テキストと分散表現のリスト
        """
        df_text = pd.read_csv(f"{embeddings_file_path}", index_col=0)
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
            分散表現に変換する際に使用するモデル名
        max_context_tokens : int, optional
            コンテキストの最大トークン数

        Returns
        -------
        str
            コンテキスト
        """

        # 質問を分散表現に変換する
        oprenai_object = openai.Embedding.create(
            input=question,
            engine=embedding_model
        )
        q_embeddings = oprenai_object["data"][0]["embedding"]

        # 「テキストの分散表現」と「質問の分散表現」の距離を計算する
        # 距離が近ければ近いほど、テキストは質問に関連していると考えられる
        df_text["distances"] = distances_from_embeddings(
            q_embeddings, df_text["embeddings"].values, distance_metric="cosine"
        )

        contexts = []
        total_tokens = 0

        # 距離(意味)が近い順にソートし、コンテキストの最大トークン数を超えるまで、テキストを追加する
        for i, row in df_text.sort_values("distances", ascending=True).iterrows():

            # テキストのトークン数を加算する
            total_tokens += row["n_tokens"] + 4

            # コンテキストの最大トークン数を超えたら、ループを抜ける
            if total_tokens > max_context_tokens:
                break

            # テキストをコンテキストに追加する
            contexts.append(row["text"])

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
        コンテキストを与えて質問の回答を得る

        Parameters
        ----------
        context : str
            コンテキスト
        question : str
            質問
        completion_model : str, optional
            自然言語を生成する際に使用するモデル名
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
        response = openai.Completion.create(
            prompt=f"Answer the question based on the context below, and if the question can't be answered based on the context, say \"I don't know\"\n\nContext: {context}\n\n---\n\nQuestion: {question}\nAnswer:",
            temperature=0,
            max_tokens=max_generate_tokens,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=stop_sequence,
            model=completion_model,
        )

        # 答えを返す
        return response["choices"][0]["text"]
