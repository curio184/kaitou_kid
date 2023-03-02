prompt_template_en = """
understand the following context Step by Step, and answer the question.

---

Context: {context}

---

Question: {question}
Answer:
"""

prompt_template_jp = """
(1) 次のコンテキストをStep by Stepで理解してください。
(2) 質問の意図をくみ取り、回答を考えてください。
(3) 読みやすい文章に整形し、回答を出力してください。

---

Context: {context}

---
Question: {question}
Answer:
"""


class PromptBuilder:

    @staticmethod
    def build(context: str, question: str) -> str:
        """
        プロンプトを作成する

        Parameters
        ----------
        context : str
            コンテキスト
        question : str
            質問

        Returns
        -------
        str
            プロンプト
        """
        return prompt_template_jp.format(context=context, question=question)
