import openai

from answer import AnswerQuestionUseCase
from embedding import BuildEmbeddingUseCase

# OpenAIのAPIキーを設定する。
openai.api_key = "YOUR_API_KEY"

# STEP1: テキストを分散表現に変換する。
BuildEmbeddingUseCase().execute(
    text_dir_path="text/fictional_novel",
    embeddings_file_path="embeddings/fictional_novel.csv"
)

# STEP2: テキストに対して質問を投げかけ、回答を得る。
answer = AnswerQuestionUseCase().execute(
    embeddings_file_path="embeddings/fictional_novel.csv",
    question="主人公の性格は？"
)
print(f"Answer: {answer}")  # トシオは、ドジっぷりがありながらも、勇敢で、友情を大切にする性格である。
