# src/managers/embedding_manager.py
import logging
from langchain_community.embeddings import OpenAIEmbeddings
# from langchain.embeddings.huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)


class EmbeddingManager:
    """
    負責將文字或段落轉為向量 (利用 LangChain Embeddings).
    注意：此 Embeddings 與 LLM 模型可分開設定
    """
    def __init__(
        self,
        openai_api_key: str,
        embedding_model_name: str = "text-embedding-ada-002"
    ):
        """
        :param embedding_model_name: 預設使用 text-embedding-ada-002
        """
        self.openai_api_key = openai_api_key
        self.embedding_model_name = embedding_model_name

        # 如果想用其他embedding,可在此切換
        self.embedding_model = OpenAIEmbeddings(
            model=self.embedding_model_name,
            openai_api_key=self.openai_api_key
        )

    def generate_embedding(self, text: str):
        """
        回傳向量 (List[float])
        """
        try:
            embedding = self.embedding_model.embed_query(text)
            return embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return []
