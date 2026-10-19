import asyncio

from fastembed import SparseTextEmbedding, TextEmbedding
from fastembed.common.model_description import DenseModelDescription
from fastembed.rerank.cross_encoder import TextCrossEncoder

from mcp_server_qdrant.embeddings.base import EmbeddingProvider, SparseEmbedding


class FastEmbedProvider(EmbeddingProvider):
    """
    FastEmbed implementation of the embedding provider.
    :param model_name: The name of the dense FastEmbed model to use.
    :param sparse_model_name: The name of the sparse FastEmbed model to use.
    :param reranker_model_name: The name of the FastEmbed cross-encoder to use.
    """

    def __init__(
        self,
        model_name: str,
        sparse_model_name: str = "Qdrant/bm25",
        reranker_model_name: str | None = None,
    ):
        self.model_name = model_name
        self.sparse_model_name = sparse_model_name
        self.reranker_model_name = reranker_model_name
        self.embedding_model = TextEmbedding(model_name)
        self.sparse_embedding_model = SparseTextEmbedding(sparse_model_name)
        self.reranker_model = TextCrossEncoder(model_name=reranker_model_name)

    async def embed_documents(self, documents: list[str]) -> list[list[float]]:
        """Embed a list of documents into vectors."""
        # Run in a thread pool since FastEmbed is synchronous
        loop = asyncio.get_running_loop()
        embeddings = await loop.run_in_executor(
            None, lambda: list(self.embedding_model.passage_embed(documents))
        )
        return [embedding.tolist() for embedding in embeddings]

    async def embed_query(self, query: str) -> list[float]:
        """Embed a query into a vector."""
        # Run in a thread pool since FastEmbed is synchronous
        loop = asyncio.get_running_loop()
        embeddings = await loop.run_in_executor(
            None, lambda: list(self.embedding_model.query_embed([query]))
        )
        return embeddings[0].tolist()

    async def embed_sparse(self, texts: list[str]) -> list[SparseEmbedding]:
        """Embed texts into sparse vectors."""
        loop = asyncio.get_running_loop()
        embeddings = await loop.run_in_executor(
            None, lambda: list(self.sparse_embedding_model.embed(texts))
        )
        return [
            (embedding.indices.tolist(), embedding.values.tolist())
            for embedding in embeddings
        ]

    async def rerank(self, query: str, documents: list[str]) -> list[float]:
        """Score documents with the configured cross-encoder without blocking the event loop."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: list(self.reranker_model.rerank(query, documents))
        )

    def get_vector_name(self) -> str:
        """
        Return the name of the densevector for the Qdrant collection.
        Important: This is compatible with the FastEmbed logic used before 0.7.3.
        """
        model_name = self.embedding_model.model_name.split("/")[-1].lower()
        return f"fast-{model_name}"

    def get_vector_size(self) -> int:
        """Get the size of the vector for the Qdrant collection."""
        model_description: DenseModelDescription = (
            self.embedding_model._get_model_description(self.model_name)
        )
        return model_description.dim

    def get_sparse_vector_name(self) -> str:
        """Return the name of the sparse vector used for the Qdrant collection."""
        model_name = self.sparse_model_name.split("/")[-1].lower()
        return f"fast-{model_name}"
