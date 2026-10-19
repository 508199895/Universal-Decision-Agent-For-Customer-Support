from abc import ABC, abstractmethod

SparseEmbedding = tuple[list[int], list[float]]


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""

    @abstractmethod
    async def embed_documents(self, documents: list[str]) -> list[list[float]]:
        """Embed a list of documents into vectors."""

    @abstractmethod
    async def embed_query(self, query: str) -> list[float]:
        """Embed a query into a vector."""

    @abstractmethod
    async def embed_sparse(self, texts: list[str]) -> list[SparseEmbedding]:
        """Embed texts into sparse vectors."""

    @abstractmethod
    async def rerank(self, query: str, documents: list[str]) -> list[float]:
        """Score documents against a query with a cross-encoder."""

    @abstractmethod
    def get_vector_name(self) -> str:
        """Get the name of the vector for the Qdrant collection."""

    @abstractmethod
    def get_vector_size(self) -> int:
        """Get the size of the vector for the Qdrant collection."""

    @abstractmethod
    def get_sparse_vector_name(self) -> str:
        """Get the name of the sparse vector for the Qdrant collection."""
