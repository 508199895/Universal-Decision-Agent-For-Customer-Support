import asyncio
import logging
import uuid
from typing import Any

from pydantic import BaseModel
from qdrant_client import AsyncQdrantClient, models

from mcp_server_qdrant.embeddings.base import EmbeddingProvider
from mcp_server_qdrant.settings import METADATA_PATH

logger = logging.getLogger(__name__)

Metadata = dict[str, Any]
ArbitraryFilter = dict[str, Any]
RELEVANCE_SCORE_KEY = "relevance_score"


class Entry(BaseModel):
    """
    A single entry in the Qdrant collection.
    """

    content: str
    metadata: Metadata | None = None


class QdrantConnector:
    """
    Encapsulates the connection to a Qdrant server and all the methods to interact with it.
    :param qdrant_url: The URL of the Qdrant server.
    :param qdrant_api_key: The API key to use for the Qdrant server.
    :param collection_name: The name of the default collection to use. If not provided, each tool will require
                            the collection name to be provided.
    :param embedding_provider: The embedding provider to use.
    :param qdrant_local_path: The path to the storage directory for the Qdrant client, if local mode is used.
    """

    def __init__(
        self,
        qdrant_url: str | None,
        qdrant_api_key: str | None,
        collection_name: str | None,
        embedding_provider: EmbeddingProvider,
        qdrant_local_path: str | None = None,
        field_indexes: dict[str, models.PayloadSchemaType] | None = None,
    ):
        self._qdrant_url = qdrant_url.rstrip("/") if qdrant_url else None
        self._qdrant_api_key = qdrant_api_key
        self._default_collection_name = collection_name
        self._embedding_provider = embedding_provider
        self._client = AsyncQdrantClient(
            location=qdrant_url, api_key=qdrant_api_key, path=qdrant_local_path
        )
        self._field_indexes = field_indexes

    async def get_collection_names(self) -> list[str]:
        """
        Get the names of all collections in the Qdrant server.
        :return: A list of collection names.
        """
        response = await self._client.get_collections()
        return [collection.name for collection in response.collections]

    async def store(self, entry: Entry, *, collection_name: str | None = None):
        """
        Store some information in the Qdrant collection, along with the specified metadata.
        :param entry: The entry to store in the Qdrant collection.
        :param collection_name: The name of the collection to store the information in, optional. If not provided,
                                the default collection is used.
        """
        collection_name = collection_name or self._default_collection_name
        assert collection_name is not None
        await self._ensure_collection_exists(collection_name)

        # Embed the document
        # ToDo: instead of embedding text explicitly, use `models.Document`,
        # it should unlock usage of server-side inference.
        dense_embeddings, sparse_embeddings = await asyncio.gather(
            self._embedding_provider.embed_documents([entry.content]),
            self._embedding_provider.embed_sparse([entry.content]),
        )

        # Add to Qdrant
        vector_name = self._embedding_provider.get_vector_name()
        sparse_vector_name = self._embedding_provider.get_sparse_vector_name()
        sparse_indices, sparse_values = sparse_embeddings[0]
        payload = {"document": entry.content, METADATA_PATH: entry.metadata}
        await self._client.upsert(
            collection_name=collection_name,
            points=[
                models.PointStruct(
                    id=uuid.uuid5(uuid.NAMESPACE_URL, entry.content).hex,
                    vector={
                        vector_name: dense_embeddings[0],
                        sparse_vector_name: models.SparseVector(
                            indices=sparse_indices,
                            values=sparse_values,
                        ),
                    },
                    payload=payload,
                )
            ],
        )

    async def search(
        self,
        query: str,
        *,
        collection_name: str | None = None,
        limit: int = 10,
        query_filter: models.Filter | None = None,
    ) -> list[Entry]:
        """
        Find points in the Qdrant collection. If there are no entries found, an empty list is returned.
        :param query: The query to use for the search.
        :param collection_name: The name of the collection to search in, optional. If not provided,
                                the default collection is used.
        :param limit: The maximum number of entries to return.
        :param query_filter: The filter to apply to the query, if any.

        :return: A list of entries found.
        """
        collection_name = collection_name or self._default_collection_name
        collection_exists = await self._client.collection_exists(collection_name)
        if not collection_exists:
            return []
        await self._validate_collection_schema(collection_name)

        # Embed the query
        # ToDo: instead of embedding text explicitly, use `models.Document`,
        # it should unlock usage of server-side inference.

        query_vector, sparse_query_vectors = await asyncio.gather(
            self._embedding_provider.embed_query(query),
            self._embedding_provider.embed_sparse([query]),
        )
        vector_name = self._embedding_provider.get_vector_name()
        sparse_vector_name = self._embedding_provider.get_sparse_vector_name()
        sparse_indices, sparse_values = sparse_query_vectors[0]
        prefetch_limit = limit * 3

        # Search in Qdrant
        search_results = await self._client.query_points(
            collection_name=collection_name,
            prefetch=[
                models.Prefetch(
                    query=query_vector,
                    using=vector_name,
                    limit=prefetch_limit,
                    filter=query_filter,
                ),
                models.Prefetch(
                    query=models.SparseVector(
                        indices=sparse_indices,
                        values=sparse_values,
                    ),
                    using=sparse_vector_name,
                    limit=prefetch_limit,
                    filter=query_filter,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit,
        )

        entries = [
            Entry(
                content=result.payload["document"],
                metadata=dict(result.payload.get(METADATA_PATH) or {}),
            )
            for result in search_results.points
        ]
        
        relevance_scores = await self._embedding_provider.rerank(
            query, [entry.content for entry in entries]
        )
        if len(relevance_scores) != len(entries):
            raise ValueError(
                "The reranker must return exactly one relevance score per result."
            )

        ranked_entries = []
        for entry, score in zip(entries, relevance_scores):
            entry.metadata[RELEVANCE_SCORE_KEY] = float(score)
            ranked_entries.append((entry, score))

        ranked_entries.sort(key=lambda item: item[1], reverse=True)
        return [entry for entry, _ in ranked_entries]

    async def _ensure_collection_exists(self, collection_name: str):
        """
        Ensure that the collection exists, creating it if necessary.
        :param collection_name: The name of the collection to ensure exists.
        """
        collection_exists = await self._client.collection_exists(collection_name)
        if collection_exists:
            await self._validate_collection_schema(collection_name)
            return

        # Create the collection with the appropriate vector size
        vector_size = self._embedding_provider.get_vector_size()

        # Use the vector names as defined in the embedding provider
        vector_name = self._embedding_provider.get_vector_name()
        sparse_vector_name = self._embedding_provider.get_sparse_vector_name()
        await self._client.create_collection(
            collection_name=collection_name,
            vectors_config={
                vector_name: models.VectorParams(
                    size=vector_size,
                    distance=models.Distance.COSINE,
                )
            },
            sparse_vectors_config={
                sparse_vector_name: models.SparseVectorParams(
                    modifier=models.Modifier.IDF,
                )
            },
        )

        # Create payload indexes if configured
        if self._field_indexes:
            for field_name, field_type in self._field_indexes.items():
                await self._client.create_payload_index(
                    collection_name=collection_name,
                    field_name=field_name,
                    field_schema=field_type,
                )

    async def _validate_collection_schema(self, collection_name: str):
        """Validate that an existing collection supports hybrid retrieval."""
        collection = await self._client.get_collection(collection_name)
        params = collection.config.params
        dense_vectors = params.vectors
        sparse_vectors = params.sparse_vectors or {}
        vector_name = self._embedding_provider.get_vector_name()
        sparse_vector_name = self._embedding_provider.get_sparse_vector_name()

        has_dense_vector = (
            isinstance(dense_vectors, dict) and vector_name in dense_vectors
        )
        has_sparse_vector = sparse_vector_name in sparse_vectors
        if not has_dense_vector or not has_sparse_vector:
            raise ValueError(
                f"Collection '{collection_name}' is not configured for hybrid "
                "dense + BM25 retrieval. Create a new collection and re-import "
                "the existing documents."
            )
