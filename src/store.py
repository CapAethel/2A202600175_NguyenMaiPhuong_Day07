from __future__ import annotations

from typing import Any, Callable

from .chunking import _dot
from .embeddings import _mock_embed
from .models import Document


class EmbeddingStore:
    """
    A vector store for text chunks.

    Tries to use ChromaDB if available; falls back to an in-memory store.
    The embedding_fn parameter allows injection of mock embeddings for tests.
    """

    def __init__(
        self,
        collection_name: str = "documents",
        embedding_fn: Callable[[str], list[float]] | None = None,
    ) -> None:
        self._embedding_fn = embedding_fn or _mock_embed
        self._collection_name = collection_name
        self._use_chroma = False
        self._store: list[dict[str, Any]] = []
        self._collection = None
        self._next_index = 0

        try:
            import chromadb  # noqa: F401

            client = chromadb.Client()
            self._collection = client.get_or_create_collection(name=collection_name)
            self._use_chroma = True
        except Exception:
            self._use_chroma = False
            self._collection = None

    def _make_record(self, doc: Document) -> dict[str, Any]:
        embedding = self._embedding_fn(doc.content)
        return {
            'id': doc.id,
            'content': doc.content,
            'embedding': embedding,
            'metadata': dict(doc.metadata, doc_id=doc.id),
        }

    def _search_records(self, query: str, records: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        query_embedding = self._embedding_fn(query)
        scored = []
        for rec in records:
            score = _dot(query_embedding, rec['embedding'])
            scored.append({**rec, 'score': score})
        scored.sort(key=lambda x: x['score'], reverse=True)
        return scored[:top_k]

    def add_documents(self, docs: list[Document]) -> None:
        """
        Embed each document's content and store it.

        For ChromaDB: use collection.add(ids=[...], documents=[...], embeddings=[...])
        For in-memory: append dicts to self._store
        """
        if self._use_chroma and self._collection is not None:
            ids = [doc.id for doc in docs]
            documents = [doc.content for doc in docs]
            embeddings = [self._embedding_fn(doc.content) for doc in docs]
            metadatas = [dict(doc.metadata, doc_id=doc.id) for doc in docs]
            self._collection.add(ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas)
        else:
            for doc in docs:
                record = self._make_record(doc)
                self._store.append(record)

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """
        Find the top_k most similar documents to query.

        For in-memory: compute dot product of query embedding vs all stored embeddings.
        """
        if self._use_chroma and self._collection is not None:
            query_embedding = self._embedding_fn(query)
            results = self._collection.query(query_embeddings=[query_embedding], n_results=top_k)
            output = []
            for i in range(len(results['ids'][0])):
                output.append({
                    'id': results['ids'][0][i],
                    'content': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i] if results.get('metadatas') else {},
                    'score': 1.0 - (results['distances'][0][i] if results.get('distances') else 0),
                })
            output.sort(key=lambda x: x['score'], reverse=True)
            return output
        else:
            return self._search_records(query, self._store, top_k)

    def get_collection_size(self) -> int:
        """Return the total number of stored chunks."""
        if self._use_chroma and self._collection is not None:
            return self._collection.count()
        return len(self._store)

    def search_with_filter(self, query: str, top_k: int = 3, metadata_filter: dict = None) -> list[dict]:
        """
        Search with optional metadata pre-filtering.

        First filter stored chunks by metadata_filter, then run similarity search.
        """
        if self._use_chroma and self._collection is not None:
            query_embedding = self._embedding_fn(query)
            where = metadata_filter if metadata_filter else None
            n = top_k
            results = self._collection.query(
                query_embeddings=[query_embedding], n_results=n, where=where
            )
            output = []
            for i in range(len(results['ids'][0])):
                output.append({
                    'id': results['ids'][0][i],
                    'content': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i] if results.get('metadatas') else {},
                    'score': 1.0 - (results['distances'][0][i] if results.get('distances') else 0),
                })
            output.sort(key=lambda x: x['score'], reverse=True)
            return output
        else:
            if metadata_filter:
                filtered = [
                    rec for rec in self._store
                    if all(rec.get('metadata', {}).get(k) == v for k, v in metadata_filter.items())
                ]
            else:
                filtered = self._store
            return self._search_records(query, filtered, top_k)

    def delete_document(self, doc_id: str) -> bool:
        """
        Remove all chunks belonging to a document.

        Returns True if any chunks were removed, False otherwise.
        """
        if self._use_chroma and self._collection is not None:
            # Get all items to find matching doc_id
            all_items = self._collection.get(where={'doc_id': doc_id})
            if all_items['ids']:
                self._collection.delete(ids=all_items['ids'])
                return True
            return False
        else:
            original_len = len(self._store)
            self._store = [rec for rec in self._store if rec.get('metadata', {}).get('doc_id') != doc_id]
            return len(self._store) < original_len
