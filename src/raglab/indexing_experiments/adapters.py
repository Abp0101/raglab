"""Framework-native, isolated in-memory indexing adapters."""

import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, cast
from uuid import NAMESPACE_URL, uuid5

# Prevent Haystack's import-time telemetry before importing any native component.
os.environ["HAYSTACK_TELEMETRY_ENABLED"] = "false"

from haystack import Document as HaystackDocument
from haystack.components.preprocessors import DocumentSplitter
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.telemetry import _telemetry as haystack_telemetry
from langchain_core.documents import Document as LangChainDocument
from langchain_core.embeddings import Embeddings as LangChainEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from llama_index.core import Document as LlamaIndexDocument
from llama_index.core import VectorStoreIndex
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode

from raglab.chunking.benchmark import ChunkBenchmarkCase, parsed_document_for_case
from raglab.chunking.fixed import FixedTokenChunker
from raglab.chunking.tokenization import count_lexical_tokens, lexical_token_spans
from raglab.core.schemas import ChunkingConfig, ChunkingStrategy
from raglab.indexing_experiments.embeddings import (
    cosine_similarity,
    deterministic_hash_embedding,
)
from raglab.indexing_experiments.models import (
    ExperimentChunk,
    IndexingBenchmarkCase,
    IndexingExperimentControls,
    IndexingFramework,
)

haystack_telemetry.telemetry = None


@dataclass(frozen=True, slots=True)
class BuiltNativeIndex:
    """Normalized observations plus a framework-owned search function."""

    chunks: tuple[ExperimentChunk, ...]
    search_fn: Callable[[str, int], tuple[str, ...]]

    def search(self, query: str, top_k: int) -> tuple[str, ...]:
        return self.search_fn(query, top_k)


class NativeIndexAdapter(Protocol):
    """Build one isolated framework-native index for one benchmark case."""

    framework: IndexingFramework
    strategy: str
    index_backend: str
    size_unit: str

    def build(
        self,
        case: IndexingBenchmarkCase,
        controls: IndexingExperimentControls,
    ) -> BuiltNativeIndex: ...


class CustomNativeIndexAdapter:
    framework = IndexingFramework.CUSTOM
    strategy = "fixed-token"
    index_backend = "raglab-in-memory-cosine"
    size_unit = "lexical-tokens"

    def build(
        self,
        case: IndexingBenchmarkCase,
        controls: IndexingExperimentControls,
    ) -> BuiltNativeIndex:
        config = ChunkingConfig(
            strategy=ChunkingStrategy.FIXED_TOKEN,
            chunk_size=controls.chunk_size,
            chunk_overlap=controls.chunk_overlap,
        )
        structural_case = ChunkBenchmarkCase(
            dataset_version=case.dataset_version,
            case_id=case.case_id,
            category=case.category,
            text=case.text,
            relevant_passages=tuple(query.relevant_passage for query in case.queries),
        )
        native_chunks = FixedTokenChunker().chunk(
            parsed_document_for_case(structural_case),
            config,
        )
        chunks = tuple(
            ExperimentChunk(
                chunk_id=str(chunk.chunk_id),
                text=chunk.text,
                start=chunk.text_span.start if chunk.text_span else 0,
                end=chunk.text_span.end if chunk.text_span else len(chunk.text),
                token_count=chunk.token_count or count_lexical_tokens(chunk.text),
            )
            for chunk in native_chunks
        )
        vectors = {
            chunk.chunk_id: deterministic_hash_embedding(
                chunk.text,
                controls.embedding_dimensions,
            )
            for chunk in chunks
        }

        def search(query: str, top_k: int) -> tuple[str, ...]:
            query_vector = deterministic_hash_embedding(query, controls.embedding_dimensions)
            ranked = sorted(
                chunks,
                key=lambda chunk: (
                    -cosine_similarity(query_vector, vectors[chunk.chunk_id]),
                    chunk.chunk_id,
                ),
            )
            return tuple(chunk.chunk_id for chunk in ranked[:top_k])

        return BuiltNativeIndex(chunks=chunks, search_fn=search)


class _LangChainHashEmbeddings(LangChainEmbeddings):
    def __init__(self, dimensions: int) -> None:
        self._dimensions = dimensions

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [deterministic_hash_embedding(text, self._dimensions) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return deterministic_hash_embedding(text, self._dimensions)


class LangChainNativeIndexAdapter:
    framework = IndexingFramework.LANGCHAIN
    strategy = "recursive-character-token-budget"
    index_backend = "langchain-in-memory-vector-store"
    size_unit = "lexical-tokens"

    def build(
        self,
        case: IndexingBenchmarkCase,
        controls: IndexingExperimentControls,
    ) -> BuiltNativeIndex:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=controls.chunk_size,
            chunk_overlap=controls.chunk_overlap,
            length_function=count_lexical_tokens,
            add_start_index=True,
        )
        native_chunks = splitter.split_documents([LangChainDocument(page_content=case.text)])
        chunks = tuple(
            _experiment_chunk(
                framework=self.framework,
                case_id=case.case_id,
                index=index,
                text=document.page_content,
                start=int(document.metadata["start_index"]),
            )
            for index, document in enumerate(native_chunks)
        )
        store = InMemoryVectorStore(_LangChainHashEmbeddings(controls.embedding_dimensions))
        documents = [
            LangChainDocument(
                id=chunk.chunk_id,
                page_content=chunk.text,
                metadata={"chunk_id": chunk.chunk_id},
            )
            for chunk in chunks
        ]
        store.add_documents(documents, ids=[chunk.chunk_id for chunk in chunks])

        def search(query: str, top_k: int) -> tuple[str, ...]:
            results = store.similarity_search_with_score(query, k=top_k)
            return tuple(str(document.metadata["chunk_id"]) for document, _ in results)

        return BuiltNativeIndex(chunks=chunks, search_fn=search)


class _LlamaIndexHashEmbedding(BaseEmbedding):
    dimensions: int = 128

    @classmethod
    def class_name(cls) -> str:
        return "raglab-deterministic-hash-v1"

    def _get_text_embedding(self, text: str) -> list[float]:
        return deterministic_hash_embedding(text, self.dimensions)

    def _get_query_embedding(self, query: str) -> list[float]:
        return deterministic_hash_embedding(query, self.dimensions)

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return self._get_query_embedding(query)


class LlamaIndexNativeIndexAdapter:
    framework = IndexingFramework.LLAMAINDEX
    strategy = "sentence-splitter"
    index_backend = "llamaindex-vector-store-index"
    size_unit = "framework-tokens"

    def build(
        self,
        case: IndexingBenchmarkCase,
        controls: IndexingExperimentControls,
    ) -> BuiltNativeIndex:
        def tokenizer(text: str) -> list[str]:
            return [span.text for span in lexical_token_spans(text)]

        splitter = SentenceSplitter(
            chunk_size=controls.chunk_size,
            chunk_overlap=controls.chunk_overlap,
            tokenizer=tokenizer,
            include_metadata=False,
            include_prev_next_rel=False,
            id_func=lambda index, _: _chunk_id(self.framework, case.case_id, index),
        )
        nodes = splitter.get_nodes_from_documents([LlamaIndexDocument(text=case.text)])
        text_nodes = tuple(cast(TextNode, node) for node in nodes)
        chunks = tuple(
            ExperimentChunk(
                chunk_id=node.node_id,
                text=node.text,
                start=node.start_char_idx or 0,
                end=node.end_char_idx or len(node.text),
                token_count=count_lexical_tokens(node.text),
            )
            for node in text_nodes
        )
        index = VectorStoreIndex(
            nodes,
            embed_model=_LlamaIndexHashEmbedding(dimensions=controls.embedding_dimensions),
        )
        retriever = index.as_retriever(similarity_top_k=controls.top_k)

        def search(query: str, top_k: int) -> tuple[str, ...]:
            if top_k != controls.top_k:
                raise ValueError("LlamaIndex retrieval cutoff must match experiment controls")
            return tuple(result.node.node_id for result in retriever.retrieve(query))

        return BuiltNativeIndex(chunks=chunks, search_fn=search)


class HaystackNativeIndexAdapter:
    framework = IndexingFramework.HAYSTACK
    strategy = "document-splitter-word"
    index_backend = "haystack-in-memory-document-store"
    size_unit = "words"

    def build(
        self,
        case: IndexingBenchmarkCase,
        controls: IndexingExperimentControls,
    ) -> BuiltNativeIndex:
        splitter = DocumentSplitter(
            split_by="word",
            split_length=controls.chunk_size,
            split_overlap=controls.chunk_overlap,
            split_threshold=0,
        )
        native_chunks = splitter.run([HaystackDocument(content=case.text)])["documents"]
        chunks = tuple(
            _experiment_chunk(
                framework=self.framework,
                case_id=case.case_id,
                index=index,
                text=document.content or "",
                start=int(document.meta["split_idx_start"]),
            )
            for index, document in enumerate(native_chunks)
        )
        documents = [
            HaystackDocument(
                id=chunk.chunk_id,
                content=chunk.text,
                meta={"chunk_id": chunk.chunk_id},
                embedding=deterministic_hash_embedding(
                    chunk.text,
                    controls.embedding_dimensions,
                ),
            )
            for chunk in chunks
        ]
        store = InMemoryDocumentStore(embedding_similarity_function="cosine")
        store.write_documents(documents)
        retriever = InMemoryEmbeddingRetriever(store, top_k=controls.top_k)

        def search(query: str, top_k: int) -> tuple[str, ...]:
            results = retriever.run(
                query_embedding=deterministic_hash_embedding(
                    query,
                    controls.embedding_dimensions,
                ),
                top_k=top_k,
            )["documents"]
            return tuple(str(document.meta["chunk_id"]) for document in results)

        return BuiltNativeIndex(chunks=chunks, search_fn=search)


DEFAULT_ADAPTER_FACTORIES: dict[IndexingFramework, Callable[[], NativeIndexAdapter]] = {
    IndexingFramework.CUSTOM: CustomNativeIndexAdapter,
    IndexingFramework.LANGCHAIN: LangChainNativeIndexAdapter,
    IndexingFramework.LLAMAINDEX: LlamaIndexNativeIndexAdapter,
    IndexingFramework.HAYSTACK: HaystackNativeIndexAdapter,
}


def _experiment_chunk(
    *,
    framework: IndexingFramework,
    case_id: str,
    index: int,
    text: str,
    start: int,
) -> ExperimentChunk:
    return ExperimentChunk(
        chunk_id=_chunk_id(framework, case_id, index),
        text=text,
        start=start,
        end=start + len(text),
        token_count=count_lexical_tokens(text),
    )


def _chunk_id(framework: IndexingFramework, case_id: str, index: int) -> str:
    return str(uuid5(NAMESPACE_URL, f"raglab:indexing:{framework.value}:{case_id}:{index}"))
