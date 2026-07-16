"""Application-scoped construction of the local RAGLab service graph."""

from dataclasses import dataclass

from qdrant_client import AsyncQdrantClient
from redis.asyncio import Redis

from raglab.chunking.registry import create_chunker
from raglab.core.config import Settings
from raglab.core.health import InfrastructureReadinessProbe, ReadinessProbe
from raglab.core.interfaces import CatalogRepository, IngestionJobManager
from raglab.core.schemas import ChunkingConfig, FrameworkName
from raglab.database import (
    SQLAlchemyCatalogRepository,
    SQLAlchemyChunkRepository,
    SQLAlchemyDocumentRepository,
    SQLAlchemyIngestionJobRepository,
    create_engine,
    create_session_factory,
)
from raglab.embeddings import SentenceTransformerEmbeddingProvider
from raglab.generation.providers import create_llm_provider
from raglab.ingestion import BackgroundIngestionManager, LangChainIngestionPipeline
from raglab.ingestion.parsers import PyMuPDFParser
from raglab.ingestion.pipeline import DocumentIngestionPipeline
from raglab.ingestion.validation import PdfUploadValidator
from raglab.pipelines import CustomRAGPipeline, LangChainRAGPipeline, PipelineRegistry
from raglab.pipelines.langchain_rag import create_ollama_structured_model_factory
from raglab.reranking import CrossEncoderReranker
from raglab.retrieval import RetrievalService
from raglab.retrieval.parent_expansion import ParentChildContextExpander
from raglab.retrieval.qdrant_index import QdrantDenseRetriever, QdrantVectorIndexer
from raglab.retrieval.redis_bm25 import RedisBM25Indexer, RedisBM25Retriever


@dataclass(slots=True)
class ApiServices:
    """Dependencies retained for the full FastAPI process lifetime."""

    catalog: CatalogRepository
    pipelines: PipelineRegistry
    ingestion_jobs: IngestionJobManager
    readiness_probe: ReadinessProbe

    async def close(self) -> None:
        await self.ingestion_jobs.close()
        await self.readiness_probe.close()


def build_api_services(settings: Settings) -> ApiServices:
    """Build local-first production adapters without opening network connections eagerly."""
    engine = create_engine(settings.postgres_dsn)
    sessions = create_session_factory(engine)
    qdrant = AsyncQdrantClient(
        url=str(settings.qdrant_url),
        api_key=settings.qdrant_api_key,
        timeout=settings.qdrant_timeout_seconds,
        check_compatibility=False,
    )
    redis = Redis.from_url(str(settings.redis_dsn), decode_responses=True)
    readiness = InfrastructureReadinessProbe(database=engine, qdrant=qdrant, redis=redis)

    catalog = SQLAlchemyCatalogRepository(sessions)
    job_repository = SQLAlchemyIngestionJobRepository(sessions)
    documents = SQLAlchemyDocumentRepository(sessions)
    chunks = SQLAlchemyChunkRepository(sessions)
    validator = PdfUploadValidator(max_size_bytes=settings.max_upload_size_mb * 1024 * 1024)
    chunking_config = ChunkingConfig()
    embeddings = SentenceTransformerEmbeddingProvider(
        settings.embedding_model,
        batch_size=settings.embedding_batch_size,
    )
    vector_index = QdrantVectorIndexer(qdrant, settings.qdrant_collection)
    sparse_index = RedisBM25Indexer(redis, key_prefix=settings.bm25_key_prefix)
    ingestion = DocumentIngestionPipeline(
        validator=validator,
        parser=PyMuPDFParser(validator, max_pages=settings.max_pdf_pages),
        chunker=create_chunker(chunking_config.strategy),
        embedding_provider=embeddings,
        document_repository=documents,
        vector_indexer=vector_index,
        sparse_indexer=sparse_index,
        chunking_config=chunking_config,
    )
    retrieval = RetrievalService(
        embedding_provider=embeddings,
        dense_retriever=QdrantDenseRetriever(qdrant, settings.qdrant_collection),
        sparse_retriever=RedisBM25Retriever(redis, key_prefix=settings.bm25_key_prefix),
        reranker=CrossEncoderReranker(
            settings.reranker_model,
            batch_size=settings.reranker_batch_size,
        ),
        context_expander=ParentChildContextExpander(chunks),
    )
    custom = CustomRAGPipeline(
        ingestion=ingestion,
        retrieval=retrieval,
        llm=create_llm_provider(settings),
        default_model=settings.llm_model,
    )
    langchain = LangChainRAGPipeline(
        ingestion=LangChainIngestionPipeline(
            validator=validator,
            parser=PyMuPDFParser(validator, max_pages=settings.max_pdf_pages),
            embedding_provider=embeddings,
            document_repository=documents,
            vector_indexer=vector_index,
            sparse_indexer=sparse_index,
        ),
        retrieval=retrieval,
        model_factory=create_ollama_structured_model_factory(
            str(settings.ollama_base_url),
            timeout_seconds=settings.llm_timeout_seconds,
        ),
        default_model=settings.llm_model,
    )
    return ApiServices(
        catalog=catalog,
        pipelines=PipelineRegistry(
            {
                FrameworkName.CUSTOM: custom,
                FrameworkName.LANGCHAIN: langchain,
            }
        ),
        ingestion_jobs=BackgroundIngestionManager(
            job_repository,
            custom,
            max_concurrency=settings.ingestion_concurrency,
        ),
        readiness_probe=readiness,
    )
