from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from uuid import UUID, uuid4

from apps.api.main import create_app
from apps.api.runtime import ApiServices
from fastapi.testclient import TestClient

from raglab.core.config import Settings
from raglab.core.exceptions import CollectionNotFoundError
from raglab.core.schemas import (
    Collection,
    CollectionCreate,
    Document,
    DocumentInput,
    DocumentStatus,
    EvidenceStatus,
    FrameworkName,
    IngestionResult,
    LatencyMetrics,
    PipelineCapabilities,
    PipelineConfig,
    QueryRequest,
    RAGResponse,
)
from raglab.pipelines import PipelineRegistry


class StubReadinessProbe:
    async def check(self) -> Mapping[str, bool]:
        return {"postgres": True, "qdrant": True, "redis": True}

    async def close(self) -> None:
        return None


class MemoryCatalog:
    def __init__(self) -> None:
        self.collections: dict[UUID, Collection] = {}
        self.documents: dict[UUID, Document] = {}

    async def create_collection(self, request: CollectionCreate) -> Collection:
        now = datetime.now(UTC)
        collection = Collection(
            collection_id=uuid4(),
            name=request.name,
            description=request.description,
            created_at=now,
            updated_at=now,
        )
        self.collections[collection.collection_id] = collection
        return collection

    async def list_collections(self) -> Sequence[Collection]:
        return tuple(self.collections.values())

    async def get_collection(self, collection_id: UUID) -> Collection:
        try:
            return self.collections[collection_id]
        except KeyError as error:
            raise CollectionNotFoundError(f"collection {collection_id} does not exist") from error

    async def list_documents(self, collection_id: UUID) -> Sequence[Document]:
        await self.get_collection(collection_id)
        return tuple(
            document
            for document in self.documents.values()
            if document.collection_id == collection_id
        )

    async def get_document(self, document_id: UUID) -> Document:
        return self.documents[document_id]


class StubPipeline:
    def __init__(self) -> None:
        self.ingested: tuple[DocumentInput, ...] = ()
        self.queries: tuple[QueryRequest, ...] = ()

    @property
    def config(self) -> PipelineConfig:
        return PipelineConfig()

    @property
    def capabilities(self) -> PipelineCapabilities:
        return PipelineCapabilities(
            sparse_retrieval=True,
            hybrid_retrieval=True,
            reranking=True,
            metadata_filtering=True,
        )

    async def ingest(self, documents: Sequence[DocumentInput]) -> Sequence[IngestionResult]:
        self.ingested = tuple(documents)
        document = self.ingested[0]
        return (
            IngestionResult(
                document_id=uuid4(),
                collection_id=document.collection_id,
                page_count=1,
                chunk_count=2,
                duration_ms=4,
                parser="stub",
                chunking_strategy="stub",
                embedding_model="local-stub",
            ),
        )

    async def query(self, request: QueryRequest) -> RAGResponse:
        self.queries = (*self.queries, request)
        return RAGResponse(
            answer="A grounded local answer.",
            framework=FrameworkName.CUSTOM,
            model="local-stub",
            latency=LatencyMetrics(total_ms=3),
            evidence_status=EvidenceStatus.SUFFICIENT,
            confidence=0.9,
        )


def make_client() -> tuple[TestClient, MemoryCatalog, StubPipeline]:
    catalog = MemoryCatalog()
    pipeline = StubPipeline()
    services = ApiServices(
        catalog=catalog,
        pipelines=PipelineRegistry({FrameworkName.CUSTOM: pipeline}),
        readiness_probe=StubReadinessProbe(),
    )
    app = create_app(Settings(environment="test", _env_file=None), services=services)
    return TestClient(app), catalog, pipeline


def test_collection_create_list_and_get() -> None:
    client, _, _ = make_client()

    with client:
        created = client.post(
            "/collections", json={"name": "Biomedical papers", "description": "Local corpus"}
        )
        collection_id = created.json()["collection_id"]
        listed = client.get("/collections")
        fetched = client.get(f"/collections/{collection_id}")

    assert created.status_code == 201
    assert listed.json()[0]["name"] == "Biomedical papers"
    assert fetched.json()["description"] == "Local corpus"


def test_pdf_upload_uses_registered_custom_pipeline() -> None:
    client, catalog, pipeline = make_client()

    with client:
        collection = client.post("/collections", json={"name": "Wearables"}).json()
        response = client.post(
            f"/collections/{collection['collection_id']}/documents",
            files={"file": ("paper.pdf", b"%PDF-1.7\nlocal-test", "application/pdf")},
            data={"display_title": "Sensor study"},
        )

    assert response.status_code == 201
    assert response.json()["embedding_model"] == "local-stub"
    assert pipeline.ingested[0].display_title == "Sensor study"
    assert pipeline.ingested[0].collection_id in catalog.collections


def test_query_and_pipeline_discovery_use_shared_contracts() -> None:
    client, _, pipeline = make_client()

    with client:
        collection = client.post("/collections", json={"name": "Clinical"}).json()
        pipelines = client.get("/pipelines")
        response = client.post(
            "/query",
            json={
                "query": "What does the evidence show?",
                "framework": "custom",
                "collection_id": collection["collection_id"],
            },
        )

    assert pipelines.status_code == 200
    assert [item["framework"] for item in pipelines.json()] == [
        "custom",
        "langchain",
        "langgraph",
        "llamaindex",
        "haystack",
    ]
    assert pipelines.json()[0]["available"] is True
    assert all(item["available"] is False for item in pipelines.json()[1:])
    assert response.status_code == 200
    assert response.json()["answer"] == "A grounded local answer."
    assert response.json()["estimated_cost"] is None
    assert pipeline.queries[0].framework is FrameworkName.CUSTOM


def test_expected_domain_error_has_stable_public_shape() -> None:
    client, _, _ = make_client()
    missing = uuid4()

    with client:
        response = client.get(f"/collections/{missing}")

    assert response.status_code == 404
    assert response.json() == {
        "error": {
            "type": "CollectionNotFound",
            "message": f"collection {missing} does not exist",
        }
    }


def test_empty_upload_and_unavailable_framework_return_safe_errors() -> None:
    client, _, _ = make_client()

    with client:
        collection = client.post("/collections", json={"name": "Safety"}).json()
        empty = client.post(
            f"/collections/{collection['collection_id']}/documents",
            files={"file": ("empty.pdf", b"", "application/pdf")},
        )
        unavailable = client.post(
            "/query",
            json={
                "query": "What does the evidence show?",
                "framework": "langchain",
                "collection_id": collection["collection_id"],
            },
        )

    assert empty.status_code == 422
    assert empty.json()["error"]["type"] == "DocumentValidation"
    assert unavailable.status_code == 501
    assert unavailable.json()["error"]["type"] == "UnsupportedFramework"


def test_document_metadata_routes() -> None:
    client, catalog, _ = make_client()
    now = datetime.now(UTC)

    with client:
        collection = client.post("/collections", json={"name": "Documents"}).json()
        collection_id = UUID(collection["collection_id"])
        document = Document(
            document_id=uuid4(),
            collection_id=collection_id,
            file_name="paper.pdf",
            display_title="Paper",
            uploaded_at=now,
            file_type="application/pdf",
            content_hash="a" * 64,
            status=DocumentStatus.READY,
        )
        catalog.documents[document.document_id] = document
        listed = client.get(f"/collections/{collection_id}/documents")
        fetched = client.get(f"/documents/{document.document_id}")

    assert listed.status_code == 200
    assert listed.json()[0]["document_id"] == str(document.document_id)
    assert fetched.json()["status"] == "ready"
