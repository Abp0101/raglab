from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from uuid import UUID, uuid4

from apps.api.main import create_app
from apps.api.runtime import ApiServices
from fastapi.testclient import TestClient

from raglab.core.config import ApiKeyCredentialSettings, Settings
from raglab.core.exceptions import CollectionNotFoundError, DocumentNotFoundError
from raglab.core.pagination import CursorKind, decode_cursor, encode_cursor
from raglab.core.schemas import (
    AuthRole,
    Collection,
    CollectionCreate,
    CursorPage,
    Document,
    DocumentDeletionResult,
    DocumentInput,
    DocumentStatus,
    EvidenceStatus,
    FrameworkName,
    IngestionJob,
    IngestionJobStatus,
    IngestionResult,
    LatencyMetrics,
    PipelineCapabilities,
    PipelineConfig,
    QueryRequest,
    RAGResponse,
)
from raglab.pipelines import PipelineRegistry
from raglab.security import ApiKeyAuthenticator

VIEWER_KEY = "viewer-key-that-is-at-least-32-characters"
EDITOR_KEY = "editor-key-that-is-at-least-32-characters"
ADMIN_KEY = "admin-key-that-is-at-least-32-characters"


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

    async def list_collections(
        self,
        *,
        limit: int = 20,
        cursor: str | None = None,
    ) -> CursorPage[Collection]:
        position = decode_cursor(cursor, kind=CursorKind.COLLECTIONS, scope=None)
        items = sorted(
            self.collections.values(),
            key=lambda item: (item.created_at, item.collection_id),
        )
        if position is not None:
            items = [
                item
                for item in items
                if (item.created_at, item.collection_id) > (position.ordered_at, position.item_id)
            ]
        selected = items[:limit]
        return CursorPage(
            items=tuple(selected),
            next_cursor=(
                encode_cursor(
                    kind=CursorKind.COLLECTIONS,
                    scope=None,
                    ordered_at=selected[-1].created_at,
                    item_id=selected[-1].collection_id,
                )
                if len(items) > limit
                else None
            ),
        )

    async def get_collection(self, collection_id: UUID) -> Collection:
        try:
            return self.collections[collection_id]
        except KeyError as error:
            raise CollectionNotFoundError(f"collection {collection_id} does not exist") from error

    async def list_documents(
        self,
        collection_id: UUID,
        *,
        limit: int = 20,
        cursor: str | None = None,
    ) -> CursorPage[Document]:
        await self.get_collection(collection_id)
        position = decode_cursor(cursor, kind=CursorKind.DOCUMENTS, scope=collection_id)
        items = sorted(
            (
                document
                for document in self.documents.values()
                if document.collection_id == collection_id
            ),
            key=lambda item: (item.uploaded_at, item.document_id),
        )
        if position is not None:
            items = [
                item
                for item in items
                if (item.uploaded_at, item.document_id) > (position.ordered_at, position.item_id)
            ]
        selected = items[:limit]
        return CursorPage(
            items=tuple(selected),
            next_cursor=(
                encode_cursor(
                    kind=CursorKind.DOCUMENTS,
                    scope=collection_id,
                    ordered_at=selected[-1].uploaded_at,
                    item_id=selected[-1].document_id,
                )
                if len(items) > limit
                else None
            ),
        )

    async def get_document(self, document_id: UUID) -> Document:
        try:
            return self.documents[document_id]
        except KeyError as error:
            raise DocumentNotFoundError(f"document {document_id} does not exist") from error


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


class MemoryJobManager:
    def __init__(self) -> None:
        self.jobs: dict[UUID, IngestionJob] = {}

    async def start(self) -> None:
        return None

    async def submit(self, document: DocumentInput) -> IngestionJob:
        now = datetime.now(UTC)
        job = IngestionJob(
            job_id=uuid4(),
            collection_id=document.collection_id,
            file_name=document.file_name,
            status=IngestionJobStatus.QUEUED,
            created_at=now,
            updated_at=now,
        )
        self.jobs[job.job_id] = job
        return job

    async def get(self, job_id: UUID) -> IngestionJob:
        return self.jobs[job_id]

    async def list_for_collection(
        self,
        collection_id: UUID,
        *,
        limit: int = 20,
        cursor: str | None = None,
    ) -> CursorPage[IngestionJob]:
        position = decode_cursor(
            cursor,
            kind=CursorKind.INGESTION_JOBS,
            scope=collection_id,
        )
        items = sorted(
            (job for job in self.jobs.values() if job.collection_id == collection_id),
            key=lambda item: (item.created_at, item.job_id),
        )
        if position is not None:
            items = [
                item
                for item in items
                if (item.created_at, item.job_id) > (position.ordered_at, position.item_id)
            ]
        selected = items[:limit]
        return CursorPage(
            items=tuple(selected),
            next_cursor=(
                encode_cursor(
                    kind=CursorKind.INGESTION_JOBS,
                    scope=collection_id,
                    ordered_at=selected[-1].created_at,
                    item_id=selected[-1].job_id,
                )
                if len(items) > limit
                else None
            ),
        )

    async def close(self) -> None:
        return None


class MemoryDeletionManager:
    def __init__(self, catalog: MemoryCatalog) -> None:
        self.catalog = catalog

    async def delete(self, document_id: UUID) -> DocumentDeletionResult:
        try:
            document = self.catalog.documents.pop(document_id)
        except KeyError as error:
            raise DocumentNotFoundError(f"document {document_id} does not exist") from error
        return DocumentDeletionResult(
            document_id=document.document_id,
            collection_id=document.collection_id,
            deleted_chunk_count=2,
        )


def make_client(
    settings: Settings | None = None,
) -> tuple[TestClient, MemoryCatalog, StubPipeline]:
    app_settings = settings or Settings(environment="test", _env_file=None)
    catalog = MemoryCatalog()
    pipeline = StubPipeline()
    services = ApiServices(
        catalog=catalog,
        authenticator=ApiKeyAuthenticator(
            enabled=app_settings.auth_enabled,
            credentials=app_settings.auth_api_keys,
        ),
        pipelines=PipelineRegistry({FrameworkName.CUSTOM: pipeline}),
        ingestion_jobs=MemoryJobManager(),
        document_deletion=MemoryDeletionManager(catalog),
        readiness_probe=StubReadinessProbe(),
    )
    app = create_app(app_settings, services=services)
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
    assert listed.json()["items"][0]["name"] == "Biomedical papers"
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


def test_background_upload_returns_pollable_durable_job_contract() -> None:
    client, _, _ = make_client()

    with client:
        collection = client.post("/collections", json={"name": "Async papers"}).json()
        accepted = client.post(
            f"/collections/{collection['collection_id']}/ingestion-jobs",
            files={"file": ("paper.pdf", b"%PDF-1.7\nlocal-test", "application/pdf")},
        )
        polled = client.get(f"/ingestion-jobs/{accepted.json()['job_id']}")
        listed = client.get(f"/collections/{collection['collection_id']}/ingestion-jobs")

    assert accepted.status_code == 202
    assert accepted.json()["status"] == "queued"
    assert polled.status_code == 200
    assert polled.json()["file_name"] == "paper.pdf"
    assert listed.json()["items"][0]["job_id"] == accepted.json()["job_id"]


def test_delete_document_returns_coordinated_cleanup_result() -> None:
    client, catalog, _ = make_client()

    with client:
        collection = client.post("/collections", json={"name": "Delete papers"}).json()
        document = Document(
            document_id=uuid4(),
            collection_id=UUID(collection["collection_id"]),
            file_name="delete.pdf",
            display_title="Delete me",
            uploaded_at=datetime.now(UTC),
            file_type="application/pdf",
            content_hash="e" * 64,
            status=DocumentStatus.READY,
        )
        catalog.documents[document.document_id] = document
        response = client.delete(f"/documents/{document.document_id}")
        missing = client.get(f"/documents/{document.document_id}")

    assert response.status_code == 200
    assert response.json() == {
        "document_id": str(document.document_id),
        "collection_id": str(document.collection_id),
        "deleted_chunk_count": 2,
    }
    assert missing.status_code == 404


def test_authentication_and_role_permissions_protect_routes() -> None:
    settings = Settings(
        environment="test",
        auth_enabled=True,
        auth_api_keys=[
            ApiKeyCredentialSettings(name="test-viewer", role=AuthRole.VIEWER, key=VIEWER_KEY),
            ApiKeyCredentialSettings(name="test-editor", role=AuthRole.EDITOR, key=EDITOR_KEY),
            ApiKeyCredentialSettings(name="test-admin", role=AuthRole.ADMIN, key=ADMIN_KEY),
        ],
        _env_file=None,
    )
    client, catalog, _ = make_client(settings)

    with client:
        health = client.get("/health/live")
        missing = client.get("/pipelines")
        invalid = client.get("/pipelines", headers={"Authorization": "Bearer invalid"})
        viewer_headers = {"Authorization": f"Bearer {VIEWER_KEY}"}
        viewer_read = client.get("/pipelines", headers=viewer_headers)
        viewer_write = client.post(
            "/collections",
            json={"name": "Denied"},
            headers=viewer_headers,
        )
        editor_headers = {"Authorization": f"Bearer {EDITOR_KEY}"}
        created = client.post(
            "/collections",
            json={"name": "Authorized"},
            headers=editor_headers,
        )
        document = Document(
            document_id=uuid4(),
            collection_id=UUID(created.json()["collection_id"]),
            file_name="protected.pdf",
            display_title="Protected",
            uploaded_at=datetime.now(UTC),
            file_type="application/pdf",
            content_hash="a" * 64,
            status=DocumentStatus.READY,
        )
        catalog.documents[document.document_id] = document
        editor_delete = client.delete(
            f"/documents/{document.document_id}",
            headers=editor_headers,
        )
        admin_headers = {"Authorization": f"Bearer {ADMIN_KEY}"}
        admin_delete = client.delete(
            f"/documents/{document.document_id}",
            headers=admin_headers,
        )
        identity = client.get("/auth/me", headers=viewer_headers)

    assert health.status_code == 200
    assert missing.status_code == 401
    assert missing.headers["WWW-Authenticate"] == "Bearer"
    assert invalid.status_code == 401
    assert viewer_read.status_code == 200
    assert viewer_write.status_code == 403
    assert created.status_code == 201
    assert editor_delete.status_code == 403
    assert admin_delete.status_code == 200
    assert identity.json()["subject"] == "test-viewer"
    assert identity.json()["role"] == "viewer"


def test_openapi_declares_bearer_security_but_health_remains_public() -> None:
    client, _, _ = make_client()

    with client:
        schema = client.get("/openapi.json").json()

    assert "RAGLabApiKey" in schema["components"]["securitySchemes"]
    assert schema["paths"]["/collections"]["post"]["security"] == [{"RAGLabApiKey": []}]
    assert "security" not in schema["paths"]["/health/live"]["get"]


def test_cursor_pagination_and_scope_validation() -> None:
    client, _, _ = make_client()

    with client:
        for name in ("First", "Second", "Third"):
            client.post("/collections", json={"name": name})
        first_page = client.get("/collections", params={"limit": 2})
        cursor = first_page.json()["next_cursor"]
        second_page = client.get("/collections", params={"limit": 2, "cursor": cursor})
        collection_id = first_page.json()["items"][0]["collection_id"]
        wrong_scope = client.get(
            f"/collections/{collection_id}/documents",
            params={"cursor": cursor},
        )
        invalid_limit = client.get("/collections", params={"limit": 101})

    assert [item["name"] for item in first_page.json()["items"]] == ["First", "Second"]
    assert [item["name"] for item in second_page.json()["items"]] == ["Third"]
    assert second_page.json()["next_cursor"] is None
    assert wrong_scope.status_code == 422
    assert wrong_scope.json()["error"]["type"] == "InvalidCursor"
    assert invalid_limit.status_code == 422


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


def test_stream_query_emits_lifecycle_then_only_validated_result() -> None:
    client, _, _ = make_client()

    with client:
        collection = client.post("/collections", json={"name": "Streaming"}).json()
        response = client.post(
            "/query/stream",
            json={
                "query": "What does the evidence show?",
                "framework": "custom",
                "collection_id": collection["collection_id"],
            },
        )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "event: query.accepted" in response.text
    assert "event: query.result" in response.text
    assert "A grounded local answer." in response.text


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
    assert listed.json()["items"][0]["document_id"] == str(document.document_id)
    assert fetched.json()["status"] == "ready"
