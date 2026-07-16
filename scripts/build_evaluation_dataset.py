"""Rebuild the deterministic synthetic biomedical/technical evaluation dataset."""

import asyncio
import hashlib
import json
from pathlib import Path
from typing import Any, cast
from uuid import NAMESPACE_URL, UUID, uuid5

import pymupdf

from raglab.chunking import RecursiveCharacterChunker
from raglab.core.schemas import (
    ChunkingConfig,
    DocumentInput,
    EvaluationDifficulty,
    EvaluationQuestion,
)
from raglab.ingestion.parsers import PyMuPDFParser
from raglab.ingestion.validation import PdfUploadValidator

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "datasets" / "evaluation" / "v1"
COLLECTION_ID = uuid5(NAMESPACE_URL, "raglab:evaluation:v1")
PUBLISHED_ON = "2026-07-16"

SOURCES: tuple[dict[str, Any], ...] = (
    {
        "file_name": "wearable_rehabilitation.pdf",
        "title": "Wearable Rehabilitation Prototype",
        "author": "RAGLab Synthetic Research Group",
        "pages": (
            (
                "METHODS",
                "The prototype used a six-axis IMU mounted on the shank and sampled motion at "
                "100 Hz. A force-sensitive resistor under the heel recorded foot contact.",
            ),
            (
                "RESULTS",
                "Ten healthy volunteers completed the walking protocol. Mean step-count error "
                "was 3.2 percent, and measured battery life was 6 hours.",
            ),
        ),
    },
    {
        "file_name": "rehabilitation_safety.pdf",
        "title": "Rehabilitation Device Safety Guidance",
        "author": "RAGLab Synthetic Clinical Engineering Group",
        "pages": (
            (
                "SAFETY",
                "The device is intended for supervised rehabilitation research and is not a "
                "diagnostic device. Stop use if skin irritation or pain occurs.",
            ),
            (
                "CALIBRATION",
                "Calibrate the IMU while stationary for 10 seconds. Zero the force-sensitive "
                "resistor with no load before each session.",
            ),
        ),
    },
    {
        "file_name": "low_power_sensor.pdf",
        "title": "Low-Power Gait Sensor Prototype",
        "author": "RAGLab Synthetic Sensor Group",
        "pages": (
            (
                "DESIGN",
                "The low-power prototype sampled IMU motion at 50 Hz to reduce energy use. "
                "The report does not evaluate diagnostic performance.",
            ),
        ),
    },
)


def _write_pdf(path: Path, source: dict[str, Any]) -> None:
    pdf = pymupdf.open()  # type: ignore[no-untyped-call]
    for heading, body in source["pages"]:
        page = pdf.new_page()
        page.insert_text((72, 72), heading, fontsize=18)
        page.insert_textbox((72, 105, 520, 720), body, fontsize=11)
    pdf.set_metadata(
        {
            "title": source["title"],
            "author": source["author"],
            "creationDate": "D:20260716000000Z",
            "modDate": "D:20260716000000Z",
        }
    )
    pdf.save(path, garbage=4, deflate=True, no_new_id=True)  # type: ignore[no-untyped-call]
    pdf.close()  # type: ignore[no-untyped-call]


async def _build() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    parser = PyMuPDFParser(PdfUploadValidator(max_size_bytes=5_000_000), max_pages=20)
    chunker = RecursiveCharacterChunker()
    config = ChunkingConfig()
    corpus_documents: list[dict[str, Any]] = []
    chunks_by_file_page: dict[tuple[str, int], UUID] = {}
    documents_by_file: dict[str, UUID] = {}

    for source in SOURCES:
        path = OUTPUT / cast(str, source["file_name"])
        _write_pdf(path, source)
        content = path.read_bytes()
        parsed = await parser.parse(
            DocumentInput(
                file_name=path.name,
                content=content,
                collection_id=COLLECTION_ID,
            )
        )
        chunks = tuple(chunker.chunk(parsed, config))
        documents_by_file[path.name] = parsed.document.document_id
        for chunk in chunks:
            page_number = chunk.metadata.page_number
            if page_number is not None:
                chunks_by_file_page[(path.name, page_number)] = chunk.chunk_id
        corpus_documents.append(
            {
                "file_name": path.name,
                "display_title": parsed.document.display_title,
                "sha256": hashlib.sha256(content).hexdigest(),
                "document_id": str(parsed.document.document_id),
                "chunks": [
                    {
                        "chunk_id": str(chunk.chunk_id),
                        "page_number": chunk.metadata.page_number,
                        "section_heading": chunk.metadata.section_heading,
                        "text": chunk.text,
                    }
                    for chunk in chunks
                ],
            }
        )

    questions = _questions(documents_by_file, chunks_by_file_page)
    questions_bytes = b"".join(
        (json.dumps(question.model_dump(mode="json"), sort_keys=True) + "\n").encode()
        for question in questions
    )
    (OUTPUT / "questions.jsonl").write_bytes(questions_bytes)
    (OUTPUT / "manifest.json").write_text(
        json.dumps(
            {
                "name": "raglab-synthetic-biomedical-technical",
                "version": "1.0.0",
                "description": (
                    "Small synthetic corpus for deterministic RAG harness validation; not a "
                    "clinical or real-world performance benchmark."
                ),
                "collection_id": str(COLLECTION_ID),
                "published_on": PUBLISHED_ON,
                "question_count": len(questions),
                "questions_sha256": hashlib.sha256(questions_bytes).hexdigest(),
                "domains": ["biomedical-engineering", "wearable-sensors", "rag-safety"],
                "license": "CC0-1.0",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (OUTPUT / "corpus.json").write_text(
        json.dumps(
            {
                "collection_id": str(COLLECTION_ID),
                "collection_name": "RAGLab Evaluation v1",
                "documents": corpus_documents,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _questions(
    documents: dict[str, UUID],
    chunks: dict[tuple[str, int], UUID],
) -> tuple[EvaluationQuestion, ...]:
    wearable = "wearable_rehabilitation.pdf"
    safety = "rehabilitation_safety.pdf"
    low_power = "low_power_sensor.pdf"

    def question(
        question_id: str,
        text: str,
        facts: tuple[str, ...],
        sources: tuple[tuple[str, int], ...],
        *,
        answerable: bool = True,
        category: str = "factoid",
        difficulty: EvaluationDifficulty = EvaluationDifficulty.EASY,
    ) -> EvaluationQuestion:
        return EvaluationQuestion(
            question_id=question_id,
            dataset_version="1.0.0",
            question=text,
            expected_key_facts=facts,
            relevant_document_ids=tuple(dict.fromkeys(documents[file] for file, _ in sources)),
            relevant_chunk_ids=tuple(chunks[source] for source in sources),
            expected_citation_chunk_ids=tuple(chunks[source] for source in sources),
            answerable=answerable,
            category=category,
            difficulty=difficulty,
        )

    return (
        question(
            "wearable-sampling-rate",
            "At what rate did the wearable rehabilitation prototype sample motion?",
            ("100 Hz",),
            ((wearable, 1),),
        ),
        question(
            "wearable-foot-contact",
            "Which sensor recorded foot contact, and where was it placed?",
            ("force-sensitive resistor", "under the heel"),
            ((wearable, 1),),
            difficulty=EvaluationDifficulty.MEDIUM,
        ),
        question(
            "wearable-battery-life",
            "What battery life was measured for the wearable prototype?",
            ("6 hours",),
            ((wearable, 2),),
        ),
        question(
            "calibration-procedure",
            "How should the IMU and force-sensitive resistor be calibrated?",
            ("stationary for 10 seconds", "no load before each session"),
            ((safety, 2),),
            difficulty=EvaluationDifficulty.MEDIUM,
        ),
        question(
            "diagnostic-intent",
            "Is the rehabilitation device intended for diagnosis?",
            ("supervised rehabilitation research", "not a diagnostic device"),
            ((safety, 1),),
        ),
        question(
            "conflicting-sampling-rates",
            "Did every prototype use the same IMU sampling rate?",
            ("100 Hz", "50 Hz"),
            ((wearable, 1), (low_power, 1)),
            category="conflicting-evidence",
            difficulty=EvaluationDifficulty.HARD,
        ),
        question(
            "unanswerable-wireless-protocol",
            "Which wireless protocol did the wearable prototype use?",
            (),
            (),
            answerable=False,
            category="unanswerable",
            difficulty=EvaluationDifficulty.MEDIUM,
        ),
    )


if __name__ == "__main__":
    asyncio.run(_build())
