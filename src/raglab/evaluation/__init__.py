"""Deterministic, provider-independent RAG evaluation harness."""

from raglab.evaluation.datasets import load_dataset
from raglab.evaluation.metrics import evaluate_response
from raglab.evaluation.reports import build_report, write_report
from raglab.evaluation.runner import EvaluationRunner

__all__ = [
    "EvaluationRunner",
    "build_report",
    "evaluate_response",
    "load_dataset",
    "write_report",
]
