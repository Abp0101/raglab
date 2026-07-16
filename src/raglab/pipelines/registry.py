"""Explicit registry for selectable RAG pipeline implementations."""

from collections.abc import Mapping, Sequence

from raglab.core.exceptions import UnsupportedFrameworkError
from raglab.core.interfaces import RAGPipeline
from raglab.core.schemas import (
    FrameworkName,
    PipelineCapabilities,
    PipelineConfig,
    PipelineSummary,
)


class PipelineRegistry:
    """Resolve framework names and expose comparable implementation metadata."""

    def __init__(self, pipelines: Mapping[FrameworkName, RAGPipeline]) -> None:
        self._pipelines = dict(pipelines)

    def get(self, framework: FrameworkName) -> RAGPipeline:
        try:
            return self._pipelines[framework]
        except KeyError as error:
            raise UnsupportedFrameworkError(
                f"the {framework.value} pipeline is not implemented yet"
            ) from error

    def summaries(self) -> Sequence[PipelineSummary]:
        unavailable = PipelineCapabilities(
            ingestion=False,
            dense_retrieval=False,
            sparse_retrieval=False,
            hybrid_retrieval=False,
            reranking=False,
            metadata_filtering=False,
        )
        return tuple(
            PipelineSummary(
                framework=framework,
                available=framework in self._pipelines,
                capabilities=(
                    self._pipelines[framework].capabilities
                    if framework in self._pipelines
                    else unavailable
                ),
                config=(
                    self._pipelines[framework].config
                    if framework in self._pipelines
                    else PipelineConfig()
                ),
            )
            for framework in FrameworkName
        )
