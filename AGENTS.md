# RAGLab working agreement

- Treat zero paid API usage as a hard project constraint.
- Keep Ollama, local Sentence Transformers, local reranking, PostgreSQL, Qdrant, and Redis as the default runtime path.
- Never add, configure, or invoke a metered cloud API during development, tests, demos, or evaluation.
- OpenAI-compatible support may remain as an optional portfolio adapter, but it must be protected by an explicit opt-in safety setting and tested with mocked HTTP only.
- Do not commit credentials or local `.env` files.
