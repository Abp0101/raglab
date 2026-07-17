# RAGLab working agreement

- Treat zero paid API usage as a hard project constraint.
- Keep Ollama, local Sentence Transformers, local reranking, PostgreSQL, Qdrant, and Redis as the default runtime path.
- Never add, configure, or invoke a metered cloud API during development, tests, demos, or evaluation.
- OpenAI-compatible support may remain as an optional portfolio adapter, but it must be protected by an explicit opt-in safety setting and tested with mocked HTTP only.
- Do not commit credentials or local `.env` files.
- Keep the web interface local-first: no hosted analytics, remote fonts, metered AI SDKs, or browser-side provider credentials.
- Label preview fixtures as sample data and never present baseline measurements as a framework leaderboard.
