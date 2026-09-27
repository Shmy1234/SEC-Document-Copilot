# Document Copilot

Document Copilot is a production-oriented RAG application for asking grounded questions over a curated SEC filing corpus. It combines a React chat workspace, a FastAPI backend, Supabase Auth/Postgres, OpenAI embeddings, and an ingestion pipeline that converts filings into searchable document chunks.

The product goal is simple: help analysts get answers from source documents with a clear trail back to the filings used to produce each answer.

## What It Does

- Authenticated chat workspace for document research.
- Persistent user-owned chat threads and message history.
- SEC filing ingestion from local Markdown into Supabase Postgres.
- Chunking and OpenAI embedding generation for retrieval-ready document records.
- Supabase `pgvector` and Postgres full-text architecture for hybrid retrieval.
- FastAPI service boundary for auth, persistence, retrieval, grounding, and LLM orchestration.
- React SPA frontend that talks to the backend with a Supabase bearer token.

> Current implementation note: authentication, chat shell, thread/message persistence, streaming stub responses, migrations, and ingestion are implemented. The production RAG answer path is represented in the architecture and ingestion foundation, with retrieval, grounding validation, and final assistant orchestration intended to sit behind the existing FastAPI chat boundary.

## Use Cases

| User | Use case | Example question | System output |
| --- | --- | --- | --- |
| Equity analyst | Compare company performance across filings | "How did Nvidia describe data center demand in 2023 vs 2024?" | Grounded synthesis with filing citations |
| Research associate | Find risk factors quickly | "Which companies mention supply chain risk in recent 10-Ks?" | Ranked source passages and summary |
| Portfolio team | Prepare meeting briefs | "Summarize Amazon's capital expenditure discussion over the last three years." | Concise briefing with source links |
| Compliance reviewer | Verify claims against primary documents | "Where does Apple discuss privacy regulation risk?" | Cited excerpts and document metadata |
| Engineering team | Build and evaluate RAG workflows | "Which chunks were retrieved and why?" | Inspectable retrieval, chunk, and citation records |

```mermaid
quadrantChart
    title Document Copilot Use Case Fit
    x-axis Low repeatability --> High repeatability
    y-axis Low citation need --> High citation need
    "Ad hoc market chat": [0.25, 0.20]
    "Document lookup": [0.45, 0.70]
    "Risk factor review": [0.70, 0.85]
    "Earnings prep": [0.62, 0.76]
    "Portfolio research workflows": [0.86, 0.90]
```

## Architecture

Document Copilot is split into a browser client, a stateless API service, hosted Supabase infrastructure, OpenAI model APIs, and a batch ingestion path.

```mermaid
flowchart LR
    user[Analyst] --> browser[React SPA]

    subgraph frontend[Frontend Service]
        app[Vite build<br/>React + TypeScript]
    end

    subgraph backend[Backend Service]
        api[FastAPI API]
        authz[Auth dependency]
        chat[Chat routes]
        retrieval[Hybrid retrieval]
        grounding[Grounding validator]
        assistant[LLM orchestration]
    end

    subgraph supabase[Supabase]
        auth[Auth]
        db[(Postgres<br/>profiles, chats, documents, chunks)]
        vector[pgvector]
        fts[Full-text search]
    end

    subgraph ingestion[Ingestion Pipeline]
        filings[SEC filings]
        markdown[Markdown corpus]
        chunks[Chunker]
        embed[Embedding batches]
    end

    openai[OpenAI<br/>Embeddings + LLM]

    app --> browser
    browser -->|email session| auth
    browser -->|JWT + chat request| api
    api --> authz
    authz -->|verify token| auth
    chat --> db
    chat --> retrieval
    retrieval --> vector
    retrieval --> fts
    retrieval --> db
    retrieval --> grounding
    grounding --> assistant
    assistant --> openai
    api -->|stream response| browser

    filings --> markdown
    markdown --> chunks
    chunks --> embed
    embed --> openai
    embed --> db
```

### RAG And Software Layers

```mermaid
flowchart TB
    subgraph client[Client Layer]
        ui[React chat UI]
        session[Supabase browser session]
        apiClient[Typed API client]
    end

    subgraph service[Application Layer]
        routes[FastAPI routes]
        deps[Auth dependencies]
        store[ChatStore persistence]
        stream[Streaming response boundary]
    end

    subgraph rag[RAG Layer]
        ingest[Markdown ingestion]
        chunk[Document chunking]
        embeddings[Embedding generation]
        hybrid[Vector + keyword retrieval]
        rrf[Reciprocal Rank Fusion]
        cite[Citation validation]
    end

    subgraph data[Data Layer]
        postgres[(Supabase Postgres)]
        tables[Profiles, threads, messages,<br/>source_documents, document_chunks]
        indexes[pgvector HNSW + GIN full-text indexes]
    end

    subgraph model[Model Layer]
        openaiEmbed[OpenAI embedding model]
        openaiChat[OpenAI chat model]
    end

    ui --> session --> apiClient --> routes
    routes --> deps --> store --> postgres
    routes --> stream
    stream --> hybrid
    ingest --> chunk --> embeddings --> openaiEmbed
    embeddings --> tables
    hybrid --> indexes
    hybrid --> rrf --> cite --> openaiChat
    tables --> postgres
    indexes --> postgres
```

## Tech Stack

| Area | Technology |
| --- | --- |
| Frontend | Vite, React, TypeScript, React Router, Tailwind CSS, shadcn/ui |
| Auth | Supabase Auth |
| Backend | Python 3.12, FastAPI, Uvicorn, Pydantic v2 |
| Database | Supabase Postgres, SQLAlchemy models, Alembic migrations |
| Retrieval | `pgvector`, Postgres full-text search, Reciprocal Rank Fusion |
| AI | OpenAI SDK, OpenAI embeddings, PydanticAI target orchestration |
| Package management | `uv` for backend, `pnpm` for frontend |
| Hosting target (planned) | AWS Lightsail + Docker Compose/Caddy + Cloudflare DNS; hosted Supabase |

## Repository Layout

```text
.
|-- backend/              # FastAPI API, migrations, ingestion, tests
|-- data/                 # Local corpus tooling and gitignored downloaded payloads
|-- docs/                 # Architecture notes, setup guides, implementation plans
|-- frontend/             # Vite React SPA
|-- AGENTS.md             # Engineering rules for coding agents
`-- README.md
```

## Implementation Details

### Frontend

The frontend is a plain React SPA. It handles UI state, routing, Supabase browser auth, and authenticated requests to FastAPI.

Key modules:

- `frontend/src/lib/env.ts` validates browser-safe environment variables at startup.
- `frontend/src/lib/supabase.ts` creates the Supabase browser client.
- `frontend/src/lib/http.ts` wraps `fetch`, applies the API base URL, and attaches bearer tokens.
- `frontend/src/lib/api.ts` exposes application API calls.
- `frontend/src/pages/ChatPage.tsx` and `frontend/src/components/chat/*` render the chat workspace.

The frontend must never contain the Supabase service role key, database URL, or OpenAI API key.

### Backend

The backend is the trusted application boundary. It verifies Supabase JWTs, owns privileged credentials, persists chat records, runs ingestion, and is the intended home for retrieval, grounding, and LLM orchestration.

Key modules:

- `backend/app/main.py` creates the FastAPI app and CORS policy.
- `backend/app/config.py` validates required server environment variables.
- `backend/app/auth/dependencies.py` verifies authenticated users.
- `backend/app/api/chat.py` exposes chat thread, message, and streaming endpoints.
- `backend/app/database/chats.py` persists chat threads and messages.
- `backend/app/database/models/*` defines SQLAlchemy models used by Alembic.
- `backend/ingest/*` prepares Markdown filings, chunks text, generates embeddings, and writes source documents/chunks.

### Ingestion

The ingestion pipeline is deliberately two-stage. From the repository root, download SEC HTML and convert it into persistent normalized Markdown:

```bash
uv run data/download.py
uv run --directory backend python ../data/html_to_markdown/convert_html_to_markdown.py --overwrite
```

The backend ingestion command then reads `data/Markdown/manifest.json`. A default or explicit dry run makes no OpenAI or database writes:

```bash
cd backend
uv run document-copilot-ingest --dry-run --limit-documents 2
```

Make the first paid/database test one document and one chunk:

```bash
cd backend
uv run document-copilot-ingest --upload --limit-documents 1 --limit-chunks 1
```

For a full upload, pass `--yes` deliberately:

```bash
cd backend
uv run document-copilot-ingest --upload --yes
```

### Database

Alembic owns schema changes. Migrations live in `backend/alembic/versions`.

The database model includes:

- User profiles tied to Supabase Auth users.
- Chat threads and chat messages.
- Source documents with filing metadata.
- Document chunks with text, metadata, embedding vectors, and search fields.

Supabase/Postgres-specific features such as `vector`, HNSW indexes, GIN indexes, generated search vectors, and RLS policies belong in explicit migration operations.

## Running locally

These commands assume a fresh checkout and can be copied as written. Secrets stay
in the ignored `.env` files; never put the service-role key, database URL, or
OpenAI key in `frontend/.env`.

### Prerequisites

- Python 3.12+
- `uv`
- Node.js and `pnpm`
- Supabase project
- OpenAI API key

### 1. Configure the backend

```bash
cd backend
cp .env.example .env
```

Edit `backend/.env` and set every placeholder below:

```text
SUPABASE_URL=https://your-project-ref.supabase.co
SUPABASE_ANON_KEY=your-anon-public-key
SUPABASE_SERVICE_ROLE_KEY=your-service-role-secret-key
DATABASE_URL=postgresql://postgres:your-password@db.your-project-ref.supabase.co:5432/postgres
OPENAI_API_KEY=sk-your-openai-api-key
OPENAI_CHAT_MODEL=openai:gpt-5-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_EMBEDDING_DIMENSIONS=1536
ALLOWED_ORIGINS=http://localhost:5173
```

Install, migrate, and start the API:

```bash
cd backend
uv sync
uv run alembic upgrade head
uv run uvicorn app.main:app --reload
```

Verify it from another terminal:

```bash
curl http://localhost:8000/health
```

Backend environment variables:

| Variable | Purpose |
| --- | --- |
| `SUPABASE_URL` | Supabase project URL |
| `SUPABASE_ANON_KEY` | Public anon key for user-scoped requests |
| `SUPABASE_SERVICE_ROLE_KEY` | Backend-only privileged Supabase key |
| `DATABASE_URL` | Direct Supabase Postgres connection string for Alembic/app DB access |
| `OPENAI_API_KEY` | OpenAI API key for embeddings and generation |
| `OPENAI_CHAT_MODEL` | PydanticAI chat model identifier, for example `openai:gpt-5-mini` |
| `OPENAI_EMBEDDING_MODEL` | Embedding model, default `text-embedding-3-small` |
| `OPENAI_EMBEDDING_DIMENSIONS` | Embedding dimension count, default `1536` |
| `ALLOWED_ORIGINS` | Comma-separated CORS origins |

### 2. Configure the frontend

```bash
cd frontend
cp .env.example .env
```

Edit `frontend/.env` with browser-safe values only:

```text
VITE_API_BASE_URL=http://localhost:8000
VITE_SUPABASE_URL=https://your-project-ref.supabase.co
VITE_SUPABASE_ANON_KEY=your-anon-public-key
```

Install and start the SPA:

```bash
cd frontend
pnpm install
pnpm dev
```

Open `http://localhost:5173`, create an email account, and sign in. If Supabase
email confirmation is enabled, confirm the address before signing in.

Frontend environment variables:

| Variable | Purpose |
| --- | --- |
| `VITE_API_BASE_URL` | FastAPI backend URL |
| `VITE_SUPABASE_URL` | Supabase project URL |
| `VITE_SUPABASE_ANON_KEY` | Browser-safe Supabase anon key |

### 3. Load or update the filing corpus

Set `USER_AGENT` in `data/download.py` to a real contact address before making
SEC requests. From the repository root, download the configured filings and
regenerate normalized Markdown:

```bash
uv run data/download.py
uv run --directory backend python ../data/html_to_markdown/convert_html_to_markdown.py --overwrite
uv run --directory backend document-copilot-ingest --dry-run
```

Review the dry-run document/chunk counts and missing metadata. New accessions are
safe to upload idempotently; already-ingested accessions are skipped:

```bash
uv run --directory backend document-copilot-ingest --upload --yes
```

If conversion or chunking changed for an accession already in the database,
replace its document and chunks atomically:

```bash
uv run --directory backend document-copilot-ingest --upload --replace --yes
```

`--replace` regenerates embeddings and deletes/recreates matching accession rows,
so run the dry-run first. Use `--limit-documents 1 --limit-chunks 1` for the first
paid/database write in a new environment.

### 4. Run pilot checks

Run local checks with no database or network access:

```bash
cd backend
uv run pytest -m "not integration"
uv run ruff check .

cd ../frontend
pnpm lint
pnpm build
```

With the configured Supabase corpus and OpenAI account available, verify the live
schema/retrieval path and then run all ten client-brief questions. The JSON report
records model, total latency, time to the first live stage, answer, and citations
for every question:

```bash
cd backend
RUN_RETRIEVAL_INTEGRATION=1 uv run pytest tests/integration/test_live_system.py
uv run python smoke_assistance.py --output pilot-smoke-report.json
```

Review every answer in `pilot-smoke-report.json`; a smoke run is accepted only if
each factual answer has usable citations and question 10 refuses to claim that
generative AI caused margin improvement beyond what the filings establish.

## Quality Checks

Backend:

```bash
cd backend
uv run pytest
uv run ruff check .
```

Frontend:

```bash
cd frontend
pnpm lint
pnpm build
```

## Deployment

The planned deployment is one AWS Lightsail Linux instance running Docker Compose, with Caddy serving the Vite build and proxying FastAPI, and Cloudflare Free providing DNS. Supabase and OpenAI remain external services. This replaces the earlier Railway and ECS proposals because the hosting budget is $10/month, excluding Supabase/OpenAI; currency and taxes still need confirmation.

The backend Docker recipe is available with [local build/run instructions](docs/guides/backend-docker.md); its runtime verification is pending. A [GitHub Actions CI workflow](docs/guides/ci.md) has been added; its remote checks are pending. AWS infrastructure and CD automation are not implemented yet. Follow the existing [Phase 9 checklist](docs/todos.md#phase-9--devops--deployment-aws--cloudflare), [architecture and budget decisions](docs/architecture.md#deployment-shape), and [current progress](docs/progress-tracker.md). The $7 USD/month 1 GB Lightsail candidate must pass memory and workload checks before acceptance; this plan does not promise high availability or zero-downtime releases.

## Security Notes

- `.env` files are intentionally ignored. Commit only `.env.example` files.
- Rotate any key that was ever committed locally or pushed to a remote.
- Keep the Supabase service role key and OpenAI API key backend-only.
- Use Supabase RLS policies for user-owned product data.
- Verify Supabase JWTs at the FastAPI boundary before any database, retrieval, or LLM work.
- Use the direct Supabase database URL for migrations, not the transaction pooler.

## Roadmap

- Implement production hybrid retrieval queries over `document_chunks`.
- Add Reciprocal Rank Fusion and source passage expansion.
- Replace the streaming stub with grounded assistant orchestration.
- Add citation validation before assistant messages are persisted.
- Add retrieval and grounding evaluation fixtures.
- Implement the budgeted Lightsail, Cloudflare, Docker, and CI/CD milestones in `docs/todos.md`, Phase 9.
