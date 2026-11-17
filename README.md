# Document Copilot

**An agentic RAG research assistant that answers questions about SEC 10-K filings and cites the exact passage behind every claim.**

Built end to end: React SPA → FastAPI → LangGraph agent → hybrid pgvector + full-text retrieval on Supabase, with a grounding validator that blocks any answer it cannot trace to the source filing.

![Document Copilot demo](docs/media/demo.gif)

<sub>17-second UI walkthrough ([MP4](docs/media/demo.mp4)). The recording uses the real frontend against a local sample-data API, so the answer shown is illustrative rather than live model output.</sub>

---

## At a glance (for recruiters and hiring managers)

| | |
| --- | --- |
| **What it is** | A production-shaped AI product: authenticated chat, streaming answers, clickable citations, evidence viewer, persisted history. |
| **Problem it solves** | Equity analysts spend about half of every week reading SEC filings before they can do any analysis. Document Copilot returns the answer *and* the passage that proves it. |
| **Hardest part** | Trust. An LLM that invents a revenue figure is worse than useless in finance, so every answer passes a code-level grounding check before the user sees it. |
| **Scale of work** | Full stack across 2 services and 3 data stores; 8 planned build phases; ~400 automated backend tests, all offline. |
| **Skills shown** | LLM engineering (LangChain, LangGraph, LangSmith, tool calling, structured output) · RAG and search (pgvector, Postgres FTS, Reciprocal Rank Fusion) · Python/FastAPI async backends · React 19/TypeScript · PostgreSQL schema design and migrations · performance profiling · test design · AI-assisted engineering workflow |

### What I'd point a reviewer to

1. **Fail-closed grounding** (`backend/app/grounding/validator.py`): every cited quote must appear verbatim in a passage retrieved during that turn, every number in the answer must appear in the cited evidence, and figures must keep their original order. Otherwise the turn fails with a typed error instead of showing a polished guess.
2. **A LangGraph agent with a budget** (`backend/app/assistant/agent.py`): a tool-calling research loop that can only finish through a validated `submit_answer` tool, with hard limits on model requests and tool calls.
3. **Measured performance work**: I found a per-request database connection leak, serial LLM calls, and N+1 queries, fixed them, and wrote benchmarks that prove the fixes ([Performance](#performance-engineering)).

---

## Screenshots

| Grounded answer with sources | Evidence viewer |
| --- | --- |
| ![Answer with citation chips](docs/media/grounded-answer.png) | ![Cited table with neighbouring passages](docs/media/evidence-panel.png) |

![Landing page](docs/media/landing.png)

---

## Purpose and use cases

The project started from a client brief for an independent equity research firm (~40 analysts, each covering ~15 US public companies). Its value is condensation: turning thousands of pages of filings into a thesis a portfolio manager can act on. The bottleneck is source intake, which means opening 10-Ks, finding the MD&A, risk factors and segment tables, and comparing them year over year.

| Use case | Example question | What the system does |
| --- | --- | --- |
| **Metric lookup** | "How did Apple's Services revenue change between fiscal 2023 and 2024?" | Quick RAG: one hybrid search and one model call, answer cited to the net-sales table. |
| **Longitudinal comparison** | "How did NVIDIA describe Data Center demand from fiscal 2021 to 2025?" | Deep RAG: the agent searches several years, reads neighbouring chunks, and synthesises. |
| **Cross-company risk review** | "Which companies changed risk-factor language on export controls?" | Deep RAG across tickers, with every claim tied to a passage. |
| **Verification** | Clicking a citation chip | Opens the exact cited text, highlighted inside its chunk, with the neighbouring passages and a link to the filing on EDGAR. |
| **Out-of-scope request** | "Should I buy Apple stock?" | Refuses: no investment advice, and no answer without evidence. |

**Corpus:** 10-K annual reports for **Apple, Microsoft, NVIDIA, Amazon and Alphabet, fiscal 2021–2025** (25 filings) pulled from SEC EDGAR, converted to Markdown with Docling. Tables are extracted as structured rows, so a cited figure points at the actual table cell rather than at flattened text.

---

## How it works

```mermaid
flowchart LR
    U[Analyst] --> FE[React SPA<br/>AI SDK streaming]
    FE -->|JWT| API[FastAPI]
    API --> T[ChatTurn]
    T --> G{{LangGraph turn graph}}
    G --> R[choose_route<br/>router + prefetch search in parallel]
    R -->|greeting / scope| D[reply_directly]
    R -->|focused fact| Q[quick_rag<br/>1 search, 1 model call]
    R -->|comparison / synthesis| A[deep_rag<br/>agentic research loop]
    Q & A --> V[Grounding validator]
    V --> T
    T -->|SSE: stages, route, text, citations| FE
    A <--> S[(Supabase Postgres<br/>pgvector + full-text)]
    Q <--> S
    G -.traces.-> LS[(LangSmith)]
```

### Agentic RAG, step by step

1. **Route.** Greetings and "what's in the corpus?" questions get canned replies without calling an LLM. Everything else goes to a fast model with structured output (`ModelRouteDecision`): answer directly (product help only), `quick_rag`, or `deep_rag`. A regex gate escalates any "direct" decision that mentions a company, metric, form or period, so factual questions can never skip retrieval.
2. **Prefetch.** While the router is thinking, the first hybrid search for the question runs in parallel. It is discarded for direct answers and reused by both RAG routes.
3. **Retrieve (hybrid).** The question is embedded (`text-embedding-3-small`) for a pgvector HNSW search while Postgres full-text search runs over a generated `tsvector`. The two ranked lists are fused with **Reciprocal Rank Fusion**, and the neighbouring chunks of every hit are fetched in one query.
4. **Answer.**
   - *Quick RAG:* evidence candidates (exact quotes with ids) go to the fast model, which returns text plus the evidence ids it relied on.
   - *Deep RAG:* the research agent starts from the prefetched evidence and may call `search_filings`, `read_chunk` and `read_surrounding_chunks` until it calls `submit_answer`.
5. **Ground.** The validator maps evidence ids back to trusted chunks, checks keywords and every financial number against the quotes, and builds canonical citations (label, page, section, quote). In deep RAG one rejection is returned to the model to fix; a second fails the turn.
6. **Stream and persist.** Stage events ("Searching filings", "Validating citations"...) stream live; answer text streams only after validation. The question, answer, citations, route and timings are saved atomically.

### LangChain and LangGraph concepts used

| Concept | Where | Why |
| --- | --- | --- |
| **`StateGraph` with typed state** | `app/chat/graph.py`, `app/assistant/agent.py` | The turn and the research loop are explicit state machines rather than hand-wired callbacks. |
| **Conditional edges** | `pick_route`, `after_tools` | Route selection and the agent's tool-loop exit are graph edges you can read and test. |
| **Runtime context (`context_schema`)** | `TurnContext`, `ResearchContext` | Models, retriever and validator are injected per run, so tests swap them without patching globals. |
| **Custom stream mode** (`get_stream_writer`) | stage and route events | The UI shows real progress as each node runs. |
| **`add_messages` reducer** | research state | Conversation history for the tool loop. |
| **Structured output** (`with_structured_output`) | router, quick answer | Typed Pydantic decisions instead of parsing free text. |
| **Tool calling** (`bind_tools`, `tool_choice="any"`) | research model | Forces every step to be a tool call, so the run always ends through `submit_answer`. |
| **Runnable retry** (`with_retry`) | router | One retry on schema-invalid output. |
| **Callbacks / tracing** (`LangChainTracer`) | `app/resources.py` | One LangSmith trace per chat turn, every node visible. |

### Backend modules

```text
backend/app/
├── resources.py      # clients built once per process: DB pool, OpenAI, Supabase, models, tracer
├── chat/             # turn.py (persist + run), graph.py (LangGraph), routing, quick_rag, streaming
├── assistant/        # agent.py (research loop), tools.py, models.py, outputs, instructions.md
├── retrieval/        # hybrid queries, RRF fusion, batched neighbour reads
├── grounding/        # evidence candidates + fail-closed validator
├── auth/             # Supabase token verification with a short success cache
└── database/         # engine, SQLAlchemy models, chat store
```

---

## Evaluation and verification

### Automated tests (offline, deterministic)

About **400 backend tests** run in about 30 seconds with no network or database. LLMs are replaced by scripted fakes *at the model seam only*: the real LangGraph graphs, tools and grounding validator run in every test.

- **Grounding contract:** forged evidence ids, numbers that are not in the quote, reversed current/prior-year values, uncited factual sentences, and insufficient-evidence refusals.
- **Agent behaviour:** budgets at their exact limits, one-retry semantics, malformed tool arguments, parallel tool calls, cancellation of sibling tools, reused message ids, and 20 concurrent turns sharing no state.
- **Wire contract:** the full SSE event sequence for every route, typed error codes that never leak exception text, persistence after a client disconnects, and ownership checks (403/404) running before any model call.
- **Performance properties:** asserted with simulated latency (see below).

### LangSmith evals

`backend/evals/` runs the real turn graph against live services and scores each answer:

| Evaluator | Checks |
| --- | --- |
| `route_correct` | Greetings stay instant, product help stays direct, research questions use RAG. |
| `completed` | The turn finished without a typed failure. |
| `grounded` | A RAG answer either cites evidence or explicitly refuses, never neither. |
| `cites_expected_company` | Citations come from the company the question is about. |
| `keyword_coverage` | Expected concepts appear in the answer or its quotes. |
| `refusal_correct` | Unanswerable questions are refused without citations. |
| `faithfulness` (optional, LLM judge) | Every claim is supported by the quoted passages. |
| `latency_ms` | End-to-end turn time. |

The dataset combines routing cases, retrieval fixtures and the client brief's ten pilot questions (including one that must be refused).

```bash
cd backend
uv run python -m evals.run_eval                       # local report
uv run python -m evals.run_eval --judge --langsmith   # LangSmith experiment
```

Set `LANGSMITH_TRACING=true` and `LANGSMITH_API_KEY` to trace every production turn as well. Tracing is off by default because traces contain questions and filing text.

### Independent review by AI test agents

After the LangGraph migration I ran three **Codex** agents in parallel, each in its own git worktree with a narrow brief (graph edge cases, HTTP/SSE contract, performance claims). Their rules were to add tests only and mark any bug they found as a strict `xfail` with a reproduction. They found **8 real bugs**, including auth outages returning HTTP 500, duplicate concurrent token checks, and a failure count that let malformed answers bypass the retry limit. I fixed all eight, and their reproductions now run as regression tests.

---

## Performance engineering

Profiling the request path turned up the following. Each fix has a test or benchmark (`backend/perf/bench_turn.py`, simulated latency):

| Problem found | Fix | Result |
| --- | --- | --- |
| A new SQLAlchemy engine (connection pool), OpenAI client and Supabase client built **per chat request**, and never disposed | `Resources` built once in the FastAPI lifespan | 20 requests → 1 engine, 1 client |
| Supabase Auth called on every API request | 30-second verified-token cache plus single-flight for concurrent requests | 12-request burst → 1 upstream call |
| Router LLM call, then embedding, then search, all serial | Prefetch search runs alongside routing | 200 ms + 200 ms → **202 ms** |
| Deep research spent its first model call asking to search | Agent starts from the prefetched evidence | 2 model calls + 1 search → **1 + 0** |
| One neighbour query per search result (N+1) | One batched query | top_k round trips → 1 |
| HNSW `ef_search` (40) silently capped the 50-candidate search | Raised per query | Full candidate set returned |
| Every streamed token re-rendered and re-parsed the Markdown of the whole thread | Memoised messages | Only the streaming answer re-renders |

---

## How I built it with AI-assisted engineering

I used Claude Code and OpenAI Codex as implementation partners, with the context, plans and verification under my control. The repository is the record of that process.

**1. Context first.** Before writing code I wrote the [client brief](docs/client-brief.md) (who the users are, what they pay for, why trust matters) and the [architecture spec](docs/architecture.md) (service boundaries, request flow, streaming contract, data model, grounding policy). I wrote [`AGENTS.md`](AGENTS.md) plus per-service `backend/AGENTS.md` and `frontend/AGENTS.md` as standing instructions for any coding agent: the locked stack, the dependency policy (justify every runtime dependency in the commit message), a single settings module, async-only request paths, and required test coverage. Every agent session started from the same ground rules.

**2. A phased roadmap.** [`docs/todos.md`](docs/todos.md) breaks the product into vertical slices: foundation → auth → chat shell with a stubbed reply → ingestion → retrieval → agent and grounding → trust UI. Phase 3 deliberately shipped a fake assistant so the full stack (auth, persistence, streaming) was proven before any AI was added.

**3. A written plan per phase.** Each phase got an implementation plan in [`docs/superpowers/plans/`](docs/superpowers/plans/) with its goal, architecture, global constraints, and checkbox tasks, for example the [adaptive routing plan](docs/superpowers/plans/2026-08-01-adaptive-chat-routing.md) and the [agent and grounding plan](docs/superpowers/plans/2026-08-01-phase-6-agent-grounding.md). Agents executed against the plan, and I reviewed the diffs.

**4. Frontend → backend → RAG, in that order.** The frontend API layer (`env.ts` → `supabase.ts` → `http.ts` → `api.ts`) and the FastAPI auth boundary came first, then the chat shell, then Docling ingestion with table-aware chunking, then hybrid retrieval, then the agent and grounding, and finally the trust UI (live stages, citation chips, evidence rail).

**5. Test-first, small commits.** Most features landed as `feat:` commits paired with their tests, followed by targeted `fix:` commits from review, for example *"canonicalize grounded citation metadata"* and *"validate agent retrieval context"*.

**6. Architecture review and refactor.** Once the product worked, I ran a structured architecture review (Matt Pocock's *improve-codebase-architecture* method: find shallow modules, propose deeper ones). That led to the `Resources` seam and the LangGraph rewrite, which replaced 7 modules, 4 protocols and 3 threaded callbacks with one graph.

**7. Independent verification.** Parallel Codex agents in isolated worktrees attacked the new code, as described in [Independent review by AI test agents](#independent-review-by-ai-test-agents). The pattern throughout: AI writes code against explicit context and plans, and tests plus review decide what ships.

---

## Tech stack

| Layer | Technology |
| --- | --- |
| Frontend | React 19, TypeScript, Vite, Tailwind CSS, shadcn/ui, Vercel AI SDK (`useChat`), react-markdown + GFM |
| Backend | Python 3.12, FastAPI, Pydantic v2, SQLAlchemy 2, Alembic, structlog, httpx |
| AI | LangChain (`langchain-openai`), LangGraph, LangSmith, OpenAI chat and embedding models |
| Data | Supabase Postgres, pgvector (HNSW), Postgres full-text search, Supabase Auth, row-level security |
| Ingestion | SEC EDGAR, Docling (HTML → Markdown, hybrid chunking), tiktoken |
| Quality | pytest, ruff, ESLint, TypeScript strict build |
| Hosting | Railway (two services) + hosted Supabase |

---

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
| `OPENAI_CHAT_MODEL` | Deep-research chat model, for example `openai:gpt-5-mini` |
| `OPENAI_FAST_MODEL` | Routing and quick-answer model, for example `openai:gpt-5-mini` |
| `OPENAI_EMBEDDING_MODEL` | Embedding model, default `text-embedding-3-small` |
| `OPENAI_EMBEDDING_DIMENSIONS` | Embedding dimension count, default `1536` |
| `ALLOWED_ORIGINS` | Comma-separated CORS origins |
| `LANGSMITH_TRACING` | `true` to send a LangSmith trace per chat turn (default `false`) |
| `LANGSMITH_API_KEY` | Required when tracing is on |
| `LANGSMITH_PROJECT` | LangSmith project name, default `document-copilot` |

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

SEC EDGAR requires a real contact in the User-Agent. Pass it through
`SEC_USER_AGENT` (it is never committed). From the repository root, download the
configured filings and regenerate normalized Markdown:

```bash
SEC_USER_AGENT="Document Copilot you@example.com" uv run data/download.py
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

The intended production deployment is:

- Railway service for `backend/`
- Railway service for `frontend/`
- Hosted Supabase project for Auth and Postgres
- OpenAI API for embeddings and LLM calls

### 1. Prepare Supabase

1. Create a Supabase project.
2. Enable email auth.
3. Copy the project URL, anon key, and service role key.
4. Copy the direct Postgres connection string from project settings.
5. Run migrations from a trusted environment:

```bash
cd backend
uv run alembic upgrade head
```

### 2. Deploy Backend On Railway

Create a Railway service rooted at `backend/`.

Set these environment variables:

```text
SUPABASE_URL=
SUPABASE_ANON_KEY=
SUPABASE_SERVICE_ROLE_KEY=
DATABASE_URL=
OPENAI_API_KEY=
OPENAI_CHAT_MODEL=openai:gpt-5-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_EMBEDDING_DIMENSIONS=1536
ALLOWED_ORIGINS=https://your-frontend-domain
```

Use a production server command equivalent to:

```bash
uv run uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

### 3. Deploy Frontend On Railway

Create a Railway service rooted at `frontend/`.

Set these environment variables:

```text
VITE_API_BASE_URL=https://your-backend-domain
VITE_SUPABASE_URL=
VITE_SUPABASE_ANON_KEY=
```

Build command:

```bash
pnpm install --frozen-lockfile && pnpm build
```

Serve the generated `dist/` directory with Railway's static hosting support or a small static file server configured by the deployment platform.

### 4. Ingest Documents

After the database schema is live, run ingestion from a secure environment that has backend secrets:

```bash
cd backend
uv run document-copilot-ingest --upload --yes
```

Do not run ingestion from the browser or expose ingestion credentials to the frontend.

## Security Notes

- `.env` files are intentionally ignored. Commit only `.env.example` files.
- Rotate any key that was ever committed locally or pushed to a remote.
- Keep the Supabase service role key and OpenAI API key backend-only.
- Use Supabase RLS policies for user-owned product data.
- Verify Supabase JWTs at the FastAPI boundary before any database, retrieval, or LLM work.
- Use the direct Supabase database URL for migrations, not the transaction pooler.

