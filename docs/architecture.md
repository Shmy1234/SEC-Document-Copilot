# Document Copilot Architecture

## Purpose

Document Copilot is an internal research assistant for analysts who need grounded answers from a curated SEC filing corpus. The architecture must optimize for trust: every answer is generated from retrieved source passages, every factual claim is citable, and the system fails clearly when the corpus does not support an answer.

This document describes the target architecture for the chat experience, LLM orchestration, and the communication layer between the React SPA, Supabase, and FastAPI backend.

Hosting decision, 2026-09-27: the proposed Lightsail + Cloudflare target below replaces Railway and the earlier ECS proposal because the user specified a $10/month AWS + Cloudflare ceiling. Infrastructure and deployment have not been implemented or verified in this planning pass. The single implementation checklist remains [todos.md](todos.md#phase-9--devops--deployment-aws--cloudflare); [progress-tracker.md](progress-tracker.md) records current evidence and open decisions.

## High-Level Architecture

The best opening diagram is a service-level view that shows the two core paths: the live chat path that serves users, and the ingestion path that prepares SEC filings for retrieval.

```mermaid
flowchart LR
    user[Analyst] --> browser[Browser<br/>React chat app]

    subgraph aws[AWS hosting target - planned]
        frontend[Caddy static hosting<br/>Vite build]
        backend[Docker Compose on Lightsail<br/>FastAPI + PydanticAI]
    end

    subgraph supabase[Supabase]
        auth[Auth<br/>email session]
        db[(Postgres<br/>chats, documents, chunks<br/>pgvector + full-text)]
    end

    openai[OpenAI<br/>LLM + embeddings]
    corpus[SEC filing corpus]
    ingestion[Ingestion pipeline<br/>download, parse, chunk, embed]

    frontend -->|serves app| browser
    browser -->|sign in| auth
    auth -->|JWT session| browser
    browser -->|chat request + JWT| backend
    backend -->|verify user| auth
    backend -->|retrieve passages<br/>persist chats + citations| db
    backend -->|generate grounded answer| openai
    backend -->|stream answer + citations| browser

    corpus --> ingestion
    ingestion -->|create embeddings| openai
    ingestion -->|store documents + chunks| db
```

## Architectural Goals

- Keep the browser thin: it renders chat state, manages the user's Supabase session, and streams assistant responses.
- Keep the backend authoritative: retrieval, grounding, citation checks, tool execution, and database writes happen in FastAPI.
- Use Supabase for identity and durable product state: users, chat threads, source documents, chunks, embeddings, and citation metadata.
- Use Supabase `pgvector` for semantic retrieval and Postgres full-text search for keyword retrieval.
- Make the LLM path typed and testable by using PydanticAI agents with explicit dependencies, outputs, and tool boundaries.
- Serve the static frontend and proxy FastAPI through Caddy on one Docker Compose Lightsail host, retaining Supabase and OpenAI. Cloudflare owns DNS initially. Accept a single point of failure within the pilot budget.

## Stack

Frontend:

- Vite + React SPA + TypeScript
- React Router for routing
- Tailwind CSS and shadcn/ui for UI
- `@supabase/supabase-js` for browser auth
- Vercel AI SDK UI packages for chat state and streaming client behavior

Backend:

- Python 3.12+
- FastAPI + Uvicorn
- Pydantic v2 + pydantic-settings
- PydanticAI for typed LLM orchestration
- OpenAI SDK for generation and embeddings
- Supabase Python client for server-side database access
- SQLAlchemy models + Alembic migrations for schema management
- Supabase `pgvector` for semantic search
- Postgres full-text search for lexical retrieval
- `httpx` for outbound HTTP
- `structlog` for structured logs

Persistence:

- Supabase Auth for email login
- Supabase Postgres for user records, chat threads, chat messages, source documents, chunks, embeddings, full-text search vectors, and citation metadata

## System Boundaries

The frontend is responsible for user interaction, local UI state, and sending the authenticated user's request to the backend. It should never hold service-role credentials, run retrieval logic, call OpenAI directly, or write privileged records to Supabase.

The backend is responsible for request authorization, retrieval, prompt construction, LLM execution, citation validation, streaming responses, and durable persistence. It owns all privileged credentials and is the only service allowed to use the Supabase service-role key.

Supabase is responsible for authentication and durable product state. Browser access uses the anon key and user JWT. Server access uses either the user's bearer token for user-scoped operations or the service-role key for privileged writes that must still be explicitly tied to the authenticated user.

## Request Flow

1. The user signs in with Supabase email auth in the React SPA.
2. The frontend stores the Supabase session through `@supabase/supabase-js`.
3. When the user opens a chat, the frontend loads the thread and prior messages through FastAPI, which reads user-scoped records from Supabase.
4. The chat UI uses the Vercel AI SDK React primitives to manage message state and submit new user messages to the FastAPI chat endpoint.
5. The frontend sends the Supabase access token as `Authorization: Bearer <token>`.
6. FastAPI verifies the token with Supabase Auth before doing any retrieval or LLM work.
7. FastAPI creates a request-scoped context containing the authenticated user, chat thread, Supabase client, retrieval service, citation policy, and LLM settings.
8. A PydanticAI agent retrieves relevant document chunks, generates a grounded answer, and returns typed output containing answer text and citations.
9. FastAPI streams assistant message parts back to the browser in the format expected by the AI SDK client.
10. FastAPI persists the final user message, assistant message, cited chunks, and usage metadata to Supabase.

## Frontend Chat Layer

The frontend remains a plain Vite SPA. It should not adopt Next.js route handlers or server components. The AI SDK is used only for its React chat primitives and streaming client behavior.

The chat module should be organized around these responsibilities:

- `src/lib/env.ts` validates `VITE_API_BASE_URL`, `VITE_SUPABASE_URL`, and `VITE_SUPABASE_ANON_KEY`.
- `src/lib/supabase.ts` creates the browser Supabase client.
- `src/lib/http.ts` wraps `fetch`, applies the backend base URL, injects the Supabase bearer token, handles timeouts, and converts failures into typed API errors.
- `src/lib/api.ts` exposes product-level calls such as loading threads, creating threads, and fetching message history.
- `src/pages/chat/*` renders chat routes and delegates chat streaming to a focused chat component.
- `src/components/chat/*` renders messages, citations, source passages, empty states, and streaming status.

The chat component should initialize with stored messages and then let the AI SDK manage in-flight UI state. The transport points to FastAPI, not to a frontend server route.

Conceptual shape:

```ts
const { messages, sendMessage, status, error } = useChat({
  id: threadId,
  messages: initialMessages,
  transport: new DefaultChatTransport({
    api: `${apiBaseUrl}/chat/stream`,
    headers: async () => ({
      Authorization: `Bearer ${await getAccessToken()}`,
    }),
  }),
});
```

The exact API surface should be verified during implementation against the installed AI SDK version. The architectural rule is stable: the browser streams to FastAPI with the user's Supabase token, and FastAPI owns the assistant run.

## Backend LLM Layer

PydanticAI should be introduced as the backend's orchestration layer for answer generation. It replaces ad hoc prompt calls with a typed agent boundary.

Recommended backend modules:

```text
backend/app/
├── api/
│   └── chat.py                 # FastAPI routes for chat threads and streaming
├── auth/
│   └── dependencies.py         # Supabase JWT verification and current user dependency
├── chat/
│   ├── orchestrator.py         # Coordinates one chat turn end-to-end
│   ├── messages.py             # Converts AI SDK messages to and from internal message types
│   └── streaming.py            # Emits AI SDK-compatible streaming events
├── assistant/
│   ├── agent.py                # PydanticAI agent definition
│   ├── deps.py                 # Runtime dependency dataclass for the agent
│   ├── outputs.py              # GroundedAnswer, Citation, and SourcePassage
│   └── instructions.md         # System instructions and product contract
├── retrieval/
│   ├── queries.py              # pgvector and full-text SQL queries
│   ├── fusion.py               # Reciprocal Rank Fusion for hybrid search
│   └── retriever.py            # Query-to-source-passage retrieval logic
├── grounding/
│   └── validator.py            # Ensures citations map to retrieved passages
└── database/
    ├── supabase.py             # Supabase client construction
    ├── models.py               # SQLAlchemy table models used by Alembic autogenerate
    ├── chats.py                # Chat, thread, message, and citation persistence
    └── documents.py            # Source document, chunk, embedding, and search queries
```

These names should follow the product workflow rather than a generic service layer. `chat/orchestrator.py` owns the full turn lifecycle, `assistant/agent.py` owns the LLM boundary, `retrieval/` owns hybrid source-passage search, and `grounding/` owns the trust contract that answers must cite retrieved evidence.

The agent should receive explicit dependencies rather than reaching into globals:

```python
@dataclass
class DocumentAgentDeps:
    user_id: str
    thread_id: str
    retriever: DocumentRetriever
    grounding_validator: GroundingValidator


class GroundedAnswer(BaseModel):
    answer: str
    citations: list[Citation]
    cited_passages: list[SourcePassage]
```

The agent's instructions should encode the product contract:

- Answer only from retrieved passages.
- Cite every factual claim.
- If the retrieved context is insufficient, say that the corpus does not contain enough evidence.
- Do not provide stock recommendations or investment advice.
- Keep answers concise enough for analyst review, but include enough cited passages to verify the answer.

Retrieval and grounding remain independent from PydanticAI. This keeps ingestion, retrieval tests, and citation validation testable without invoking the LLM.

## Retrieval Strategy

Document Copilot uses hybrid retrieval:

1. Embed the user's query with the configured OpenAI embedding model.
2. Run a semantic search over `document_chunks.embedding` with `pgvector`.
3. Run a lexical search over `document_chunks.search_vector` with Postgres full-text search.
4. Fuse the two ranked lists in Python with Reciprocal Rank Fusion.
5. Fetch the selected chunks, source document metadata, and optional neighboring chunks for grounding.

This keeps the database responsible for efficient ranked retrieval and keeps the application responsible for product-specific ranking policy. The first implementation should avoid agent-generated SQL; the PydanticAI agent receives bounded tools such as `search_filings`, `read_chunk`, and `read_surrounding_chunks`.

## Supabase and FastAPI Communication

Supabase Auth is the identity source. FastAPI must treat the browser's Supabase JWT as the request credential.

Frontend rules:

- Use the anon key only in the browser.
- Read the current session through the shared Supabase client.
- Send the access token to FastAPI through the shared API client.
- Never pass tokens through component props.
- Never expose the service-role key to the frontend.

Backend rules:

- Verify `Authorization: Bearer <token>` at the FastAPI boundary.
- Reject unauthenticated requests before retrieval or LLM work.
- Derive `user_id` and email from the verified Supabase user.
- Use user-scoped database operations wherever possible.
- Use the service-role key only on the backend for privileged writes that cannot be safely performed with the anon key.
- Always attach persisted chat records to the authenticated `user_id`.

The backend can verify the JWT by calling Supabase Auth's user endpoint or by validating the project's JWT signing keys. For the first implementation, calling Supabase Auth is simpler and avoids local JWT validation mistakes. If request volume grows, local JWT verification can be added behind the same `AuthService` interface.

Recommended backend units:

- `app/auth/dependencies.py` validates bearer tokens and exposes `get_current_user`.
- `app/database/supabase.py` creates user-scoped and admin Supabase clients.
- `app/database/chats.py` stores and reads chat threads, messages, and citation records.
- `app/database/documents.py` stores and reads source documents, chunks, embeddings, and full-text search data.

## Streaming Contract

The frontend should receive incremental assistant output, not wait for a full answer. FastAPI should expose a streaming endpoint that emits AI SDK-compatible message parts.

Recommended endpoint:

```text
POST /chat/stream
Authorization: Bearer <supabase_access_token>
Content-Type: application/json
```

Request body:

```json
{
  "threadId": "uuid",
  "messages": []
}
```

The `messages` payload should use the AI SDK UI message format at the frontend boundary. FastAPI can translate that wire format into internal Pydantic models before invoking the agent.

Streaming responsibilities:

- Send text deltas as the answer is generated.
- Send citation/source metadata as structured parts once available.
- Send clear error events for authentication failures, missing threads, retrieval failures, and grounding failures.
- Persist only after the assistant run completes successfully, unless a separate partial-message model is deliberately introduced later.

## Data Model

Supabase tables should be small and product-oriented:

- `profiles`: one row per authenticated user, keyed by Supabase `auth.users.id`.
- `chat_threads`: thread metadata, owner, title, timestamps.
- `chat_messages`: user and assistant messages in order, with AI SDK-compatible message JSON where useful.
- `message_citations`: normalized citation records linked to assistant messages.
- `source_documents`: original document records with filing metadata, source URL, and normalized Markdown content.
- `document_chunks`: chunk text, chunk metadata, embeddings, and generated full-text search vectors.

`source_documents` stores the normalized Markdown version of each filing so the application can re-chunk, inspect, and cite the original extracted text without reaching back into downloaded HTML files. `document_chunks` stores retrieval-ready passages:

- chunk ID
- document ID
- chunk index
- page or section metadata
- chunk text
- embedding vector
- generated `tsvector` for full-text search
- token count
- metadata JSON for ticker, company, filing type, filing date, year, accession number, page, section, and source offsets

Hybrid retrieval runs two bounded queries against `document_chunks`: a semantic `pgvector` query and a Postgres full-text query. The backend fuses those ranked lists with Reciprocal Rank Fusion, then fetches the selected chunks and neighboring context for grounding.

## Schema Management

Database schema changes are managed from the backend with SQLAlchemy models and Alembic migrations. Supabase is the hosted Postgres database, but the Supabase dashboard is not the source of truth for table definitions.

The workflow is:

1. Update SQLAlchemy models in `app/database/models.py`.
2. Generate a candidate migration with `uv run alembic revision --autogenerate -m "<change>"`.
3. Review the generated migration file in `backend/alembic/versions/`.
4. Add explicit migration operations for Postgres/Supabase features that autogenerate cannot infer reliably.
5. Apply the migration locally or against the linked Supabase database with `uv run alembic upgrade head`.
6. Commit both the model changes and the migration file.

Normal tables and ordinary indexes should be represented in SQLAlchemy models where practical. The following should be written explicitly in migrations with `op.execute()` or carefully reviewed Alembic operations:

- `create extension if not exists vector`
- `vector(1536)` embedding columns if the SQLAlchemy type renderer is not sufficient
- generated `tsvector` columns
- HNSW indexes for vector search
- GIN indexes for full-text search and JSON metadata
- RLS enablement and policies
- grants or Supabase role-specific permissions

Alembic must connect with Supabase's direct/session database connection string. Do not run migrations through the transaction pooler URL, because schema migrations, extension setup, and index creation require session-level database behavior.

## Grounding and Citation Policy

Grounding is part of the architecture, not a prompt preference.

The backend should enforce these invariants:

- Every assistant answer has at least one citation unless the answer explicitly says there is not enough evidence.
- Every citation maps to a retrieved source passage.
- Cited passages include enough metadata for the frontend to show company, filing, date, page or section, and excerpt.
- The model cannot cite documents that were not retrieved for the current request.
- If citation validation fails, the backend returns a controlled failure instead of a polished unsupported answer.

This policy should be covered by backend unit tests around retrieval, citation extraction, and grounding enforcement.

## Error Handling

Expected error classes:

- `401 Unauthorized`: missing, expired, or invalid Supabase token.
- `403 Forbidden`: authenticated user tries to access another user's thread.
- `404 Not Found`: thread or source document does not exist.
- `422 Unprocessable Entity`: invalid request payload.
- `502 Bad Gateway`: upstream LLM or Supabase failure.
- `500 Internal Server Error`: unexpected backend failure.

The frontend should render friendly messages while preserving enough technical detail in logs for debugging. Network and CORS failures should be distinguishable from HTTP failures in the shared API client.

## Configuration

Each service must keep one settings module as the source of truth.

Frontend settings:

- `VITE_API_BASE_URL`
- `VITE_SUPABASE_URL`
- `VITE_SUPABASE_ANON_KEY`

Backend settings:

- `SUPABASE_URL`
- `SUPABASE_ANON_KEY`
- `SUPABASE_SERVICE_ROLE_KEY`
- `DATABASE_URL` for Alembic and direct Postgres access
- `OPENAI_API_KEY`
- `ALLOWED_ORIGINS`
- embedding model name and dimensions

Do not read environment variables directly from components, route handlers, or services. Frontend code should use `src/lib/env.ts`. Backend code should use `app/config.py`.

## Deployment Shape

Status: proposed, not provisioned or verified. The backend Docker recipe has been added, with image/runtime verification pending. The GitHub Actions CI workflow is written, with remote verification pending. Infrastructure templates and deployment automation are not implemented. No AWS, Cloudflare, or GitHub account state was inspected.

### Budget and scope

The user specified **$10/month for AWS + Cloudflare**, excluding Supabase and OpenAI. Currency is still unconfirmed: all vendor amounts below are **USD**, before tax. Confirm currency, tax, region, and whether an existing domain is available before spending. Domain registration/renewal is additional unless included explicitly in the final budget. Do not rely on temporary credits or trials.

The candidate is Lightsail's **$7/month Linux public-IPv4 bundle with 1 GB RAM**, leaving at most $3 before tax for extras. It is a sizing experiment, not proof that this app fits. Build images on CI, never on this small host. Measure cold startup, memory, disk, and one real chat before accepting the instance size. If it runs out of memory, reduce scope or revisit the budget; do not silently resize. [AWS bundle table](https://docs.aws.amazon.com/lightsail/latest/userguide/amazon-lightsail-bundles.html)

Cloudflare DNS is available free. Keep a static Lightsail IP attached; audit unused resources. Snapshot storage, excess transfer, additional instances, logs, CI artifact overages, taxes, and domains can break the ceiling. Estimate and review these explicitly. Budget alerts do not enforce a hard cap. [Cloudflare DNS](https://developers.cloudflare.com/dns/faq/), [Lightsail pricing](https://aws.amazon.com/lightsail/pricing/)

### Service boundaries

| Responsibility | Proposed owner |
| --- | --- |
| Hosting | One Lightsail Linux instance, 1 GB candidate, region chosen for users and Supabase connectivity |
| Frontend and HTTPS | Caddy container serving the compiled Vite assets and automatically obtaining/renewing public certificates |
| API | FastAPI container on a private Compose network; one worker initially |
| DNS | Cloudflare Free; DNS-only A records for app/api pointing to the attached static IP |
| Infrastructure | CloudFormation for Lightsail instance, static IP, and declared networking; reviewed bootstrap script for OS/Docker |
| CI/CD | GitHub Actions, immutable release artifacts, protected production deployment |
| Secrets | Restricted server-side configuration file provisioned separately over authenticated SSH; no secrets in bootstrap user data |
| Durable data/auth | Existing hosted Supabase, with backup coverage verified separately |
| AI | Existing OpenAI integration, with explicit workload/spend controls |
| Monitoring | Lightsail metrics, rotated container logs, bounded health monitoring and billing alerts |

Caddy is a standard web server used for TLS renewal, static file serving, and reverse proxying; it is infrastructure software, not a new application framework. Store its certificate data persistently and patch its pinned image. Expose only 80/443 publicly; HTTP redirects to HTTPS. Do not publish backend port 8000, the Docker socket, or the Caddy administration endpoint. [Caddy automatic HTTPS](https://caddyserver.com/docs/automatic-https)

DNS is a lookup step, not an HTTP hop when records are DNS-only. Requests follow browser → Lightsail/Caddy → static files or FastAPI. Browser authentication also contacts Supabase. The backend contacts Supabase and OpenAI. Cloudflare DNS-only provides no Cloudflare HTTP WAF, caching, or HTTP DDoS filtering. [Proxy status](https://developers.cloudflare.com/dns/proxy-status/)

Use `app.example.com` for the SPA and `api.example.com` for FastAPI. Preserve existing email records during DNS migration and verify DNSSEC delegation when changing providers. Supply Vite variables at build time; update backend CORS and Supabase site/redirect URLs together. Preserve SPA navigation fallback without turning missing assets into HTML responses. Hashed assets may be cached; `index.html` must revalidate.

### Release design

Build the backend image and frontend assets from the same checked commit on GitHub-hosted runners. Keep current and previous release identifiers, checksums, and artifacts within verified GitHub storage/minute allowances. Initially transfer a compressed Docker image archive and frontend archive over SSH; a paid registry is unnecessary. Do not make private images public to obtain free storage.

CloudFormation owns host infrastructure; the application workflow owns versioned application files and Compose releases. Keep host replacement out of normal application deployment. Review replacement/deletion in infrastructure change sets and preserve a recovery path for secrets, certificates, and host configuration. [CloudFormation Lightsail resources](https://docs.aws.amazon.com/AWSCloudFormation/latest/TemplateReference/AWS_Lightsail.html)

Use GitHub OIDC for short-lived AWS deployment credentials. A bounded implementation experiment should use Lightsail's temporary SSH access API, verify the server host key against a trusted bootstrap record, and allow the runner's SSH source IP only for the deployment window. Treat that temporary firewall rule as an explicit exception to the declared baseline; revoke it even after failure and verify cleanup. Never log the temporary private key. If this path cannot be constrained and verified, keep releases operator-driven until the access design is resolved. [GitHub OIDC](https://docs.github.com/en/actions/how-tos/secure-your-work/security-harden-deployments/oidc-in-aws), [Lightsail temporary access](https://docs.aws.amazon.com/cli/latest/reference/lightsail/get-instance-access-details.html)

Deployment access can replace executable application code and may imply host administration. Restrict it to trusted production workflows; pull requests must never obtain it. Do not assume a normal Lightsail host has an EC2 instance profile or put permanent AWS keys on the server to simplify deployment.

Deployments have a maintenance window at this budget. Bound and drain current turns, stop the old API to free memory, run an additive Alembic migration once from the release image, then start the new API and switch compatible frontend assets. Verify health and a bounded smoke check. On failure, restore the previous application artifacts only if the database schema remains compatible. An application rollback does not undo schema/data changes. Keep destructive migrations separate.

Run migrations over a verified direct/session Supabase endpoint, never the transaction pooler. Test DNS/IP-family reachability from the host. Ingestion remains a separate bounded operation on a capable trusted machine; do not run corpus conversion/embedding builds alongside the API on a 1 GB host.

### Limits and later decisions

Existing `/health` reports liveness, not successful retrieval or model access. Chat tasks are process-local even after browser disconnect, so restarts can interrupt them. Verify stream delivery, quiet intervals, bounded turn duration, client retry behavior, and graceful shutdown. Avoid promising zero downtime, durable background execution, or capacity for 40 concurrent users.

Security work must verify API ownership checks because service-role calls can bypass database RLS. Apply per-user quotas, concurrent-turn limits, bounded inputs, timeouts, model/tool budgets, and deletion/retention rules before expanding beyond invited test users. Cloudflare filtering cannot replace application authorization or spending limits.

Optional later step: proxy selected HTTP hostnames through Cloudflare Free after verifying feature availability. Use Full (strict) TLS, bypass all authenticated API caching, test streaming and authentication, and prevent direct-origin bypass with suitable network/origin controls. Account for certificate renewal when restricting ingress. Cloudflare would then terminate TLS and process API traffic; include it in the data-flow review. [TLS guidance](https://developers.cloudflare.com/ssl/origin-configuration/ssl-modes/full-strict/), [timeout guidance](https://developers.cloudflare.com/support/troubleshooting/http-status-codes/cloudflare-5xx-errors/error-524/)

Deferred until budget or measured requirements change: ECS/Fargate, ALB, NAT gateways, S3/CloudFront frontend hosting, ECR, paid WAF, managed secrets/logging additions, a permanent staging server, and multi-zone availability. Do not add RDS, Cognito, another vector store, Kubernetes, Redis, or queues speculatively. The AWS application stack stays Python/React/Supabase/OpenAI.

## Implementation Sequence

1. Scaffold the frontend SPA and backend FastAPI app according to the repo conventions.
2. Add SQLAlchemy models and Alembic migration setup in the backend.
3. Add the initial Alembic migration for `pgvector`, source document, chunk, full-text, chat, and citation tables.
4. Add Supabase Auth in the frontend and token verification in FastAPI.
5. Add the shared frontend API client with automatic bearer-token injection.
6. Add the chat streaming endpoint with a stubbed assistant response.
7. Add AI SDK chat UI on the frontend pointed at FastAPI.
8. Add Markdown ingestion, chunking, embeddings, and Supabase writes.
9. Add semantic search with `pgvector`.
10. Add Postgres full-text search and Python RRF fusion.
11. Add PydanticAI document agent with typed dependencies and typed answer output.
12. Add citation validation and grounding enforcement.
13. Add final UI for citations, source passages, empty states, and errors.

## Non-Goals

- No Next.js, SSR, server components, or frontend route handlers.
- No direct OpenAI calls from the browser.
- No separate managed vector database outside Supabase.
- No multi-tenant architecture.
- No external market/news data.
- No trading recommendations or generated stock picks.
