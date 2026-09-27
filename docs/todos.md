# Document Copilot — implementation checklist

Work top to bottom. Each phase unlocks the next. Check items off as you go.

## Where to start: backend, frontend, or both?

Start with foundation, then backend-led vertical slices.

| Order | Why |
| --- | --- |
| 1. Supabase + sample data | Everything persists here; you need a project and a corpus to test against. |
| 2. Backend schema + migrations | Auth, chat, retrieval, and citations all depend on the data model. |
| 3. Thin vertical slices | Wire auth, then a stubbed chat stream, then real RAG — each slice touches frontend + backend together. |
| 4. Frontend in parallel (lightly) | Scaffold the SPA early, but don't build citation UI or chat polish until the backend can return real grounded answers. |

The critical path is data model → ingestion → retrieval → LLM → citations. The frontend is mostly a streaming chat shell with auth and citation display — it shouldn't get far ahead of working APIs.

## Phase 0 — Prerequisites & foundation

- [x] Install toolchain: Python 3.12+, uv, Node 20+, pnpm (see README)
- [x] Create Supabase project and collect credentials (supabase-setup)
- [x] Create OpenAI API key (needed from Phase 6 onward)
- [ ] Set a real contact email in `USER_AGENT` in `data/download.py` before downloading:

  ```bash
  uv run data/download.py
  ```

- [x] Confirm `data/downloads/manifest.json` lists AAPL, MSFT, NVDA, AMZN, GOOGL filings (2021–2025)

## Phase 1 — Backend scaffold & database

Goal: a running FastAPI service with a migrated Supabase schema.

- [x] Init backend deps and project layout (backend-setup)
- [x] `app/config.py` — settings module, fail fast on missing env vars
- [x] `app/main.py` — FastAPI app, CORS, health check (`GET /health`)
- [x] SQLAlchemy models in `app/database/models/`:
  - [x] `users`
  - [x] `source_documents`
  - [x] `document_chunks` (embedding + generated tsvector)
  - [x] `chat_threads`
  - [x] `chat_messages`
  - [x] `message_citations`
- [x] Alembic init + first migration:
  - [x] `create extension if not exists vector`
  - [x] `vector(1536)` embedding column
  - [x] Generated `tsvector` column on chunks
  - [x] HNSW index (vector) + GIN index (full-text)
  - [x] RLS policies (users see only their own chats)
- [x] Run `uv run alembic upgrade head` against the replacement Supabase direct connection
- [x] `app/database/supabase.py` — user-scoped and service-role clients
- [x] Verify against the replacement environment: `uv run uvicorn app.main:app --reload` → health check returns 200

## Phase 2 — Auth (full stack)

Goal: analysts can sign in with email; backend rejects unauthenticated requests.

### Backend

- [x] `app/auth/dependencies.py` — verify `Authorization: Bearer <supabase_jwt>`, expose `get_current_user`
- [x] Reject missing/expired tokens with 401 before any chat or retrieval work

### Frontend

- [x] Scaffold Vite + React + TypeScript + Tailwind + shadcn (frontend-setup)
- [x] `src/lib/env.ts` — validate `VITE_API_BASE_URL`, `VITE_SUPABASE_URL`, `VITE_SUPABASE_ANON_KEY`
- [x] `src/lib/supabase.ts` — browser Supabase client
- [x] `src/lib/http.ts` + `src/lib/api.ts` — fetch wrapper with automatic bearer token
- [x] Sign-in / sign-up pages (email only, no SSO)
- [x] Protected routes — redirect unauthenticated users to login
- [ ] Verify: sign up, sign in, token reaches backend on a test authenticated endpoint

## Phase 3 — Chat shell (vertical slice, stubbed)

Goal: end-to-end chat UI streaming from FastAPI, no real retrieval yet.

### Backend

- [x] Chat thread CRUD: list threads, create thread, load message history
- [x] `POST /chat/stream` — accepts AI SDK message format, streams a stubbed assistant reply
- [x] Persist user + assistant messages to `chat_messages` after stream completes
- [x] 403 when user accesses another user's thread

### Frontend

- [x] React Router: login, chat list, chat thread routes
- [x] AI SDK chat primitives pointed at `POST /chat/stream` with Supabase bearer token
- [x] Thread sidebar (past conversations)
- [x] Basic message list + input + streaming indicator
- [ ] Verify: create thread, send message, see streamed stub response, reload and see history

## Phase 4 — Ingestion pipeline

Goal: SEC filings in the corpus are parsed, chunked, embedded, and stored in Supabase.

- [x] `ingest/` scripts (or CLI entrypoint) for one-off corpus loading
- [x] HTML → normalized Markdown extraction with stable SEC item headings
- [x] Chunking strategy (size + overlap; store chunk index and available page/section metadata plus ticker, filing type, year)
- [x] Write `source_documents` rows with filing metadata from `manifest.json`
- [x] Write `document_chunks` rows with text + metadata
- [x] OpenAI embedding generation → store `vector(1536)` per chunk
- [x] Generated `tsvector` populated for full-text search
- [x] Idempotent re-run (skip already-ingested documents)
- [x] Unit tests: chunking logic, metadata extraction
- [x] Run ingestion on the replacement Supabase project (25 filings × 5 companies)
- [x] Verify replacement-project chunks; spot-check a known passage (e.g. Apple revenue mix table)

## Phase 5 — Retrieval

Goal: a user question returns ranked, relevant source passages.

- [x] `retrieval/queries.py` — pgvector semantic search over `document_chunks`
- [x] `retrieval/queries.py` — Postgres full-text search over `search_vector`
- [x] `retrieval/fusion.py` — Reciprocal Rank Fusion in Python
- [x] `retrieval/retriever.py` — query → fused ranked passages + neighbor chunks
- [x] Unit tests: fusion ranking, query assembly (mock DB)
- [x] Integration test (optional, `@pytest.mark.integration`): real query against ingested corpus
- [x] Verify: test queries from client-brief return relevant chunks (manual or scripted)

## Phase 6 — LLM agent & grounding

Goal: grounded answers with enforced citations — the core product contract.

- [x] `assistant/instructions.md` — product contract (cite everything, refuse to invent, no stock picks)
- [x] PydanticAI agent with typed deps (`DocumentAgentDeps`) and output (`GroundedAnswer`)
- [x] Agent tools: `search_filings`, `read_chunk`, `read_surrounding_chunks`
- [x] `chat/orchestrator.py` — one turn: retrieve → agent → validate → stream → persist
- [x] `grounding/validator.py` — every citation maps to a retrieved passage; fail closed on violation
- [x] `chat/streaming.py` — AI SDK-compatible stream (text deltas + citation metadata parts)
- [x] Persist `message_citations` linked to assistant messages
- [x] Unit tests: citation validation, grounding enforcement, message conversion
- [ ] Verify against client-brief example questions:
  - [ ] Answers cite specific filings and pages
  - [ ] Under-specified questions get “not enough evidence” responses
  - [ ] Question 10 (generative AI margins) refuses to infer beyond filings

## Phase 7 — Trust UI (citations & source passages)

Goal: analysts can verify every claim in one click — this is what makes the product usable.

- [x] Citation chips/links on assistant messages (company, filing type, date, page/section)
- [x] Source passage panel — show underlying excerpt for selected citation
- [x] Empty states (no threads, no corpus match)
- [x] Error states (auth expired, retrieval failure, grounding failure, network/CORS)
- [x] Loading/streaming status during assistant run
- [ ] Verify: click a citation → see the exact passage from the filing

### Design system pass (see `specs/2026-08-01-phase-7-design-system-design.md`)

- [x] shadcn primitives installed; `components/ui` is generated, not hand-rolled
- [x] Collapsible icon-rail sidebar with grouped thread history and search
- [x] User menu with identity, theme switch, and sign-out
- [x] Threads titled from their first question (`PATCH /chat/threads/{id}`)
- [x] Backend `data-status` / `data-error` stream parts; status reflects real stages
- [x] Markdown answers with tables; auto-growing composer with stop control
- [x] Evidence rail highlights the cited quote inside its surrounding chunk
- [x] Dark mode (system default, manual override, no first-paint flash)
- [ ] Manual browser pass while signed in: sidebar collapse, mobile sheet, live stages,
      citation hover and selection, focus restoration, each error state

## Phase 8 — Pilot readiness

Goal: 5 senior analysts can use it for a week and report ≥3 hours saved per analyst per week.

- [x] README “Running locally” section — copy-paste commands for backend + frontend + env vars
- [x] Seed or document how to ingest/update the corpus
- [ ] Smoke-test all 10 example questions from the client brief
- [x] Confirm chat history persists across sessions
- [x] Confirm ~40-user scale assumptions (no hardcoded single-user shortcuts)
- [x] Basic structured logging on backend (`structlog`) for debugging failed turns
- [x] Review latency: streaming starts within a few seconds for typical queries

## Phase 9 — DevOps & deployment (AWS + Cloudflare)

Status: **planned, not implemented**. Updated 2026-09-27 after the user set a $10/month AWS + Cloudflare ceiling, excluding Supabase and OpenAI. Currency remains unconfirmed; vendor estimates are USD before tax. This replaces the earlier ECS/ALB/NAT proposal. Architecture and tradeoffs belong in [architecture.md](architecture.md#deployment-shape); current evidence belongs in [progress-tracker.md](progress-tracker.md).

This file remains the single project checklist. New deployment tasks use `[]` not started, `[*]` in progress, and `[x]` verified complete. Historical phase statuses above are preserved, not revalidated. Each work package below is split into small execution milestones when selected; it is not a promise to complete an entire package in five minutes.

Target: Cloudflare DNS → one Lightsail Linux host → Caddy + Docker Compose → FastAPI; Caddy also serves the Vite build. Retain Supabase and OpenAI. Accept a single point of failure and brief deployment downtime for an invite-only pilot.

### 9.0 — Decisions before paid infrastructure

Outcome: a costed pilot scope with known tradeoffs. Boundary: planning only.

- [] Confirm whether the $10 ceiling is USD or CAD, and whether tax and domain renewal must fit it.
- [] Confirm domain ownership, audience, relevant regions/data requirements, and AWS/Supabase locations.
- [] Price the $7 USD/month public-IPv4, 1 GB Linux Lightsail candidate in the chosen region. Include transfer, snapshots, temporary test instance hours, artifact retention, and taxes. Do not rely on trial credits.
- [] Confirm GitHub Actions minutes/storage allowance and Supabase backup plan separately; keep model costs outside this hosting estimate but bounded.
- [] Agree pilot downtime, maximum simultaneous chats, acceptable data loss, and recovery time. Define initial targets before verification, not after observing results.

Acceptance: documented estimate fits the confirmed ceiling, with explicit exclusions and an agreed response if the smallest suitable instance does not fit.

### 9.1 — Next milestone: local backend container

Outcome: reproducible startup without installing development tooling into production. Boundary: local files and checks only; no account writes, migrations, or paid API calls.

2026-09-27 implementation status: Docker recipe, build-context exclusions, synthetic configuration, and [local run guide](guides/backend-docker.md) added. Runtime dependency export and static checks pass. Container verification is blocked because the local Docker engine does not respond; this milestone is not complete.

- [*] Add `backend/Dockerfile` with a pinned supported Python base, locked production dependencies using uv, a non-root runtime user, and one Uvicorn worker on `0.0.0.0:8000` without reload. Recipe written; image build pending.
- [*] Add `backend/.dockerignore` excluding `.env`, caches, virtual environments, Git data, tests where unnecessary, and corpus payloads. Include required prompt files, ingestion package, and Alembic files in the image deliberately. Allowlist written; image-content inspection pending.
- [] Build for the eventual host CPU architecture; do not assume an Apple Silicon local build runs on an x86 host.
- [] Run locally with synthetic required settings and verify `/health` returns 200. Inspect image configuration/layers for accidental secrets; do not print real values.
- [] Verify removing `OPENAI_FAST_MODEL` causes startup failure. Record cold-start memory with the candidate memory limit and stop if it exceeds the budgeted host capacity.

Acceptance: a fresh build starts with synthetic configuration, returns health, fails on missing required configuration, and contains no supplied secret files. This does not verify Supabase/OpenAI connectivity or production capacity.

### 9.2 — Pull-request checks and repository hygiene

2026-09-27 status: [CI workflow](guides/ci.md), isolated container smoke script, and deployment ignore rules added. Shell syntax and ignore-rule checks pass. First GitHub run, branch protection, full dependency/security scanning, and deployment automation remain unverified or unimplemented.

Outcome: unsafe or broken changes fail before deployment. Boundary: repository and CI configuration; enable remote settings only with authorized account access.

- [] Pin supported Python, Node, uv, and pnpm versions; honor existing release-age policies and lockfiles. Check Vite's actual Node requirement rather than relying on an old README minimum.
- [*] Add `ci.yml`: `uv sync --frozen`, `uv run ruff check .`, `uv run pytest -m "not integration"`, `pnpm install --frozen-lockfile`, `pnpm lint`, `pnpm build`, and a backend image build.
- [] Keep PR jobs free of production credentials and paid/live tests. Pin third-party Actions to reviewed full commit SHAs, use minimal token permissions, and set job timeouts.
- [] Require the checks on the main branch, review changes to workflows/infrastructure, and use small PRs with acceptance evidence. Keep required check reporting reliable for path-filtered jobs.
- [] Add suitable secret/dependency/image scanning within available free tooling; triage findings with an owner and fix/exception decision rather than equating a clean scan with security.
- [] Repair the live schema test's stale `0003` expectation against the actual migration head and verify RLS/functions as well as revision. Treat corpus-count assertions as fixture-specific.

Acceptance: a deliberately failing unit/lint/build check blocks merging; clean PR checks need no live credentials. No frontend test runner is introduced, per `frontend/AGENTS.md`.

### 9.3 — Account security and Lightsail infrastructure

Outcome: one reproducible, budgeted host. Boundary: approved pilot resources only.

- [] Enable root MFA and recovery methods; use a non-root administrative identity with temporary sessions for setup. Protect Cloudflare and GitHub accounts with MFA too.
- [] Configure budget notifications at useful fractions of the confirmed ceiling and review projected costs. Alerts are not a hard spending limit.
- [] Create a small CloudFormation template for the chosen Lightsail instance, attached static IP, tags, and baseline firewall. Review a change set before creation; avoid automated deletion/replacement of the only live host.
- [] Add a reviewed, idempotent OS/Docker/Compose bootstrap script with pinned supported sources, security updates, reboot procedure, log rotation, and disk limits. Never embed credentials in user data.
- [] Permit public HTTP/HTTPS only; restrict SSH to approved source addresses. Keep API/container admin ports private and verify from outside the host.
- [] Provision production secrets separately into restricted files; verify only the administrator and intended runtime identity can read them. Record recovery and rotation procedures without secret values.

Acceptance: the host can be rebuilt from reviewed definitions; only intended ports are reachable; no unused billable infrastructure remains. Confirm static-IP/snapshot billing rules before cleanup.

### 9.4 — Compose, frontend, Cloudflare, and HTTPS

Outcome: a manually verified HTTPS pilot. Boundary: one host and DNS-only Cloudflare records.

- [] Add `deploy/compose.yaml` and `deploy/Caddyfile`; Caddy serves versioned frontend assets and proxies the API over a private container network. Persist Caddy certificate state and set restart/log policies.
- [] Build the frontend off-host with browser-safe `VITE_` settings. Include `OPENAI_FAST_MODEL` and all other required settings in the deployment configuration inventory; never place backend secrets in Vite variables.
- [] Add the domain to Cloudflare Free, preserve mail/TXT records, verify DNSSEC handling, and change nameservers only after checking the complete record set.
- [] Create DNS-only `app` and `api` A records for the static IP. Obtain Caddy public certificates and verify renewal/storage; configure Supabase site/redirect URLs and exact backend CORS origins.
- [] Configure SPA fallback for navigations, correct missing-asset errors, revalidated HTML, long-lived hashed assets, and bounded old-asset retention for rollback/open browser sessions.
- [] Apply migrations once against a non-production/synthetic environment first. Verify direct/session database connectivity from the host. Run a bounded live chat only with authorized credentials/spend.

Acceptance: valid HTTPS, HTTP redirect, direct chat-page refresh, sign-in, streamed answer, citation opening, and history reload all work. Record peak memory/CPU/disk; a 1 GB host is accepted only on evidence.

### 9.5 — Reproducible releases and rollback

Outcome: release a tested commit with a known recovery path. Boundary: one protected pilot environment; no zero-downtime promise.

- [] Add separate infrastructure and application workflows. CloudFormation owns host resources; application deploys must not replace the host.
- [] Configure GitHub OIDC restricted to this repository and production environment. Prototype Lightsail temporary SSH access, trusted host-key verification, and a runner-IP-only temporary SSH firewall rule with guaranteed cleanup verification.
- [] Build once on CI; record commit, backend image identity/checksum, frontend archive, migration revision, and configuration version. Transfer artifacts securely; retain current/previous releases within storage quotas. Do not expose private images publicly.
- [] Serialize deployments with a remote lock as well as GitHub concurrency; never cancel halfway through migration without a recovery decision. Use environment approval if supported by the repository plan.
- [] Enter maintenance/drain mode, bound in-flight work, stop the old API if needed for memory, run the release migration once, then start the new API and publish compatible frontend assets.
- [] Verify the actual release identifier plus health/auth/stream checks. On failure, restore the previous image/frontend when schema-compatible; report deployment failure even if rollback succeeds. Test this with a deliberately unhealthy release in a safe environment.

Acceptance: a checked commit is traceable to the running release; failed deployment is visible; rollback restores a working application; temporary access is revoked after success and failure. If safe CI access cannot be verified, keep deployment manual until resolved.

### 9.6 — Application security and AI cost controls

Outcome: invited users can access only their own data within bounded work. Boundary: targeted backend changes, each selected as a small milestone.

- [] Verify authentication and account-B-versus-account-A isolation across thread list/read/update/delete, streaming, citations, and tool paths. Service-role database calls require explicit backend ownership checks even when RLS exists.
- [] Set input/history limits, request deadlines, outbound timeouts/retries, per-user quotas, and concurrent-turn limits. Existing agent request/tool limits do not cap total user spending. Enforce shared quotas across processes using the current database where practical.
- [] Verify retries/disconnects cannot accidentally duplicate paid turns; document whether a disconnected turn continues, fails, or can be resumed. Add request identifiers/idempotency only for the defined behavior.
- [] Treat filings/model output as untrusted; preserve citation validation and bounded tool access. Keep ingestion private and separately authorized.
- [] Review CSP and other browser headers, CORS, error responses, and log redaction against the actual frontend/auth integrations. Start restrictive headers in a verifiable rollout rather than breaking sign-in blindly.
- [] Before real-user onboarding, settle data categories, audience/regions, retention/deletion, provider handling, and appropriate notices. Validate implemented deletion and backup retention; do not infer legal compliance from these checks.

Acceptance: unauthorized access is denied before data/model work, quota exhaustion stops new work predictably, and errors/logs contain no credentials or unnecessary private chat content.

### 9.7 — Monitoring, backup, and recovery

Outcome: detect failures and recover within agreed pilot targets. Boundary: fit the confirmed hosting budget; do not silently enable paid monitoring.

- [] Distinguish cheap liveness checks from bounded readiness/live smoke checks. Monitor external HTTPS, Lightsail resource metrics, memory/disk locally, certificate expiry, container restarts, and billing.
- [] Rotate/cap structured logs. Record request IDs, release IDs, timings, typed failures, and aggregate model usage; avoid raw tokens, prompts, and private passages. Identify one person responsible for alerts.
- [] Verify the actual Supabase plan's backup coverage and retention. If using exports, encrypt them off-host, restrict access, budget storage, and check coverage of roles/auth/schema/data and any storage objects separately.
- [] Reconstruct a test database from a backup and verify chats, ownership policies, citations, vector dimensions, and migration state. Measure recovery time/data loss against agreed targets.
- [] Create a short runbook for host rebuild, restore, application rollback, leaked-key rotation, certificate/DNS failure, disk exhaustion, interrupted migration, and model/Supabase outage. Keep checklist ownership here; the runbook contains procedures/evidence.

Acceptance: a real restore/rebuild exercise works, alerts reach the responsible person, and logs/disk/backups have bounded growth. Supabase plan-specific backup behavior must be checked: [official guidance](https://supabase.com/docs/guides/platform/backups).

### 9.8 — Pilot release gate

Outcome: an evidence-backed invite-only release. Boundary: the agreed pilot load; no public-scale claims.

- [] Complete [pilot-readiness.md](guides/pilot-readiness.md), including the ten-question grounding review and account isolation.
- [] Verify keyboard navigation, focus, narrow screens, loading/error states, and citation accessibility manually, consistent with the frontend testing policy.
- [] Test the agreed concurrent workload with synthetic data and capped model calls; record memory, response timing, failure rate, and provider/database limits. Separate mock throughput results from live end-to-end evidence.
- [] Verify sign-up policy/email confirmation, SMTP delivery/configuration if needed, data notices, secrets, recovery evidence, known downtime, and monthly cost estimate before invitations.
- [] Record release identifiers, checks, unresolved risks, owner, and rollback procedure. Review actual spend and failures after the first day and week.

Acceptance: all required release checks have evidence or an explicitly accepted scope reduction; no silent upgrade to a larger AWS plan.

### 9.9 — Maintenance and optional growth

Outcome: keep a small system operable without adding unnecessary services.

- [] Schedule a human-reviewed cadence for dependencies/base images, OS security updates/reboots, vulnerability triage, restore checks, access review, and cost review. Automations require their own authorization.
- [] Maintain a small retrieval/grounding evaluation set when prompts/models/chunking change; record model/configuration and compare citation quality, latency, and bounded cost.
- [] If Cloudflare proxying is selected, verify Free-plan capabilities, Full (strict) TLS, API cache bypass, streaming, origin access restrictions, certificate renewal, and the changed data flow before enabling it.
- [] Revisit ECS/ALB, managed secrets/logs, separate staging, S3/CloudFront, or higher availability only after budget increases or measured requirements justify them.

Acceptance: future work has a measured reason, a cost estimate, and a small observable check.

Provider guidance: [GitHub workflow security](https://docs.github.com/en/actions/reference/security/secure-use), [AWS Lightsail bundles](https://docs.aws.amazon.com/lightsail/latest/userguide/amazon-lightsail-bundles.html), and the linked sources in [architecture.md](architecture.md#deployment-shape). Verify account-specific pricing/quotas before implementation.

## Quick reference

| Doc | Purpose |
| --- | --- |
| `client-brief.md` | What Equity Research Assistant needs and example questions |
| `architecture.md` | System design, data model, streaming contract |
| `guides/supabase-setup.md` | Hosted Postgres + Auth |
| `guides/backend-setup.md` | FastAPI + Alembic commands |
| `guides/frontend-setup.md` | Vite + React scaffold commands |
