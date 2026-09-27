# Current deployment work

Updated 2026-09-27. The single checklist is [docs/todos.md, Phase 9](todos.md#phase-9--devops--deployment-aws--cloudflare); architecture decisions belong in [architecture.md](architecture.md#deployment-shape).

Current milestone: user-authorized Docker/CI foundation and repository push. Container runtime verification from 9.1 remains pending. Dockerfile, .dockerignore, synthetic docker.env.example, and a beginner-oriented build/run guide have been added. Verification is blocked on the local Docker engine. The user specified a $10/month AWS + Cloudflare ceiling excluding Supabase/OpenAI. The plan now proposes one Lightsail host with Docker Compose/Caddy and Cloudflare Free DNS. ECS/ALB/NAT and the earlier Railway target are deferred/replaced.

Evidence: the production-only lockfile export succeeds offline and excludes pytest, ruff, docling, ipykernel, and torch. Direct runtime version pins match uv.lock; required prompt/ingestion/migration inputs exist. `git diff --check` passes. Docker Desktop startup was attempted; the daemon socket appeared but the engine did not answer its ping. The amd64 build and diagnostic calls were canceled after stalling. No image build, container health, missing-setting failure, image-layer inspection, or memory check has passed. No cloud resources, migrations, real API calls, or deployment were performed.

Open decisions: budget currency/tax treatment and domain costs; domain ownership; AWS/Supabase region/connectivity; real-user audience/data requirements; pilot concurrency and downtime; backup/recovery objectives; GitHub quota/protection availability. A 1 GB host's capacity is unverified. The existing live test expects migration 0003 while the repo contains 0004; correcting and verifying that test remains implementation work.

CI additions: three GitHub Actions jobs (backend, frontend, container), an isolated container smoke script, and AWS/deployment ignore rules. Action commit IDs were verified against official GitHub repositories. Shell syntax and eight ignored/six allowed path checks passed. Local application tests are not available because dependencies are not installed; remote CI results must be checked after push. No AWS/CD deployment is configured.

Next action: review the first CI run after push; get Docker Desktop to report a running engine, then resume the build and verification commands in [backend-docker.md](guides/backend-docker.md). Check Git state before continuing. Complete 9.1 before selecting CI work. Confirm budget currency/domain costs before any paid infrastructure.
