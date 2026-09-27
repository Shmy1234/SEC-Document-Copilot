# CI foundation

`.github/workflows/ci.yml` runs on pull requests, pushes to `main` and `devops/**`, and manual dispatch. It has three jobs: backend lint/unit tests, frontend lint/type-check/build, and an isolated backend container smoke check. No AWS credentials, real model calls, or database migrations are configured.

Actions are pinned to full commit IDs resolved from their official repositories. Workflow permissions are read-only and checkout credentials are not persisted. Job timeouts bound failures; newer commits cancel older CI runs. This does not configure branch protection or guarantee GitHub usage stays within a particular account's allowance.

The frontend uses pnpm 10.17.1 with the existing seven-day release-age policy and a frozen lockfile. Node 22.17.1 satisfies the Vite requirement recorded in the lockfile. Backend checks use Python 3.12.14 and uv 0.12.5. Development dependencies are installed for backend tests; the Docker runtime excludes them.

The container check runs with synthetic settings, no external network, no published port, a 768 MiB memory cap, and a one-CPU quota. It checks health, the non-root identity, packaged migration/prompt inputs, development-tool exclusions, and failure on missing `OPENAI_FAST_MODEL`. Its memory snapshot is not a production load test or a peak-memory measurement. The shell check targets the Ubuntu CI runner, where `timeout` is available.

GitHub Actions must actually run successfully before these checks are considered verified. Repository settings still need required checks for `Backend lint and tests`, `Frontend lint and build`, and `Backend container smoke check` if branch protection is enabled.

This is CI, not CD: AWS templates, Compose/Caddy hosting, cloud credentials, and deployment automation remain later milestones in the [single checklist](../todos.md#phase-9--devops--deployment-aws--cloudflare). The workflow does not publish images or application artifacts and cannot deploy the site.
