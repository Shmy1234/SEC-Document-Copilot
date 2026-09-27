# Local backend container

This is milestone 9.1 only: package FastAPI and check startup. It does not deploy AWS, run migrations, or validate real chat. Docker Desktop must be running.

From the repository root, build for the proposed x86 Lightsail host. On an Apple Silicon Mac, Docker uses emulation, so build speed and memory measurements do not predict server performance precisely.

```bash
docker build --platform linux/amd64 -t document-copilot-backend:local backend
docker run --detach --name document-copilot-backend-check --platform linux/amd64 --memory 768m --memory-swap 768m --cpus 1 --env-file backend/docker.env.example -p 127.0.0.1:18000:8000 document-copilot-backend:local
curl --fail http://127.0.0.1:18000/health
docker stats --no-stream document-copilot-backend-check
docker inspect --format '{{.State.Health.Status}}' document-copilot-backend-check
docker stop document-copilot-backend-check
docker rm document-copilot-backend-check
```

The memory cap leaves some of a 1 GB host for its operating system and future web server; actual capacity still needs a host test. Wait for startup before requesting health. The health status initially reads `starting` until Docker runs its check.

Check that missing configuration prevents startup. This command must exit unsuccessfully and identify `OPENAI_FAST_MODEL` as missing:

```bash
docker run --rm --platform linux/amd64 --env-file backend/docker.env.example --entrypoint /bin/sh document-copilot-backend:local -c 'unset OPENAI_FAST_MODEL; exec python -m uvicorn app.main:app --host 0.0.0.0 --port 8000'
```

`Dockerfile` is the recipe. The image is the resulting application package. A container is a running copy of that image. The builder stage installs dependencies with uv; the runtime stage keeps the installed application and migration files without uv, notebooks, or test tools. This follows [uv's multi-stage guidance](https://docs.astral.sh/uv/guides/integration/docker/).

The Python and uv image versions are pinned by version tag, not immutable digest; refresh and verify them deliberately during maintenance. The existing `uv.lock` fixes application dependency versions. Builds may still need network access to fetch images, packages, and the project's build backend.

`.dockerignore` permits only the necessary source, lockfile, and migration inputs into Docker's build context. Secrets are supplied when starting a container, never copied into the image. The committed example contains fake values. Do not use it for a real deployment.

The app runs as an ordinary user instead of the administrator. `/health` confirms that the web process responds; it does not prove the database or AI service works. Startup does not run Alembic or ingestion automatically. Later release work must run migrations as a deliberate separate step.

Only your own machine can reach the mapped test port because it is bound to `127.0.0.1`. Uvicorn listens on `0.0.0.0` inside the container so Docker can forward that port to it.
