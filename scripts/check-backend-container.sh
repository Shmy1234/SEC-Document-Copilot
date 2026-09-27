#!/usr/bin/env bash
set -euo pipefail

image="${1:?Usage: bash scripts/check-backend-container.sh IMAGE}"
container="document-copilot-smoke-$$"
missing_container="${container}-missing"
result_dir="$(mktemp -d)"
cleanup() {
  docker rm -f "$container" "$missing_container" >/dev/null 2>&1 || true
  rm -rf "$result_dir"
}
trap cleanup EXIT

# No published ports or outbound network: synthetic settings cannot reach providers.
docker run --detach --name "$container" --network none \
  --memory 768m --memory-swap 768m --cpus 1 \
  --env-file backend/docker.env.example "$image" >/dev/null

healthy=false
for attempt in {1..60}; do
  if docker exec "$container" python -c \
    'import json, urllib.request; response = urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=2); assert response.status == 200; assert json.load(response) == {"status": "ok"}' \
    >/dev/null 2>&1; then
    healthy=true
    break
  fi
  if [ "$(docker inspect --format '{{.State.Running}}' "$container")" != true ]; then
    break
  fi
  sleep 1
done
if [ "$healthy" != true ]; then
  docker logs "$container"
  echo 'Container did not become healthy.' >&2
  exit 1
fi

docker exec "$container" python -c '
import importlib.util
import os
from pathlib import Path
import app.assistant.agent
assert os.getuid() == 10001
assert Path("/app/alembic/env.py").is_file()
assert Path(app.assistant.agent.__file__).with_name("instructions.md").is_file()
assert not any(Path("/app").rglob(".env"))
for name in ("pytest", "ruff", "docling", "ipykernel", "torch"):
    assert importlib.util.find_spec(name) is None, name
print("Health, runtime identity, packaged inputs, and dev-tool exclusions passed.")
'
docker stats --no-stream "$container"
docker stop --time 10 "$container" >/dev/null

if timeout 30s docker run --rm --name "$missing_container" --network none \
  --env-file backend/docker.env.example --entrypoint /bin/sh "$image" \
  -c 'unset OPENAI_FAST_MODEL; exec python -m uvicorn app.main:app --host 0.0.0.0 --port 8000' \
  >"$result_dir/missing-config.log" 2>&1; then
  echo 'Startup unexpectedly accepted missing OPENAI_FAST_MODEL.' >&2
  exit 1
else
  exit_status=$?
fi
if [ "$exit_status" -eq 124 ]; then
  echo 'Missing-setting check timed out instead of rejecting configuration.' >&2
  exit 1
fi
grep -q OPENAI_FAST_MODEL "$result_dir/missing-config.log"
grep -q 'Field required' "$result_dir/missing-config.log"
echo 'Missing required configuration was rejected.'
