# Local GitOps RAG over MCP

This repository packages a local Retrieval Augmented Generation stack that forces Cursor (or any MCP-aware client) to treat Git as the only source of truth. The system mirrors GitLab repositories, incrementally indexes code, and exposes hardened MCP tools that *must* be called before an assistant answers any engineering question.

## Components

| Service | Responsibility |
| ------- | -------------- |
| **poller** | Maintains `git clone --mirror` copies of the allow-listed GitLab repositories and triggers re-indexing whenever a branch changes. |
| **indexer** | Computes git deltas, chunks textual files, generates embeddings via TEI, and syncs BM25 documents to Meilisearch and vectors to Qdrant. Also writes blobs to the local store and persists branch head state. |
| **mcp-rag** | Exposes MCP tools (`gitops_assert`, `search_code`, `get_file`, `grep`, `explain_symbol`) with strict JSON schemas, a policy resource, intent classifier, and health endpoint. Enforces "call tools first" behaviour. |
| **rerank** | FastAPI service running `BAAI/bge-reranker-large` on GPU for cross-encoder re-ranking. |
| **emb** | HuggingFace Text Embeddings Inference serving `BAAI/bge-m3` for dense vectors. |
| **meilisearch** | BM25 retrieval backend for code chunks. |
| **qdrant** | Vector database storing embeddings and metadata. |
| **blob store** | Volume-mounted hierarchy containing full file contents per repo/branch. |

All services communicate on the internal `ragnet` Docker network and share `/data` volumes for repos, blobs, indexer state, and cached models.

## Configuration

The default configuration lives in [`config.yaml`](./config.yaml). Adjust it to match your GitLab organisation:

```yaml
gitlab:
  url: ssh://git@gitlab.int.bachatka.ru
  deploy_key_path: /secrets/gitlab_readonly_key
  poll_interval_sec: 30

projects:
  - repo: bachatka/backend
    branches: [main, develop]
    denylist:
      - "secrets/**"
      - "infra/keys/**"
      - "**/*.pem"
      - "**/*.key"
  - repo: bachatka/frontend
    branches: [main]
    denylist:
      - "secrets/**"

chunking:
  max_chars: 1800
  overlap: 200

services:
  meilisearch_url: http://meilisearch:7700
  qdrant_url: http://qdrant:6333
  tei_url: http://emb:8080
  rerank_url: http://rerank:8081
  blob_root: /data/blobs
  repo_root: /data/repos
  state_path: /data/state/indexer-state.json

mcp:
  host: 0.0.0.0
  port: 8765
  enforce_meta_tool: true
  server_description: "Git is the only source of truth. Always call MCP tools before answering."
  classification_enabled: true

limits:
  http_timeout_sec: 10
  max_parallel_index_tasks: 4
```

Place the **read-only GitLab deploy key** at `./secrets/gitlab_readonly_key` (chmod `0400`). The compose file mounts `./secrets` into each service so that SSH operations can authenticate against GitLab.

## Running the stack

1. Create required directories:
   ```bash
   mkdir -p secrets && chmod 700 secrets
   # copy your read-only deploy key
   cp /path/to/id_rsa_gitlab secrets/gitlab_readonly_key
   chmod 400 secrets/gitlab_readonly_key
   ```
2. Adjust [`config.yaml`](./config.yaml) to enumerate the repositories and branches you want to index.
3. Start everything with Docker Compose:
   ```bash
   docker compose up -d --build
   ```
4. Wait for health checks (all services should be healthy within ~60 seconds once models are warmed).

### Verifying ingestion

* Trigger a manual re-index via the `indexer` API:
  ```bash
  curl -X POST http://localhost:8000/reindex \
    -H 'Content-Type: application/json' \
    -d '{"repo":"bachatka/backend","branch":"main"}'
  ```
  Poll `GET /tasks/<task_id>` to observe progress.
* Inspect Meilisearch and Qdrant to ensure new chunks and vectors are present (`docker compose exec meilisearch curl http://localhost:7700/indexes`).
* Validate blob persistence: files appear under `./data/blobs/<repo>/<branch>/...` in the mounted volume.

### MCP integration

1. Point Cursor (or any MCP client) at the server by editing `~/.cursor/mcp.json`:
   ```json
   {
     "servers": {
       "gitops": {
       "command": "python",
       "args": ["services/mcp_rag/server.py"],
         "description": "Git is the only source of truth. Always call search_code/get_file/grep/explain_symbol before answering.",
         "env": {"RAG_ENDPOINT": "tcp://127.0.0.1:8765"}
       }
     }
   }
   ```
2. Add the workspace/system prompt: `Full GitOps. You MUST call MCP tools for any code/config/doc question. If nothing is found — say 'NOT FOUND IN GIT'.`
3. Upon session start, call `gitops_assert`. The response echoes the GitOps policy and timestamp. Every answer about code/config must be preceded by `search_code` (and optionally `get_file`, `grep`, `explain_symbol`).
4. The `/intent` custom route provides a light-weight classifier that clients can use to decide when to surface GitOps warnings.

### Tool schemas (strict)

| Tool | Required inputs | Behaviour |
| ---- | ----------------| ----------|
| `gitops_assert()` | none | Returns GitOps policy text with timestamp. Non-negotiable. |
| `search_code(query, limit=16)` | `query` (`minLength=1`), optional `limit` (`1-64`) | Performs hybrid Meilisearch + Qdrant search with reranking; returns chunk metadata. |
| `get_file(path, repo, branch)` | all `minLength=1` | Reads sanitized blob content, returning `{path,size,sha,content}`. |
| `grep(path, pattern, repo, branch, max_lines=200)` | same as `get_file` plus regex `pattern` | Regular-expression search over stored blob lines. |
| `explain_symbol(symbol, lang?, repo?, branch?)` | `symbol` (`minLength=1`) | Focused symbol lookup filtered by repo/branch when provided. |

Every schema blocks unknown fields (`additionalProperties: false`) to encourage repeated tool invocation on malformed payloads.

## Health & observability

* `indexer`: `GET /healthz` / `GET /readyz`
* `rerank`: `GET /healthz`
* `mcp-rag`: `GET /healthz`
* `emb`: `GET /health`
* `qdrant`: `GET /readyz`
* `meilisearch`: `GET /health`

Logs are JSON-formatted (`ts`, `lvl`, `svc`, `event`, etc.), making them ingest-friendly. Extend the services to export Prometheus metrics such as `indexer_docs_total`, `search_latency_ms`, and `mctr_ratio` as needed.

## Test plan (Definition of Done)

1. `docker compose up -d` brings the full stack to a healthy state within 60 s after warm-up.
2. `POST /reindex` against the indexer ingests a test repository:
   * New/updated files produce documents in Meilisearch and vectors in Qdrant.
   * Deletions remove both chunk documents and vectors.
3. `search_code`, `get_file`, `grep`, and `explain_symbol` return accurate responses referencing the blob store. Out-of-scope requests fail fast with validation errors.
4. Search latency meets SLOs (P50 ≤ 300 ms, P95 ≤ 800 ms) on a warm cache; rerank gracefully degrades when the GPU service is unavailable.
5. MCP logs demonstrate an MCP Tool-Call Rate ≥ 90 % for code-related prompts (sample ≥ 200 events).
6. No secrets or binary blobs leak into indexes thanks to deny-list enforcement.

## Development tips

* Use `docker compose logs -f indexer` to debug delta processing.
* To simulate policy enforcement, call the `/intent` endpoint: `curl -X POST http://localhost:8765/intent -d '{"question":"How does the Helm chart work?"}' -H 'Content-Type: application/json'`.
* When extending chunking logic, modify [`src/rag_common/chunking.py`](./src/rag_common/chunking.py) and re-build the indexer.

## Licensing

All third-party models (TEI & reranker) retain their upstream licenses. Ensure you comply with the licensing terms of `BAAI/bge-m3` and `BAAI/bge-reranker-large` before deploying in production.
