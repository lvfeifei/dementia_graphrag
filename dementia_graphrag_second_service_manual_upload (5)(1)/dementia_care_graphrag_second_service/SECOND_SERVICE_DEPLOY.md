# Second GraphRAG Service Deployment

This package deploys an independent copy of the dementia care GraphRAG API and a built-in chat web page. It does not modify the existing online GraphRAG service.

## 1. What is included

- FastAPI GraphRAG service
- Built-in chat web page at `/`
- API docs at `/docs`
- Health check at `/health`
- Docker image definition
- Docker Compose configuration
- Persistent Docker volume for sessions, answers, PDF files, ICS files, and logs

Default web port: `8010`

## 2. Prepare VPN and server access

On your local computer, connect to the company VPN, then connect to the server:

```bash
ssh -i /path/to/your-server-key.pem -p 22 root@YOUR_SERVER_IP
```

The current execution environment cannot install the local VPN client or use your SSH private key, so the final server upload and startup must be run from your computer or a machine with VPN access.

## 3. Upload the package

From your local computer:

```bash
scp -i /path/to/your-server-key.pem \
  dementia_care_graphrag_second_service.tar.gz \
  root@YOUR_SERVER_IP:/root/
```

On the server:

```bash
cd /root
mkdir -p graphrag_second_service
tar -xzf dementia_care_graphrag_second_service.tar.gz -C graphrag_second_service
cd graphrag_second_service
```

## 4. Configure environment

```bash
cp .env.example .env
vi .env
```

Required values:

```env
GRAPH_RAG_PORT=8010
NEO4J_URI=bolt://host.docker.internal:7688
NEO4J_USER=neo4j
NEO4J_LOCAL_PASSWORD=replace_with_real_password
DEEPSEEK_API_KEY=replace_with_real_key
API_AUTH_TOKEN=
```

Neo4j address guidance:

- If Neo4j runs on the server host and publishes port `7688`, keep `bolt://host.docker.internal:7688`.
- If Neo4j runs in another Docker container and does not publish a port, put both containers on the same Docker network and use `bolt://neo4j-container-name:7687`.
- If Neo4j is on another machine, use `bolt://server-ip:port`.

Leave `API_AUTH_TOKEN` empty for the built-in web page. The browser page cannot safely store a backend API token. If token protection is required, place the page behind an authenticated reverse proxy or create a separate external API credential.

## 5. Start the service

```bash
docker compose up -d --build
```

Check status:

```bash
docker compose ps
docker compose logs -f graphrag-second
```

Test from the server:

```bash
curl http://127.0.0.1:8010/health
```

Open in a browser through VPN:

```text
http://YOUR_SERVER_IP:8010/
```

## 6. Firewall

If the server uses firewalld and the port is not reachable:

```bash
firewall-cmd --state
firewall-cmd --list-ports
firewall-cmd --zone=public --add-port=8010/tcp --permanent
firewall-cmd --reload
```

Do not expose this port to the public internet unless it is protected by a gateway or reverse proxy.

## 7. Common operations

```bash
# Stop
docker compose down

# Restart
docker compose restart graphrag-second

# Rebuild after code changes
docker compose up -d --build

# Inspect persistent files
docker volume inspect dementia-care-graphrag-second-outputs

# Enter the container
docker exec -it dementia-care-graphrag-second /bin/bash
```

## 8. Interface used by the page

The page sends:

```json
{
  "session_id": "web_xxx",
  "question": "user question",
  "reset_session": false,
  "memory_context_mode": "auto",
  "include_active_problem_ids": true,
  "temperature": 0.2,
  "max_tokens": 3200
}
```

to:

```text
POST /chat
```

The current service already accepts a session ID and stores server-side session files. The next multi-turn task can focus on conversation history display, context summarization, and follow-up question handling.

## 9. Verification checklist

- `GET /health` returns `success: true`
- The browser opens `/`
- A question appears in the chat thread
- The assistant returns Markdown content
- Matching problem IDs and evidence sources are displayed
- Restarting the container keeps session and output files in the Docker volume
