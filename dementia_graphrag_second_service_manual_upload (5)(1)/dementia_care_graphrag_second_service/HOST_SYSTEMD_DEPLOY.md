# Host systemd deployment

Use this method when Docker or Docker Hub is unavailable and an existing Python environment already contains the required dependencies.

## Assumptions

- Server IP: `YOUR_SERVER_IP`
- Existing service port: `8000`
- Second service port: `8010`
- Project directory: `/root/graphrag_second_service`
- Python runtime: `/root/miniconda3/envs/dementia_graphrag_api/bin/uvicorn`

Adjust these values according to the target server.

## Configure

Copy the environment file from the existing deployment or create one from `.env.example`, then set the host Neo4j address:

```bash
cp /root/dementia_care_graphrag_api_deploy_minimal/.env /root/graphrag_second_service/.env
sed -i 's#^NEO4J_URI=.*#NEO4J_URI=bolt://127.0.0.1:7688#' /root/graphrag_second_service/.env
sed -i 's#^API_AUTH_TOKEN=.*#API_AUTH_TOKEN=#' /root/graphrag_second_service/.env
grep -q '^GRAPH_RAG_PORT=' /root/graphrag_second_service/.env || echo 'GRAPH_RAG_PORT=8010' >> /root/graphrag_second_service/.env
```

## Install service

```bash
mkdir -p /root/graphrag_second_service/outputs/logs
cp /root/graphrag_second_service/deploy/dementia-graphrag-second.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable dementia-graphrag-second
systemctl restart dementia-graphrag-second
```

## Verify

```bash
systemctl status dementia-graphrag-second
curl http://127.0.0.1:8010/health
curl http://YOUR_SERVER_IP:8010/health
```

Web page:

```text
http://YOUR_SERVER_IP:8010/
```

## Logs and operations

```bash
# View logs
tail -f /root/graphrag_second_service/outputs/logs/service_8010.log

# Restart
systemctl restart dementia-graphrag-second

# Stop
systemctl stop dementia-graphrag-second

# Check old service
curl http://127.0.0.1:8000/health
```

## Compatibility note

Neo4j 4.4 does not support the Neo4j 5 `elementId()` function. The retrieval code in this package uses `id(a)` for compatibility.
