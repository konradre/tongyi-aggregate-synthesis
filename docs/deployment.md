# Deployment Guide

## Docker Deployment (Recommended)

### Single Container

```bash
# Build
docker build -t research-tool .

# Run
docker run -d \
  --name research-tool \
  -p 8000:8000 \
  -e RESEARCH_SEARXNG_HOST="http://192.168.1.3:8888" \
  -e RESEARCH_LLM_API_BASE="http://172.17.0.1:8080/v1" \
  research-tool
```

### Docker Compose

```bash
# Start
docker compose up -d

# View logs
docker compose logs -f

# Stop
docker compose down
```

### Custom docker-compose.yml

```yaml
version: "3.8"

services:
  research-tool:
    build: .
    container_name: research-tool
    restart: unless-stopped
    ports:
      - "8000:8000"
    environment:
      # Your SearXNG instance
      RESEARCH_SEARXNG_HOST: "http://192.168.1.3:8888"
      RESEARCH_SEARXNG_ENGINES: "google,bing,duckduckgo"

      # Optional API keys
      RESEARCH_TAVILY_API_KEY: "${TAVILY_API_KEY:-}"
      RESEARCH_LINKUP_API_KEY: "${LINKUP_API_KEY:-}"

      # LLM server on host
      RESEARCH_LLM_API_BASE: "http://172.17.0.1:8080/v1"
      RESEARCH_LLM_MODEL: "tongyi-deepresearch-30b"
      RESEARCH_LLM_TEMPERATURE: "0.85"
      RESEARCH_LLM_MAX_TOKENS: "8192"
    extra_hosts:
      - "host.docker.internal:host-gateway"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/api/v1/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

## VM Deployment

### Prerequisites

On target VM:
- Docker and Docker Compose installed
- LLM server running (llama.cpp, vLLM, etc.)
- Network access to SearXNG instance

### Deploy Script

```bash
# From your local machine
./scripts/deploy.sh 192.168.1.119
```

Or manually:

```bash
# Sync files
rsync -avz --exclude '.git' --exclude '__pycache__' \
  . user@192.168.1.119:/home/user/research-tool/

# SSH and start
ssh user@192.168.1.119
cd /home/user/research-tool
docker compose up -d --build
```

### Systemd Service (Optional)

Create `/etc/systemd/system/research-tool.service`:

```ini
[Unit]
Description=Research Tool
Requires=docker.service
After=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=/home/user/research-tool
ExecStart=/usr/bin/docker compose up -d
ExecStop=/usr/bin/docker compose down

[Install]
WantedBy=multi-user.target
```

Enable:

```bash
sudo systemctl enable research-tool
sudo systemctl start research-tool
```

## Production Deployment

### Reverse Proxy (nginx)

```nginx
upstream research-tool {
    server 127.0.0.1:8000;
}

server {
    listen 80;
    server_name research.example.com;

    location / {
        proxy_pass http://research-tool;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Timeout for slow LLM responses
        proxy_read_timeout 120s;
    }
}
```

### TLS with Traefik

```yaml
version: "3.8"

services:
  research-tool:
    build: .
    labels:
      - "traefik.enable=true"
      - "traefik.http.routers.research.rule=Host(`research.example.com`)"
      - "traefik.http.routers.research.entrypoints=websecure"
      - "traefik.http.routers.research.tls.certresolver=letsencrypt"
      - "traefik.http.services.research.loadbalancer.server.port=8000"
    networks:
      - traefik

networks:
  traefik:
    external: true
```

### Basic Authentication

Add nginx basic auth:

```nginx
location / {
    auth_basic "Research Tool";
    auth_basic_user_file /etc/nginx/.htpasswd;

    proxy_pass http://research-tool;
    # ...
}
```

Create password file:

```bash
htpasswd -c /etc/nginx/.htpasswd username
```

## Infrastructure Examples

### Alongside llama.cpp

```
┌─────────────────────────────────────────┐
│                   VM                     │
├─────────────────────────────────────────┤
│                                          │
│  ┌────────────────────────────────────┐ │
│  │        llama.cpp server            │ │
│  │        Port: 8080                  │ │
│  │        Model: Tongyi 30B           │ │
│  └────────────────────────────────────┘ │
│                    ▲                     │
│                    │ http://172.17.0.1:8080
│                    │                     │
│  ┌────────────────────────────────────┐ │
│  │     Research Tool Container        │ │
│  │     Port: 8000                     │ │
│  └────────────────────────────────────┘ │
│                                          │
└─────────────────────────────────────────┘
```

### With SearXNG on Same Host

```yaml
version: "3.8"

services:
  searxng:
    image: searxng/searxng
    container_name: searxng
    ports:
      - "8888:8080"
    volumes:
      - ./searxng:/etc/searxng

  research-tool:
    build: .
    depends_on:
      - searxng
    environment:
      RESEARCH_SEARXNG_HOST: "http://searxng:8080"
      RESEARCH_LLM_API_BASE: "http://172.17.0.1:8080/v1"
    ports:
      - "8000:8000"
```

### Full Stack

```yaml
version: "3.8"

services:
  searxng:
    image: searxng/searxng
    volumes:
      - ./searxng:/etc/searxng
    networks:
      - research

  llama:
    image: ghcr.io/ggerganov/llama.cpp:server-cuda
    runtime: nvidia
    command: >
      --model /models/tongyi-30b.gguf
      --ctx-size 32768
      --n-gpu-layers 99
      --host 0.0.0.0
    volumes:
      - /path/to/models:/models
    networks:
      - research

  research-tool:
    build: .
    depends_on:
      - searxng
      - llama
    environment:
      RESEARCH_SEARXNG_HOST: "http://searxng:8080"
      RESEARCH_LLM_API_BASE: "http://llama:8080/v1"
    ports:
      - "8000:8000"
    networks:
      - research

networks:
  research:
    driver: bridge
```

## Monitoring

### Health Check

```bash
curl http://localhost:8000/api/v1/health
```

### Docker Logs

```bash
docker compose logs -f research-tool
```

### Prometheus Metrics (Future)

Add to `src/main.py`:

```python
from prometheus_fastapi_instrumentator import Instrumentator

Instrumentator().instrument(app).expose(app)
```

Access metrics at `/metrics`.

## Troubleshooting

### Container won't start

```bash
# Check logs
docker compose logs research-tool

# Check config
docker compose config
```

### Can't reach LLM server

```bash
# Test from container
docker exec research-tool curl http://172.17.0.1:8080/v1/models

# Or use host.docker.internal
docker exec research-tool curl http://host.docker.internal:8080/v1/models
```

### SearXNG connection refused

```bash
# Test SearXNG
curl "http://192.168.1.3:8888/search?q=test&format=json"

# Check firewall
sudo ufw status
```

### Slow responses

- Check LLM inference speed
- Reduce `top_k` for fewer sources
- Use `reasoning_effort: "low"` for faster responses
- Consider vLLM with tensor parallelism for faster inference
