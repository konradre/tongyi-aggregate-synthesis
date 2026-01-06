# Deployment Guide

## Quick Deployment

### 1. Set Environment Variables

```bash
export OPENROUTER_API_KEY="sk-or-v1-your-key-here"
export TAVILY_API_KEY="tvly-xxxxx"        # Optional
export LINKUP_API_KEY="xxxxx"              # Optional
```

### 2. Deploy with Docker Compose

```bash
docker compose up -d
```

The default `docker-compose.yml` is pre-configured for OpenRouter with:
- Model: `alibaba/tongyi-deepresearch-30b-a3b`
- Per-request API key support for multi-tenant deployments

### 3. Verify

```bash
curl http://localhost:8000/api/v1/health
```

## Docker Commands

```bash
# Start
docker compose up -d

# View logs
docker compose logs -f

# Rebuild after changes
docker compose up -d --build

# Stop
docker compose down
```

## VM Deployment

### Prerequisites

On target VM:
- Docker and Docker Compose installed
- Network access to SearXNG instance
- OpenRouter API key

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
export OPENROUTER_API_KEY="sk-or-v1-your-key"
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

        # Timeout for LLM responses
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

## Infrastructure Example

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
      RESEARCH_LLM_API_KEY: "${OPENROUTER_API_KEY}"
      RESEARCH_LLM_API_BASE: "https://openrouter.ai/api/v1"
      RESEARCH_LLM_MODEL: "alibaba/tongyi-deepresearch-30b-a3b"
    ports:
      - "8000:8000"
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

## Troubleshooting

### Container won't start

```bash
# Check logs
docker compose logs research-tool

# Check config
docker compose config
```

### OpenRouter errors

- **401 Unauthorized**: Check `RESEARCH_LLM_API_KEY` is set correctly
- **429 Rate Limit**: Check your OpenRouter credits or API key limits
- **502 Bad Gateway**: OpenRouter service issue, retry later

### SearXNG connection refused

```bash
# Test SearXNG
curl "http://192.168.1.3:8888/search?q=test&format=json"

# Check firewall
sudo ufw status
```

### Slow responses

- Reduce `top_k` for fewer sources
- Use `reasoning_effort: "low"` for faster responses
- Check OpenRouter status at https://status.openrouter.ai
