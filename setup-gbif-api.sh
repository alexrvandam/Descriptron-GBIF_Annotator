#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════
# Descriptron GBIF API — Isolated Setup Script
# ═══════════════════════════════════════════════════════════════════════
# 
# Run on Hetzner as root:
#   bash setup-gbif-api.sh
#
# This creates a COMPLETELY SEPARATE environment from your existing
# Descriptron Portal stack. Nothing shared except the nginx reverse proxy.
#
# Structure:
#   /root/descriptron-gbif-api/
#   ├── app.py                 ← FastAPI SAM2.1 backend
#   ├── Dockerfile             ← Isolated Python 3.11 + SAM2.1-tiny
#   ├── docker-compose.yml     ← Separate container, network, volumes
#   └── static/
#       └── descriptron-gbif-annotator.html  ← The web UI
#
# Exposed via nginx:
#   https://YOUR_IP/gbif/              → static HTML annotator
#   https://YOUR_IP/gbif-api/          → SAM2 API endpoints
#   https://YOUR_IP/gbif-api/docs      → Swagger API docs

set -euo pipefail

GBIF_DIR="/root/descriptron-gbif-api"
NGINX_CONF="/root/descriptron-guacamole/nginx/nginx.conf"

echo "═══════════════════════════════════════════════════════════"
echo " Descriptron GBIF API — Isolated Setup"
echo "═══════════════════════════════════════════════════════════"

# ─── 1. Create isolated directory ─────────────────────────────────────
echo ""
echo "[1/5] Creating isolated directory: $GBIF_DIR"
mkdir -p "$GBIF_DIR/static"

# Check that files exist (you should have copied them here already)
if [ ! -f "$GBIF_DIR/app.py" ]; then
    echo "  ⚠ app.py not found in $GBIF_DIR — copy it there first!"
    echo "  Files needed: app.py, Dockerfile, docker-compose.yml"
    echo "  And the HTML file in static/"
    exit 1
fi

echo "  ✓ Directory ready"

# ─── 2. Update nginx config ──────────────────────────────────────────
echo ""
echo "[2/5] Updating nginx configuration..."

# Check if gbif-api block already exists
if grep -q "gbif-api" "$NGINX_CONF" 2>/dev/null; then
    echo "  ✓ nginx already has gbif-api config (skipping)"
else
    echo "  Adding /gbif/ and /gbif-api/ location blocks to nginx..."
    
    # We need to add the location blocks inside the server block.
    # Find the line with the first 'location' directive and insert before it.
    # This is the safest approach.
    
    # Create a temp file with the new blocks
    NGINX_SNIPPET=$(cat <<'NGINX_EOF'

        # ── Descriptron GBIF Web Annotator (ISOLATED) ────────────────
        # Static HTML annotator
        location /gbif/ {
            alias /root/descriptron-gbif-api/static/;
            index descriptron-gbif-annotator.html;
            add_header Cache-Control "no-cache";
        }

        # SAM2 API backend (proxied to isolated container)
        location /gbif-api/ {
            proxy_pass http://127.0.0.1:8100/;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            proxy_read_timeout 120s;  # Encoding can take ~30s on CPU
            proxy_send_timeout 120s;
            client_max_body_size 50M;
            
            # CORS headers
            add_header Access-Control-Allow-Origin "*" always;
            add_header Access-Control-Allow-Methods "GET, POST, DELETE, OPTIONS" always;
            add_header Access-Control-Allow-Headers "Content-Type" always;
            
            if ($request_method = 'OPTIONS') {
                return 204;
            }
        }
        # ── End GBIF block ───────────────────────────────────────────

NGINX_EOF
    )
    
    # Backup nginx config
    cp "$NGINX_CONF" "${NGINX_CONF}.bak.$(date +%Y%m%d%H%M%S)"
    
    # Insert before the first 'location' line in the server block
    # Find line number of first "location /" that's NOT our new block
    FIRST_LOC_LINE=$(grep -n "location " "$NGINX_CONF" | head -1 | cut -d: -f1)
    
    if [ -n "$FIRST_LOC_LINE" ]; then
        # Insert before the first location block
        head -n $((FIRST_LOC_LINE - 1)) "$NGINX_CONF" > "${NGINX_CONF}.new"
        echo "$NGINX_SNIPPET" >> "${NGINX_CONF}.new"
        tail -n +"$FIRST_LOC_LINE" "$NGINX_CONF" >> "${NGINX_CONF}.new"
        mv "${NGINX_CONF}.new" "$NGINX_CONF"
        echo "  ✓ nginx config updated"
    else
        echo "  ⚠ Could not find location block in nginx.conf"
        echo "  You'll need to manually add the /gbif/ and /gbif-api/ blocks."
        echo "  See the snippet above."
    fi
fi

# ─── 3. Build the isolated container ─────────────────────────────────
echo ""
echo "[3/5] Building GBIF SAM2 container (this may take 5-10 minutes)..."
cd "$GBIF_DIR"
docker compose build

# ─── 4. Start the service ────────────────────────────────────────────
echo ""
echo "[4/5] Starting GBIF SAM2 API..."
docker compose up -d

# ─── 5. Reload nginx ─────────────────────────────────────────────────
echo ""
echo "[5/5] Reloading nginx..."
cd /root/descriptron-guacamole
docker compose exec -T nginx nginx -t && docker compose exec -T nginx nginx -s reload
echo "  ✓ nginx reloaded"

# ─── Status ───────────────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════════"
echo " Setup Complete!"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo " Web Annotator:  https://$(hostname -I | awk '{print $1}')/gbif/"
echo " API Docs:       https://$(hostname -I | awk '{print $1}')/gbif-api/docs"
echo " Health Check:   https://$(hostname -I | awk '{print $1}')/gbif-api/health"
echo ""
echo " Container status:"
docker compose ps
echo ""
echo " Logs:"
echo "   docker compose -f $GBIF_DIR/docker-compose.yml logs -f gbif-sam2"
echo ""
echo " ⚠ First image encoding will take ~15-30s on CPU (SAM2.1-tiny)."
echo "   Subsequent prompts on the same image are near-instant (~50ms)."
echo ""
echo " The main Descriptron Portal is completely untouched."
echo "═══════════════════════════════════════════════════════════"
