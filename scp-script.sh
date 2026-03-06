KEY=~/Desktop/Descriptron/segment-anything-2/gui/envs/hetzner-descriptron-portal
HOST=root@46.225.84.116
REMOTE_DIR=/root/descriptron-gbif-api

scp -i "$KEY" "$HOST:$REMOTE_DIR/app.py" .
scp -i "$KEY" "$HOST:$REMOTE_DIR/Dockerfile" .
scp -i "$KEY" "$HOST:$REMOTE_DIR/docker-compose.yml" .
scp -i "$KEY" "$HOST:$REMOTE_DIR/descriptron-gbif-annotator.html" .
scp -i "$KEY" "$HOST:$REMOTE_DIR/sam2-client.js" .
scp -i "$KEY" "$HOST:$REMOTE_DIR/setup-gbif-api.sh" .
scp -i "$KEY" -r "$HOST:$REMOTE_DIR/static" .

