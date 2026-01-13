#!/bin/bash

PORT=8000
TUNNEL_USER="REPLACE_WITH_USER"
TUNNEL_HOST="REPLACE_WITH_HOST"
TUNNEL_PASS="REPLACE_WITH_PASSWORD"
TUNNEL_REMOTE_PORT=8000
TUNNEL_LOCAL_PORT=8000
TUNNEL_SSH_PORT=REPLACE_WITH_PORT

# --- Check if port is in use (IPv4 + IPv6) ---
PIDS=$(netstat -tulnp 2>/dev/null | grep ":$PORT" | awk '{print $7}' | cut -d'/' -f1)
if [ -n "$PIDS" ]; then
    echo "[Tunnel] Port $PORT is already in use. Killing existing process(es): $PIDS"
    echo "$PIDS" | xargs kill -9
    sleep 2
fi

# --- Install sshpass (if not installed) ---
if ! command -v sshpass &> /dev/null; then
    echo "[Tunnel] Installing sshpass..."
    apt-get update && apt-get install -y sshpass
fi

# --- Check for existing tunnel before starting ---
EXISTING_TUNNEL=$(pgrep -f "ssh.*-L ${TUNNEL_LOCAL_PORT}:127.0.0.1:${TUNNEL_REMOTE_PORT}")
if [ -n "$EXISTING_TUNNEL" ]; then
    echo "[Tunnel] SSH tunnel already running with PID: $EXISTING_TUNNEL"
else
    echo "[Tunnel] Starting SSH Tunnel to Cloud..."
    sshpass -p "$TUNNEL_PASS" ssh -o StrictHostKeyChecking=no -CNgf \
        -L ${TUNNEL_LOCAL_PORT}:127.0.0.1:${TUNNEL_REMOTE_PORT} \
        -p $TUNNEL_SSH_PORT $TUNNEL_USER@$TUNNEL_HOST

    if [ $? -eq 0 ]; then
        echo "[Tunnel] Tunnel established in background!"
    else
        echo "[Tunnel] Failed to establish tunnel."
        exit 1
    fi
fi
