#!/usr/bin/env bash
# =====================================================================
# Install the agent as a systemd service on a Linux VPS.
# Usage:  sudo bash scripts/install_systemd.sh
# =====================================================================
set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SERVICE_FILE="/etc/systemd/system/pumpfun-agent.service"

if [[ $EUID -ne 0 ]]; then
  echo "Must run as root (use sudo)"; exit 1
fi

# Detect the user that owns the project dir
RUN_USER="$(stat -c '%U' "$PROJECT_DIR")"
echo "Installing service for user: $RUN_USER"
echo "Project dir: $PROJECT_DIR"

# Make sure data dir exists
sudo -u "$RUN_USER" mkdir -p "$PROJECT_DIR/data"

cat > "$SERVICE_FILE" <<EOF
[Unit]
Description=Pump.fun Trading Agent
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=$RUN_USER
WorkingDirectory=$PROJECT_DIR
EnvironmentFile=$PROJECT_DIR/.env
ExecStart=$PROJECT_DIR/.venv/bin/python orchestrator.py
Restart=on-failure
RestartSec=10
StandardOutput=journal
StandardError=journal

# Hardening
NoNewPrivileges=true
ProtectSystem=full
ProtectHome=read-only
PrivateTmp=true

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable pumpfun-agent
echo
echo "Installed. Start with:  sudo systemctl start pumpfun-agent"
echo "Tail logs with:         sudo journalctl -u pumpfun-agent -f"
echo "Stop with:              sudo systemctl stop pumpfun-agent"
