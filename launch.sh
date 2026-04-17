#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
#  RoboScout Mission Control — Launch Script
#  Usage: ./launch.sh [--sim | --ros]
# ─────────────────────────────────────────────────────────────
set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
MODE=${1:---sim}

echo ""
echo "  ██████╗  ██████╗ ██████╗  ██████╗ ███████╗ ██████╗ ██████╗ ██╗   ██╗████████╗"
echo "  ██╔══██╗██╔═══██╗██╔══██╗██╔═══██╗██╔════╝██╔════╝██╔═══██╗██║   ██║╚══██╔══╝"
echo "  ██████╔╝██║   ██║██████╔╝██║   ██║███████╗██║     ██║   ██║██║   ██║   ██║   "
echo "  ██╔══██╗██║   ██║██╔══██╗██║   ██║╚════██║██║     ██║   ██║██║   ██║   ██║   "
echo "  ██║  ██║╚██████╔╝██████╔╝╚██████╔╝███████║╚██████╗╚██████╔╝╚██████╔╝   ██║   "
echo "  ╚═╝  ╚═╝ ╚═════╝ ╚═════╝  ╚═════╝ ╚══════╝ ╚═════╝ ╚═════╝ ╚═════╝    ╚═╝   "
echo ""
echo "  Mission Control v1.0 | Chennai, India | $(date)"
echo "  Mode: $MODE"
echo ""

# Install Python deps if needed
if ! python3 -c "import flask" 2>/dev/null; then
  echo "[*] Installing Python dependencies..."
  pip3 install -r "$DIR/requirements.txt" --quiet
fi

# Source ROS if available
if [ "$MODE" = "--ros" ]; then
  if [ -f /opt/ros/noetic/setup.bash ]; then
    source /opt/ros/noetic/setup.bash
    echo "[*] ROS Noetic sourced"
  elif [ -f /opt/ros/humble/setup.bash ]; then
    source /opt/ros/humble/setup.bash
    echo "[*] ROS2 Humble sourced"
  fi
fi

echo "[*] Starting Flask backend on http://0.0.0.0:5000"
cd "$DIR"
python3 app.py
