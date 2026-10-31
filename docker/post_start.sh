#!/usr/bin/env bash
# Run by the base image's /start.sh once SSH and Jupyter are up.
mkdir -p /workspace/logs
nohup /opt/app/run_ui.sh >> /workspace/logs/ui.log 2>&1 &
echo "UI starting on port 7860 (log: /workspace/logs/ui.log)"
