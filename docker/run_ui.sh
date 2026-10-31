#!/usr/bin/env bash
# Starts optional weight prefetch in the background, then keeps the UI running.
cd /opt/app || exit 1
mkdir -p /workspace/logs /workspace/outputs

if [ -n "${PREFETCH_MODELS:-}" ]; then
    # shellcheck disable=SC2086  # space-separated list of worker names
    /opt/venvs/ui/bin/python -m app.prefetch $PREFETCH_MODELS >> /workspace/logs/prefetch.log 2>&1 &
fi

if [ -z "${UI_PASSWORD:-}" ]; then
    echo "WARNING: UI_PASSWORD is not set; anyone with the pod's proxy URL can use the UI."
fi

while true; do
    echo "[$(date '+%F %T')] starting UI"
    /opt/venvs/ui/bin/python -m "${UI_MODULE:-app.ui}"
    echo "[$(date '+%F %T')] UI exited with status $?; restarting in 5 s"
    sleep 5
done
