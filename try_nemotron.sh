#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

COMPOSE_FILE=docker-compose.cu130.yml
ENV_FILE=.env.nemotron
PORT=8000
HEALTH_URL="http://localhost:${PORT}/health"

# Load env
set -a; source "$ENV_FILE"; set +a

stop_vllm() {
    echo "Stopping vLLM..."
    docker compose -f "$COMPOSE_FILE" down 2>/dev/null || true
    sleep 3
}

wait_for_health() {
    local timeout=$1
    echo "  Waiting up to ${timeout}s for healthy..."
    local tries=0
    local max=$((timeout / 5))
    while ! curl -sf "$HEALTH_URL" > /dev/null 2>&1; do
        tries=$((tries + 1))
        if [ $tries -ge $max ]; then
            return 1
        fi
        sleep 5
    done
    echo "  Healthy!"
    return 0
}

smoke_test() {
    echo "  Running smoke test..."
    local resp
    resp=$(curl -sf "http://localhost:${PORT}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{\"model\": \"$MODEL_ID\", \"messages\": [{\"role\": \"user\", \"content\": \"Say hello.\"}], \"max_tokens\": 32}" 2>&1) || {
        echo "  Smoke test FAILED: request error"
        return 1
    }
    echo "  Smoke test response: $(echo "$resp" | python3 -c 'import sys,json; r=json.load(sys.stdin); print(r.get("choices",[{}])[0].get("message",{}).get("content","(empty)")[:100])' 2>/dev/null || echo "$resp" | head -c 200)"
    return 0
}

# === Attempt 1: FlashInfer ===
echo ""
echo "=========================================="
echo " Attempt 1: FlashInfer MoE FP4"
echo "=========================================="
stop_vllm

FLASHINFER_MOE_FP4=1 docker compose --env-file "$ENV_FILE" -f "$COMPOSE_FILE" up -d
echo "  Container started, waiting for model load..."

if wait_for_health 900 && smoke_test; then
    echo ""
    echo "SUCCESS: Nemotron running with FlashInfer backend"
    echo "  API: http://localhost:${PORT}"
    echo "  Model: $MODEL_ID"
    exit 0
fi

echo ""
echo "FlashInfer failed. Checking logs..."
docker compose -f "$COMPOSE_FILE" logs --tail=30
echo ""

# === Attempt 2: Marlin fallback ===
echo "=========================================="
echo " Attempt 2: Marlin fallback"
echo "=========================================="
stop_vllm

FLASHINFER_MOE_FP4=0 docker compose --env-file "$ENV_FILE" -f "$COMPOSE_FILE" up -d
echo "  Container started, waiting for model load..."

if wait_for_health 900 && smoke_test; then
    echo ""
    echo "SUCCESS: Nemotron running with Marlin backend"
    echo "  API: http://localhost:${PORT}"
    echo "  Model: $MODEL_ID"
    exit 0
fi

echo ""
echo "Both backends failed. Last logs:"
docker compose -f "$COMPOSE_FILE" logs --tail=50
stop_vllm
exit 1
