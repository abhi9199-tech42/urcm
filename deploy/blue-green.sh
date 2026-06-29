#!/usr/bin/env bash
# Blue-Green deployment switch for URCM
# Usage:
#   ./deploy-blue-green.sh green   # Switch traffic to green (deploy new version)
#   ./deploy-blue-green.sh blue    # Rollback to blue
#   ./deploy-blue-green.sh status  # Show current active slot

set -euo pipefail

NAMESPACE="${URCM_NAMESPACE:-default}"
SERVICE="urcm-api"

active_slot() {
    kubectl get svc "$SERVICE" -n "$NAMESPACE" -o jsonpath='{.spec.selector.slot}' 2>/dev/null || echo "unknown"
}

switch_to() {
    local target="$1"
    local current
    current=$(active_slot)

    if [ "$current" = "$target" ]; then
        echo "Already on $target, nothing to do."
        exit 0
    fi

    echo "Switching $SERVICE from $current to $target..."

    # Update service selector
    kubectl patch svc "$SERVICE" -n "$NAMESPACE" -p "{\"spec\":{\"selector\":{\"slot\":\"$target\"}}}"

    # Verify health of new target
    echo "Waiting for $target pods to be ready..."
    kubectl rollout status deployment/urcm-$target -n "$NAMESPACE" --timeout=120s

    # Quick health check via port-forward (background)
    echo "Running health check..."
    local pod
    pod=$(kubectl get pod -n "$NAMESPACE" -l "app=urcm,slot=$target" -o jsonpath='{.items[0].metadata.name}')
    kubectl exec -n "$NAMESPACE" "$pod" -- python -c "
import urllib.request, json, sys
try:
    r = urllib.request.urlopen('http://localhost:8008/health', timeout=5)
    data = json.loads(r.read())
    if data.get('ok'):
        print('Health check: OK')
        sys.exit(0)
    else:
        print('Health check: DEGRADED')
        sys.exit(1)
except Exception as e:
    print(f'Health check failed: {e}')
    sys.exit(1)
" || {
        echo "Health check failed! Rolling back to $current..."
        kubectl patch svc "$SERVICE" -n "$NAMESPACE" -p "{\"spec\":{\"selector\":{\"slot\":\"$current\"}}}"
        echo "Rolled back to $current"
        exit 1
    }

    echo "Switched to $target successfully."
    echo "Previous slot: $current (available for rollback)"
}

scale_slot() {
    local slot="$1"
    local replicas="$2"
    kubectl scale deployment "urcm-$slot" -n "$NAMESPACE" --replicas="$replicas"
}

show_status() {
    echo "=== URCM Blue-Green Deployment Status ==="
    echo ""
    echo "Active slot: $(active_slot)"
    echo ""
    echo "Deployments:"
    kubectl get deployments -n "$NAMESPACE" -l app=urcm -o custom-columns=\
'NAME:.metadata.name,REPLICAS:.spec.replicas,READY:.status.readyReplicas,UPDATED:.status.updatedReplicas,SLOT:.metadata.labels.slot'
    echo ""
    echo "Service selector:"
    kubectl get svc "$SERVICE" -n "$NAMESPACE" -o jsonpath='{.spec.selector}' | python -m json.tool
    echo ""
}

usage() {
    echo "Usage: $0 {green|blue|status|scale-blue|scale-green}"
    echo ""
    echo "Commands:"
    echo "  green       Switch traffic to green slot"
    echo "  blue        Rollback traffic to blue slot"
    echo "  status      Show current deployment status"
    echo "  scale-blue N    Scale blue deployment to N replicas"
    echo "  scale-green N   Scale green deployment to N replicas"
}

case "${1:-}" in
    green|blue)
        switch_to "$1"
        ;;
    status)
        show_status
        ;;
    scale-blue)
        scale_slot "blue" "${2:-2}"
        ;;
    scale-green)
        scale_slot "green" "${2:-2}"
        ;;
    *)
        usage
        exit 1
        ;;
esac
