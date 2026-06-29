# URCM High Latency - Runbook

## Problem Statement

High latency (>50ms P99) in URCM reasoning queries impacting user experience and service level objectives.

## Symptoms

### Observable Signs
- **P99 latency > 50ms** for reasoning queries
- **Convergence rate drop** due to timeouts
- **User complaints** about slow responses
- **Error rates** increase as clients retry

### Root Causes
1. **Memory pressure** - Concept map too large for available RAM
2. **CPU bottlenecks** - Inefficient algorithms or resource contention
3. **Network latency** - External dependencies or load balancer issues
4. **Algorithm complexity** - Poor scaling with query complexity
5. **Resource contention** - Shared resources overwhelmed

## Diagnosis

### Step 1: Baseline Measurement
```bash
# Current latency metrics
kubectl exec -it <pod-name> -- curl -s http://localhost:8008/metrics | grep urcm_reason_latency

# P99 calculation
curl http://prometheus-server:9090/api/v1/query?query=histogram_quantile(0.99, sum(rate(urcm_reason_latency_seconds_bucket[5m])) by (le))

# Distribution
curl http://prometheus-server:9090/api/v1/query_range?query=urcm_reason_latency_seconds&start=$(date -d '5 minutes ago' --iso-8601)&end=$(date --iso-8601)&step=15s
```

### Step 2: Component Analysis
```bash
# Check individual components
kubectl top pods -l app=urcm

# Resource usage trends
kubectl top pods -l app=urcm --watch

# Application logs for slow queries
kubectl logs -l app=urcm --since=5m | grep "Reasoning failed"
```

### Step 3: Pattern Identification
- **Memory-related**: High RSS, frequent GC pauses
- **CPU-related**: High CPU utilization, context switches
- **I/O-related**: Disk I/O waits, network latency
- **Algorithm-related**: Query complexity, nested loops

## Immediate Actions (First 30 minutes)

### 1. Scale Resources
```bash
# Scale up immediately
kubectl scale deployment urcm-api --replicas=4

# Monitor recovery
kubectl get hpa urcm-blue-hpa
```

### 2. Implement Circuit Breaker
```bash
# Temporarily disable problematic queries
kubectl patch deployment urcm-api -p '{"spec":{"template":{"spec":{"containers":[{"name":"api","env":[{"name":"URCM_CIRCUIT_BREAKER","value":"enabled"}]}]}}}}'
```

### 3. Log Analysis
```bash
# Identify slow queries
kubectl logs -l app=urcm --since=10m | grep "latency"

# Check for patterns
kubectl logs -l app=urcm --since=10m | grep -E "(timeout|failed|slow)"
```

## Long-term Solutions

### 1. Algorithm Optimization
- **Caching**: Implement query result caching
- **Indexing**: Optimize concept map access patterns
- **Parallelization**: Parallel processing for independent queries
- **Early termination**: Cut off long-running queries

### 2. Resource Management
- **Memory**: Implement memory-efficient data structures
- **CPU**: Optimize critical paths, use async processing
- **Network**: Implement connection pooling, CDN for static assets

### 3. Architecture Changes
- **Horizontal scaling**: Auto-scaling based on latency
- **Load balancing**: Distribute queries across multiple instances
- **Geographic distribution**: Edge deployment for reduced latency

## Monitoring Setup

### Alerting Rules
```yaml
# high_latency.yaml
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: urcm-high-latency
  labels:
    severity: warning
spec:
  groups:
  - name: urcm
    rules:
    - alert: URCMHighLatency
      expr: histogram_quantile(0.99, sum(rate(urcm_reason_latency_seconds_bucket[5m])) by (le)) > 50
      for: 5m
      labels:
        severity: critical
      annotations:
        summary: "URCM P99 latency exceeded 50ms"
        description: "P99 latency is {{ $value }}ms, threshold is 50ms"
        runbook: "https://runbooks.company.com/high-latency"
```

### Dashboard Configuration
```json
{
  "title": "URCM Latency Monitoring",
  "panels": [
    {
      "title": "P99 Latency",
      "type": "stat",
      "targets": [
        {
          "expr": "histogram_quantile(0.99, sum(rate(urcm_reason_latency_seconds_bucket[5m])) by (le))",
          "legendFormat": "P99"
        }
      ]
    },
    {
      "title": "Latency Distribution",
      "type": "heatmap",
      "targets": [
        {
          "expr": "sum(rate(urcm_reason_latency_seconds_bucket[5m])) by (le)",
          "format": "heatmap"
        }
      ]
    }
  ]
}
```

## Prevention Measures

### 1. Capacity Planning
- **Monitor**: Set up alerts at 70% of limits
- **Forecast**: Predict growth based on usage patterns
- **Provision**: Plan for 2x expected peak load

### 2. Performance Testing
- **Baseline**: Establish performance benchmarks
- **Regression testing**: Monitor for performance degradation
- **Load testing**: Simulate production traffic

### 3. Architecture Review
- **Review**: Monthly architecture reviews
- **Update**: Update performance models
- **Optimize**: Continuously optimize critical paths

## Post-Mortem

### Immediate (within 1 hour)
- Document what happened
- Identify root cause
- Note actions taken

### Short-term (within 24 hours)
- Analyze response time
- Identify improvement opportunities
- Update runbook if needed

### Long-term (within 1 week)
- Review system architecture
- Plan preventive measures
- Update monitoring and alerting

---

## Runbook Maintenance

### Review Frequency
- **Critical Runbooks**: Monthly
- **Standard Runbooks**: Quarterly
- **Emergency Procedures**: After each incident

### Update Process
1. Document new failure patterns
2. Add recovery steps
3. Update contact information
4. Test procedures
5. Review with team

---

## Emergency Contacts

### Immediate Response
- **Platform Team**: platform@company.com, +1-555-0123
- **Engineering Lead**: eng-lead@company.com, +1-555-0456

### 24/7 On-Call
- **Primary**: On-call rotation via PagerDuty
- **Secondary**: Backup on-call via email

---

*This runbook is maintained by the URCM Operations Team.*
*Last updated: $(date +%Y-%m-%d)*
*Version: 1.0*