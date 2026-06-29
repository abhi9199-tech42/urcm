# URCM Service Down - Disaster Recovery Runbook

## Emergency Response Procedures

### Immediate Actions (First 5 Minutes)

#### 1. Verify Service Status
```bash
# Check pod status
kubectl get pods -l app=urcm

# Check service
kubectl get svc urcm-api

# Check endpoints
kubectl describe endpoints urcm-api
```

#### 2. Identify Root Cause
```bash
# Check pod logs
kubectl logs -l app=urcm --tail=100

# Check events
kubectl get events --field-selector involvedObject.kind=Pod

# Check resource usage
kubectl top pods -l app=urcm
```

#### 3. Basic Troubleshooting
- **Memory Issues**: Check if pod is OOMKilled
- **CPU Issues**: Check if pod is throttled
- **Network Issues**: Check if pod can't reach dependencies
- **Application Errors**: Look for exceptions in logs

### Escalation Path

| Symptom | First Responder | Time Limit | Escalation |
|---------|-----------------|------------|------------|
| Single pod failure | Service Owner | 15 min | Platform Team |
| Multiple pods failure | Service Owner | 5 min | Platform Team |
| API endpoint down | Service Owner | 2 min | Platform Team |
| Cluster-wide issues | Platform Team | N/A | Engineering Lead |

### Recovery Procedures

#### Pod Restart
```bash
# Restart failed pod
kubectl delete pod <failed-pod-name>

# Watch for recovery
kubectl logs -f <new-pod-name> --tail=50
```

#### Scale Up
```bash
# Scale deployment to 3 replicas
kubectl scale deployment urcm-api --replicas=3

# Wait for new pods
kubectl get pods -l app=urcm --watch
```

#### Resource Limits Adjustment
```bash
# Temporarily increase limits
kubectl patch deployment urcm-api -p '{"spec":{"template":{"spec":{"containers":[{"name":"api","resources":{"limits":{"memory":"2Gi"}}}]}}}}'
```

### Post-Recovery Actions

1. **Run Health Checks**
```bash
# Full system validation
kubectl exec -it <pod-name> -- curl -s http://localhost:8008/api/validate

# Metrics check
kubectl exec -it <pod-name> -- curl -s http://localhost:8008/metrics
```

2. **Monitor Recovery**
```bash
# Watch for 5 minutes
kubectl logs -f <pod-name> --since=5m
```

3. **Document Incident**
- Create incident ticket
- Document root cause
- Update runbook if needed

---

## Common Failure Scenarios

### Scenario 1: Memory Exhaustion

**Symptoms:**
- Pod OOMKilled
- Memory usage at limit
- High GC pressure

**Recovery:**
1. Increase memory limits
2. Optimize concept map size
3. Implement memory-efficient data structures

**Prevention:**
- Monitor memory usage trends
- Set alerts at 80% of limits
- Regular profiling

### Scenario 2: CPU Saturation

**Symptoms:**
- High CPU utilization (>90%)
- Pending pods
- Slow response times

**Recovery:**
1. Scale up replicas
2. Optimize algorithms
3. Check for infinite loops

**Prevention:**
- Set CPU limits appropriately
- Monitor query patterns
- Implement circuit breakers

### Scenario 3: Network Partition

**Symptoms:**
- API endpoints unreachable
- Health checks fail
- Timeouts

**Recovery:**
1. Check network policies
2. Verify DNS resolution
3. Restart affected pods

**Prevention:**
- Configure proper network policies
- Set up health check probes
- Implement retry logic

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

### Version Control
- Store runbooks in Git
- Use semantic versioning
- Document changes with incident context

---

## Contact Information

### Emergency Contacts
- **Platform Team**: platform@company.com, +1-555-0123
- **Engineering Lead**: eng-lead@company.com, +1-555-0456
- **On-Call**: Rotate via PagerDuty

### Internal Documentation
- Confluence: URCM Operations
- Slack: #urcm-incidents
- Teams: URCM Support Channel

---

## Post-Incident Review

After any service disruption:

1. **Immediate (within 1 hour)**
   - Document what happened
   - Identify root cause
   - Note actions taken

2. **Short-term (within 24 hours)**
   - Analyze response time
   - Identify improvement opportunities
   - Update runbook if needed

3. **Long-term (within 1 week)**
   - Review system architecture
   - Plan preventive measures
   - Update monitoring and alerting

---

## Checklist for Service Restoration

### Pre-Restoration
- [ ] Verify backup systems are healthy
- [ ] Check monitoring systems
- [ ] Confirm resource availability
- [ ] Update stakeholder notifications

### During Restoration
- [ ] Monitor recovery progress
- [ ] Check for cascading failures
- [ ] Verify data consistency
- [ ] Validate performance targets

### Post-Restoration
- [ ] Run full health checks
- [ ] Validate all endpoints
- [ ] Monitor for 30 minutes
- [ ] Document lessons learned
- [ ] Update runbook if needed

---

## Emergency Contact Scripts

### Quick Health Check
```bash
#!/bin/bash
# health-check.sh

POD_NAME=$(kubectl get pods -l app=urcm -o jsonpath='{.items[0].metadata.name}')

if [ -z "$POD_NAME" ]; then
    echo "No URCM pods found"
    exit 1
fi

# Check health endpoint
if kubectl exec -it $POD_NAME -- curl -s http://localhost:8008/health | grep -q '"ok": true'; then
    echo "URCM service is healthy"
else
    echo "URCM service is unhealthy"
    exit 1
fi
```

### Emergency Restart
```bash
#!/bin/bash
# emergency-restart.sh

echo "Starting emergency restart of URCM service..."

# Scale to 0 temporarily
kubectl scale deployment urcm-api --replicas=0

# Wait for pods to terminate
sleep 30

# Scale back to 2
kubectl scale deployment urcm-api --replicas=2

# Wait for recovery
sleep 60

# Verify health
POD_NAME=$(kubectl get pods -l app=urcm -o jsonpath='{.items[0].metadata.name}')
if kubectl exec -it $POD_NAME -- curl -s http://localhost:8008/health | grep -q '"ok": true'; then
    echo "Service recovered successfully"
else
    echo "Service recovery failed"
    exit 1
fi
```

---

## Appendix: Useful Commands

### Pod Management
```bash
# List all URCM pods
kubectl get pods -l app=urcm

# Get pod logs
kubectl logs <pod-name>

# Execute into pod
kubectl exec -it <pod-name> -- /bin/sh

# Describe pod
kubectl describe pod <pod-name>

# Get pod events
kubectl get events --field-selector involvedObject.name=<pod-name>
```

### Resource Management
```bash
# Check resource usage
kubectl top pods -l app=urcm

# Describe resource limits
kubectl describe deployment urcm-api

# Update resource limits
kubectl patch deployment urcm-api -p '{"spec":{"template":{"spec":{"containers":[{"name":"api","resources":{"limits":{"memory":"2Gi","cpu":"2"}}}]}}}}'
```

### Monitoring
```bash
# Prometheus query examples
# System health
curl http://prometheus-server:9090/api/v1/query?query=urcm_health_ok

# Latency
curl http://prometheus-server:9090/api/v1/query?query=histogram_quantile(0.99, sum(rate(urcm_reason_latency_seconds_bucket[5m])) by (le))

# Convergence rate
curl http://prometheus-server:9090/api/v1/query?query=rate(urcm_reason_counts_converged[5m]) / rate(urcm_process_total[5m])
```

---

*This runbook is maintained by the URCM Operations Team.*
*Last updated: $(date +%Y-%m-%d)*
*Version: 1.0*