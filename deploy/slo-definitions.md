# URCM Service Level Objectives (SLOs)

## Service Level Indicators (SLIs)

| SLI | Metric | Query | Description |
|-----|--------|-------|-------------|
| **Availability** | `urcm_health_ok` | `urcm_health_ok == 1` | System health endpoint returns OK |
| **Latency** | `urcm_reason_latency_seconds` | `histogram_quantile(0.99, sum(rate(urcm_reason_latency_seconds_bucket[5m])) by (le))` | P99 reasoning latency |
| **Convergence Rate** | `urcm_reason_counts_converged / urcm_process_total` | `rate(urcm_reason_counts_converged[5m]) / rate(urcm_process_total[5m])` | Fraction of queries that converge |
| **Max Steps Rate** | `urcm_max_steps_rate` | `urcm_max_steps_rate` | Fraction of queries hitting max steps |
| **Final Mu** | `urcm_last_final_mu` | `urcm_last_final_mu` | Final convergence value (0-1) |

## Service Level Objectives (SLOs)

| SLO | Target | Measurement Window | Error Budget |
|-----|--------|-------------------|--------------|
| **Availability** | ≥ 99.9% | 30 days | 43.2 min downtime/month |
| **Latency (P99)** | ≤ 50ms | 30 days | 0.1% of requests > 50ms |
| **Convergence Rate** | ≥ 85% | 30 days | 15% of queries may not converge |
| **Max Steps Rate** | ≤ 10% | 30 days | 10% of queries hitting max steps |
| **Final Mu** | ≥ 0.7 | 30 days | 30% of queries below threshold |

## Error Budget Policies

| Error Budget Remaining | Action |
|------------------------|--------|
| > 50% | Normal operations, deploy freely |
| 25-50% | Caution - limit risky deployments |
| 10-25% | Freeze non-critical deployments |
| < 10% | Emergency - only hotfixes allowed |
| 0% | All hands on deck - restore service |

## Alerting Rules

### Critical Alerts (page immediately)
- `URCMDown` - `urcm_health_ok == 0` for 2m
- `URCMLatencyHigh` - P99 latency > 200ms for 5m

### Warning Alerts (notify during business hours)
- `URCMLatencyElevated` - P99 latency > 50ms for 10m
- `URCMConvergenceLow` - Convergence rate < 80% for 15m
- `URCMMaxStepsRateHigh` - Max steps rate > 15% for 10m
- `URCMFinalMuLow` - Final Mu < 0.4 for 15m

## Burn Rate Alerts

| Burn Rate | Window | Alert |
|-----------|--------|-------|
| 14.4x (2% budget/hour) | 1h | Critical |
| 6x (1% budget/hour) | 6h | Warning |
| 2x (0.3% budget/hour) | 24h | Info |

## Dashboard Links
- **System Overview**: [URCM Overview Dashboard](../grafana/dashboards/urcm-overview.json)
- **Reasoning Performance**: [URCM Reasoning Dashboard](../grafana/dashboards/urcm-reasoning.json)

## Runbook Links
- [Runbook: High Latency](../runbooks/high-latency.md)
- [Runbook: Low Convergence](../runbooks/low-convergence.md)
- [Runbook: Service Down](../runbooks/service-down.md)
