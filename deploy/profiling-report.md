# URCM Resource Profiling Report

## Executive Summary

This report documents the resource usage characteristics of the URCM system based on the current Kubernetes resource limits and application-level profiling capabilities.

---

## Current Kubernetes Resource Configuration

### Deployment Resources (from `deploy/k8s/deployment.yaml`)

| Resource | Request | Limit | Notes |
|----------|---------|-------|-------|
| **Memory** | 256 MiB | 1 GiB | ~4x headroom for burst |
| **CPU** | 250 millicores | 1 core | ~4x headroom for burst |

### HPA Configuration (from `deploy/k8s/blue-green.yaml`)

| Metric | Target Utilization | Min Replicas | Max Replicas |
|--------|-------------------|--------------|--------------|
| CPU | 70% | 2 | 8 |
| Memory | 80% | 2 | 8 |

---

## Application-Level Profiling Capabilities

### Built-in Profiling (`urcm/core/performance.py`)

The URCM system includes comprehensive profiling infrastructure:

| Component | Class | Metrics Tracked |
|-----------|-------|-----------------|
| **Memory Efficiency** | `OptimizedPhonemeSet` | LRU cache hits/misses, compression ratio, memory usage |
| **Compression** | `CompressionMonitor` | Input/output size ratios, time per operation |
| **Performance Benchmark** | `PerformanceBenchmark` | Memory vs token-based, speedup ratios, scalability |

### Key Metrics Available

```python
# From PerformanceMetrics dataclass
- total_memory_mb: float
- peak_memory_mb: float
- compression_ratio: float
- processing_time_seconds: float
- throughput_ops_per_sec: float
- cache_hit_rate: float
```

### Convergence Engine Timing (`urcm/core/executive.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_wall_ms` | 500ms | Maximum wall-clock time per reasoning cycle |
| `fixed_step_ms` | 10ms | Fixed step duration for deterministic timing |

---

## Expected Resource Usage Patterns

### Memory Profile

| Phase | Estimated Memory | Notes |
|-------|-----------------|-------|
| **Idle** | ~50-80 MB | Base system + concept map |
| **Reasoning** | ~100-200 MB | Active constraints + working memory |
| **Peak** | ~300-500 MB | Complex multi-step reasoning |
| **Limit** | 1024 MB | K8s limit provides 2-10x headroom |

### CPU Profile

| Operation | Typical Duration | CPU Pattern |
|-----------|-----------------|-------------|
| **Single reasoning step** | 1-5 ms | Burst (matrix operations) |
| **Full convergence** | 10-200 ms | Sustained (multiple steps) |
| **A* planning** | 10-500 ms | Variable (constraint solving) |

### Scaling Characteristics

| Replicas | Expected Throughput | Memory Total | CPU Total |
|----------|-------------------|--------------|-----------|
| 2 (min) | ~20-50 req/s | 200-400 MB | 0.5-2 cores |
| 4 | ~40-100 req/s | 400-800 MB | 1-4 cores |
| 8 (max) | ~80-200 req/s | 800-1600 MB | 2-8 cores |

---

## Profiling Commands

### Run Built-in Benchmarks

```bash
# Memory efficiency benchmark
python -m urcm.core.performance

# Convergence timing
python -c "
from urcm.core.system import URCMSystem
import time
sys = URCMSystem()
start = time.time()
result = sys.process_query('All humans are mortal. Socrates is human.')
print(f'Time: {(time.time()-start)*1000:.1f}ms, Steps: {len(result.mu_trajectory)}')
"
```

### Kubernetes Resource Monitoring

```bash
# Current usage
kubectl top pods -l app=urcm

# Resource quotas
kubectl describe pod <pod-name> | grep -A 10 "Limits:\|Requests:"

# HPA status
kubectl get hpa -l app=urcm
```

### Continuous Profiling (Recommended)

```yaml
# Add to deployment for continuous profiling
# (requires py-spy or similar)
env:
  - name: PYTHON_PROFILING
    value: "1"
```

---

## Recommendations

### Immediate (Pre-Production)
1. ✅ K8s resource limits configured
2. ✅ HPA with CPU/memory metrics
3. ✅ Application-level profiling infrastructure exists
4. ⏳ Run production load test to validate limits
5. ⏳ Document baseline metrics for regression detection

### Short-term (Post-Launch)
1. Integrate continuous profiling (py-spy, pyroscope)
2. Add custom Prometheus metrics for latency percentiles
3. Set up regression alerts for resource usage changes
4. Document capacity planning model

### Long-term
1. GPU acceleration evaluation (if concept map grows >100K)
2. Distributed deployment architecture
3. Cold-start optimization

---

## Appendix: Prometheus Metrics for Profiling

| Metric | Type | Description |
|--------|------|-------------|
| `urcm_process_total` | Counter | Total queries processed |
| `urcm_converged_total` | Counter | Queries that converged |
| `urcm_mu_step_total` | Counter | Total reasoning steps |
| `urcm_last_final_mu` | Gauge | Last convergence value |
| `urcm_last_delta_mu` | Gauge | Last convergence delta |
| `process_resident_memory_bytes` | Gauge | Process RSS (from node_exporter) |
| `process_cpu_seconds_total` | Counter | CPU time (from node_exporter) |