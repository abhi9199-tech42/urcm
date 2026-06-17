import os
import json
from urllib.parse import urlparse, parse_qs
from http.server import BaseHTTPRequestHandler, HTTPServer

def aggregate(log_dir, env=None):
    path = os.path.join(log_dir, "logs", "metrics.jsonl")
    counts = {
        "process_total": 0,
        "converged_total": 0,
        "dead_end_total": 0,
        "max_steps_total": 0,
        "oscillation_total": 0,
        "mu_step_total": 0,
        "reason_counts": {"converged": 0, "dead_end": 0, "max_steps": 0}
    }
    gauges = {
        "last_final_mu": 0.0,
        "last_delta_mu": 0.0,
    }
    if not os.path.exists(path):
        return counts, gauges
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    e = json.loads(line)
                except Exception:
                    continue
                if env and str(e.get("env", "")).lower() != str(env).lower():
                    continue
                ev = e.get("event", "")
                if ev == "process_end":
                    counts["process_total"] += 1
                    v = e.get("final_mu")
                    if isinstance(v, (int, float)):
                        gauges["last_final_mu"] = float(v)
                elif ev == "mu_path_complete":
                    if bool(e.get("converged")):
                        counts["converged_total"] += 1
                        counts["reason_counts"]["converged"] += 1
                    else:
                        reason = str(e.get("reason", "")).lower()
                        if "max steps" in reason:
                            counts["max_steps_total"] += 1
                            counts["reason_counts"]["max_steps"] += 1
                        elif "dead end" in reason:
                            counts["dead_end_total"] += 1
                            counts["reason_counts"]["dead_end"] += 1
                elif ev == "mu_oscillation_detected":
                    counts["oscillation_total"] += 1
                elif ev == "mu_step":
                    counts["mu_step_total"] += 1
                    v = e.get("delta_mu")
                    if isinstance(v, (int, float)):
                        gauges["last_delta_mu"] = float(v)
    except Exception:
        pass
    return counts, gauges

def render_metrics(log_dir, env=None):
    counts, gauges = aggregate(log_dir, env=env)
    # Prometheus exposition with HELP/TYPE and labels
    lines = []
    # HELP/TYPE
    lines.append("# HELP urcm_process_total Total number of completed processes.")
    lines.append("# TYPE urcm_process_total counter")
    lines.append("# HELP urcm_mu_path_complete_total Count of μ path completions by reason.")
    lines.append("# TYPE urcm_mu_path_complete_total counter")
    lines.append("# HELP urcm_oscillation_total Count of oscillation detections.")
    lines.append("# TYPE urcm_oscillation_total counter")
    lines.append("# HELP urcm_mu_step_total Count of μ step updates.")
    lines.append("# TYPE urcm_mu_step_total counter")
    lines.append("# HELP urcm_last_final_mu Last observed final μ value.")
    lines.append("# TYPE urcm_last_final_mu gauge")
    lines.append("# HELP urcm_last_delta_mu Last observed Δμ value.")
    lines.append("# TYPE urcm_last_delta_mu gauge")
    # Counters with labels
    env_lbl = str(env) if env else ""
    lines.append(f'urcm_process_total{{env="{env_lbl}"}} {int(counts["process_total"])}')
    for reason, val in counts["reason_counts"].items():
        lines.append(f'urcm_mu_path_complete_total{{reason="{reason}",env="{env_lbl}"}} {int(val)}')
    lines.append(f'urcm_oscillation_total{{env="{env_lbl}"}} {int(counts["oscillation_total"])}')
    lines.append(f'urcm_mu_step_total{{env="{env_lbl}"}} {int(counts["mu_step_total"])}')
    # Gauges
    lines.append(f'urcm_last_final_mu{{env="{env_lbl}"}} {float(gauges["last_final_mu"])}')
    lines.append(f'urcm_last_delta_mu{{env="{env_lbl}"}} {float(gauges["last_delta_mu"])}')
    return "\n".join(lines) + "\n"

def compute_health(log_dir, env=None):
    counts, gauges = aggregate(log_dir, env=env)
    # SLO defaults (override with env vars)
    min_final_mu = float(os.environ.get("URCM_SLO_MIN_FINAL_MU", "0.4"))
    max_steps_rate_max = float(os.environ.get("URCM_SLO_MAX_STEPS_RATE", "0.5"))
    total_completions = sum(counts["reason_counts"].values())
    max_steps_rate = (counts["reason_counts"]["max_steps"] / total_completions) if total_completions > 0 else 0.0
    ok = (gauges["last_final_mu"] >= min_final_mu) and (max_steps_rate <= max_steps_rate_max)
    return {
        "ok": bool(ok),
        "last_final_mu": gauges["last_final_mu"],
        "max_steps_rate": max_steps_rate,
        "thresholds": {"min_final_mu": min_final_mu, "max_steps_rate": max_steps_rate_max},
        "totals": {"completions": total_completions}
    }

class Handler(BaseHTTPRequestHandler):
    def _authorized(self, qs):
        tok = os.environ.get("URCM_METRICS_TOKEN")
        if not tok:
            return True
        qtok = None
        if "token" in qs and qs["token"]:
            qtok = qs["token"][0]
        ah = self.headers.get("Authorization", "")
        if ah.startswith("Bearer "):
            if ah[len("Bearer "):] == tok:
                return True
        if qtok and qtok == tok:
            return True
        return False

    def do_GET(self):
        try:
            parsed = urlparse(self.path)
            qs = parse_qs(parsed.query)
            env = None
            if "env" in qs and qs["env"]:
                env = qs["env"][0]
            if not self._authorized(qs):
                self.send_response(401)
                self.end_headers()
                return
            if parsed.path == "/metrics":
                body = render_metrics(os.getcwd(), env=env).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; version=0.0.4")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif parsed.path == "/health":
                summary = compute_health(os.getcwd(), env=env)
                body = json.dumps(summary).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                self.send_response(404)
                self.end_headers()
        except Exception:
            self.send_response(500)
            self.end_headers()

def main():
    port = int(os.environ.get("URCM_METRICS_PORT", "8008"))
    bind = os.environ.get("URCM_METRICS_BIND", "127.0.0.1")
    server = HTTPServer((bind, port), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()

if __name__ == "__main__":
    main()
