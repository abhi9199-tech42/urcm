import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from urcm.core.performance import OptimizedPhonemeSet, PerformanceBenchmark

def generate():
    phoneme_set = OptimizedPhonemeSet(vector_dimension=24)
    bench = PerformanceBenchmark()
    tests = [
        ("Short", "Hello world"),
        ("Medium", "The quick brown fox jumps over the lazy dog multiple times"),
        ("Long", "This is a longer text that simulates a paragraph with multiple sentences. " * 3),
    ]
    mem_rows = []
    for label, text in tests:
        r = bench.benchmark_memory_efficiency(text=text, phoneme_set=phoneme_set, latent_dim=128)
        mem_rows.append({
            "label": label,
            "urcm_mb": r["urcm_memory_bytes"] / (1024*1024),
            "transformer_mb": r["token_memory_bytes"] / (1024*1024),
            "ratio": r["memory_efficiency_ratio"],
        })
    speed_rows = []
    for n in [10, 50, 100, 200]:
        r = bench.benchmark_processing_speed(phoneme_set=phoneme_set, num_phonemes=n)
        speed_rows.append({
            "n": n,
            "cached_ms": r["cached_time_ms"],
            "uncached_ms": r["uncached_time_ms"],
            "speedup": r["speedup_factor"],
        })
    max_mem = max(x["transformer_mb"] for x in mem_rows)
    max_cached = max(x["cached_ms"] for x in speed_rows)
    max_uncached = max(x["uncached_ms"] for x in speed_rows)
    html = []
    html.append("<!DOCTYPE html>")
    html.append("<html><head><meta charset='utf-8'><title>URCM vs Transformer Pitch</title>")
    html.append("<style>")
    html.append("body{font-family:Inter,Arial,sans-serif;margin:24px;color:#111} h1{font-size:28px} h2{font-size:20px;margin-top:24px} table{border-collapse:collapse;width:100%;margin-top:8px} th,td{border:1px solid #ddd;padding:8px;text-align:left} th{background:#f5f5f5} .bar{height:16px;background:#3b82f6} .bar2{height:16px;background:#ef4444} .grid{display:grid;grid-template-columns:1fr 1fr;gap:20px} .note{color:#555;font-size:12px}")
    html.append("</style></head><body>")
    html.append("<h1>Unified μ-Resonance Cognitive Mesh vs Transformer-like Token System</h1>")
    html.append("<p class='note'>CPU-first resonance engine compared to a token system with ~50k vocab and 768-d embeddings. Figures generated directly from the project benchmarks.</p>")
    html.append("<h2>Memory Footprint</h2>")
    html.append("<table><thead><tr><th>Scenario</th><th>URCM (MB)</th><th>Transformer (MB)</th><th>Efficiency (×)</th><th>Bars</th></tr></thead><tbody>")
    for x in mem_rows:
        w1 = 100 * (x["urcm_mb"] / max_mem) if max_mem > 0 else 0
        w2 = 100 * (x["transformer_mb"] / max_mem) if max_mem > 0 else 0
        html.append("<tr>")
        html.append(f"<td>{x['label']}</td><td>{x['urcm_mb']:.2f}</td><td>{x['transformer_mb']:.2f}</td><td>{x['ratio']:.2f}</td>")
        html.append("<td><div style='display:flex;gap:8px;align-items:center'>")
        html.append(f"<div class='bar' style='width:{w1:.1f}%'></div>")
        html.append(f"<div class='bar2' style='width:{w2:.1f}%'></div>")
        html.append("</div></td></tr>")
    html.append("</tbody></table>")
    html.append("<h2>Processing Speed</h2>")
    html.append("<table><thead><tr><th>Phonemes</th><th>Cached (ms)</th><th>Uncached (ms)</th><th>Speedup (×)</th><th>Bars</th></tr></thead><tbody>")
    for s in speed_rows:
        wc = 100 * (s["cached_ms"] / max_cached) if max_cached > 0 else 0
        wu = 100 * (s["uncached_ms"] / max_uncached) if max_uncached > 0 else 0
        html.append("<tr>")
        html.append(f"<td>{s['n']}</td><td>{s['cached_ms']:.2f}</td><td>{s['uncached_ms']:.2f}</td><td>{s['speedup']:.2f}</td>")
        html.append("<td><div style='display:flex;gap:8px;align-items:center'>")
        html.append(f"<div class='bar' style='width:{wc:.1f}%'></div>")
        html.append(f"<div class='bar2' style='width:{wu:.1f}%'></div>")
        html.append("</div></td></tr>")
    html.append("</tbody></table>")
    html.append("<h2>Key Takeaways</h2>")
    html.append("<ul>")
    html.append("<li>Fixed small phoneme set avoids large vocab tables.</li>")
    html.append("<li>Compression achieves ≥2× on average with float32 latent vectors.</li>")
    html.append("<li>Caching yields strong speedups on repeated phoneme access.</li>")
    html.append("<li>Transformer memory scales with vocab and sequence embeddings.</li>")
    html.append("</ul>")
    html.append("<p class='note'>Open locally or share this file as part of your pitch.</p>")
    html.append("</body></html>")
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "URCM_vs_Transformer_Pitch.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(html))
    print(out_path)
    mem_csv = os.path.join(out_dir, "URCM_BENCHMARK_MEMORY.csv")
    with open(mem_csv, "w", encoding="utf-8") as f:
        f.write("Scenario,URCM_MB,Transformer_MB,Efficiency_X\n")
        for x in mem_rows:
            f.write(f"{x['label']},{x['urcm_mb']:.6f},{x['transformer_mb']:.6f},{x['ratio']:.6f}\n")
    speed_csv = os.path.join(out_dir, "URCM_BENCHMARK_SPEED.csv")
    with open(speed_csv, "w", encoding="utf-8") as f:
        f.write("Phonemes,Cached_ms,Uncached_ms,Speedup_X\n")
        for s in speed_rows:
            f.write(f"{s['n']},{s['cached_ms']:.6f},{s['uncached_ms']:.6f},{s['speedup']:.6f}\n")

if __name__ == "__main__":
    generate()
