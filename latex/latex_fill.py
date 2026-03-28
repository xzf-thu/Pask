"""
Generate final LaTeX tables for the IntentFlow benchmark.

Reads {model}_{level}.jsonl from scores/ and results/ dirs.
Picks the best prompt level per model (by overall balanced score).
Generates: main_table.tex, latency.tex, multi_turn.tex, demand_type.tex, demand_latency.tex

Usage:
    python -m latex.latex_fill
    python -m latex.latex_fill --out-dir latex/output
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict

import sys
PROJ_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ_ROOT / "bench"))

EVAL_DIR    = PROJ_ROOT / "eval"
RESULTS_DIR = EVAL_DIR / "results"
SCORES_DIR  = EVAL_DIR / "scores"
ANN_FILE    = PROJ_ROOT / "bench" / "data" / "LatentNeeds-Bench.jsonl"
DEFAULT_OUT = PROJ_ROOT / "latex" / "output"

SUBCATS = ["W1", "W2", "W3", "W4", "L1", "L2", "L3", "D1", "D2", "D3"]

# (display_name, score_key_prefix, $/M-in, $/M-out)
MODELS_ORDER = [
    ("GPT-5-Mini",            "gpt-5-mini",            0.25,  2.0),
    ("GPT-5-Nano",            "gpt-5-nano",            0.05,  0.4),
    ("GPT-oss-120b",          "gpt-oss-120b",          0.0,   0.0),
    ("Gemini-3-Flash",        "gemini-3-flash",        0.5,   3.0),
    ("Gemini-2.5-Flash-Lite", "gemini-2.5-flash-lite", 0.1,   0.4),
    ("Claude-Haiku-4.5",      "claude-haiku-4.5",      1.0,   5.0),
    ("DeepSeek-V3.2",         "deepseek-v3.2",         0.25,  0.4),
    ("Qwen3.5-Flash",         "qwen3.5-flash",         0.1,   0.4),
    ("Qwen3-30B-A3B",        "qwen3-30b-a3b",         0.0,   0.0),
    (r"\textbf{IntentFlow}",  "intentflow",            0.0,   0.0),
]

BUCKET_SIZE = 4
N_BUCKETS   = 15  # 15 × 4 = 60 turns


# ── helpers ──────────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    with open(path) as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def build_ground_truth() -> dict:
    """Return {(session_id, turn_id): has_demand}."""
    gt = {}
    for s in load_jsonl(ANN_FILE):
        sid = s["session_id"]
        for i, t in enumerate(s["turns"]):
            has = t.get("annotation", {}).get("has_demand", False)
            gt[(sid, i)] = has
    return gt


def build_demand_type_index() -> dict:
    """Return {(session_id, turn_id): category} where category is 'Req'/'Ins'/None."""
    idx = {}
    for s in load_jsonl(ANN_FILE):
        sid = s["session_id"]
        for i, t in enumerate(s["turns"]):
            ann = t.get("annotation", {})
            if not ann.get("has_demand"):
                idx[(sid, i)] = None
                continue
            cats = set(d.get("category", "") for d in ann.get("demands", []))
            if "Req" in cats:
                idx[(sid, i)] = "Req"
            elif "Ins" in cats:
                idx[(sid, i)] = "Ins"
            else:
                idx[(sid, i)] = None  # 'other' etc
    return idx


def find_best_level(key: str, gt: dict) -> tuple[str | None, list[dict], list[dict]]:
    """Find the prompt level with best balanced score for a model.
    Returns (level, scores, results)."""
    best_level = None
    best_bal = -1
    best_scores = []
    best_results = []

    for score_file in sorted(SCORES_DIR.glob(f"{key}_*.jsonl")):
        stem = score_file.stem
        level = stem[len(key) + 1:]
        scores = load_jsonl(score_file)
        if not scores:
            continue

        # Compute balanced score
        d_ok = d_n = nd_ok = nd_n = 0
        for s in scores:
            k = (s["session_id"], s["turn_id"])
            has_demand = gt.get(k)
            if has_demand is None:
                # fallback to label
                has_demand = s["label"] in ("fn", "tp", "tp_bad")
            if has_demand:
                d_n += 1
                if s["score"] == 1:
                    d_ok += 1
            else:
                nd_n += 1
                if s["score"] == 1:
                    nd_ok += 1

        d_acc = d_ok / d_n if d_n else 0
        nd_acc = nd_ok / nd_n if nd_n else 0
        bal = (d_acc + nd_acc) / 2 * 100

        if bal > best_bal:
            best_bal = bal
            best_level = level
            best_scores = scores
            best_results = load_jsonl(RESULTS_DIR / f"{stem}.jsonl")

    return best_level, best_scores, best_results


def fmt(v) -> str:
    return "--" if v is None else f"{v:.1f}"


def bold(s):   return s if s == "--" else r"\textbf{" + s + "}"
def italic(s): return s if s == "--" else r"\textit{" + s + "}"


def decorate(s: str, rank: int) -> str:
    if s == "--":
        return s
    if rank == 1:
        return bold(s)
    if rank == 2:
        return italic(s)
    return s


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN TABLE
# ══════════════════════════════════════════════════════════════════════════════

def compute_subcat_scores(scores: list[dict], gt: dict) -> dict:
    """Return {subcategory: {demand, no_demand, avg}} + 'overall' key."""
    buckets = defaultdict(lambda: {"d": [0, 0], "nd": [0, 0]})

    for s in scores:
        key = (s["session_id"], s["turn_id"])
        sc = s["subcategory"]
        val = s["score"]
        has_demand = gt.get(key)
        if has_demand is None:
            continue
        if has_demand:
            buckets[sc]["d"][0] += val
            buckets[sc]["d"][1] += 1
        else:
            buckets[sc]["nd"][0] += val
            buckets[sc]["nd"][1] += 1

    def pct(correct, total):
        return round(correct / total * 100, 1) if total else None

    def avg2(a, b):
        if a is None and b is None:
            return None
        if a is None:
            return b
        if b is None:
            return a
        return round((a + b) / 2, 1)

    result = {}
    for sc in SUBCATS:
        b = buckets.get(sc, {"d": [0, 0], "nd": [0, 0]})
        d = pct(*b["d"])
        nd = pct(*b["nd"])
        result[sc] = {"demand": d, "no_demand": nd, "avg": avg2(d, nd)}

    def macro(key):
        vals = [result[sc][key] for sc in SUBCATS if result[sc][key] is not None]
        return round(sum(vals) / len(vals), 1) if vals else None

    result["overall"] = {
        "demand": macro("demand"),
        "no_demand": macro("no_demand"),
        "avg": macro("avg"),
    }
    return result


def compute_ranks(all_data: dict[str, dict]) -> dict:
    """Return {col: {row_type: {model_key: rank}}}."""
    cols = SUBCATS + ["overall"]
    row_types = ["demand", "no_demand", "avg"]
    ranks = {}
    for col in cols:
        ranks[col] = {}
        for rt in row_types:
            vals = []
            for mkey, bd in all_data.items():
                v = bd.get(col, {}).get(rt)
                if v is not None:
                    vals.append((v, mkey))
            vals.sort(reverse=True)
            col_ranks = {}
            for i, (_, mkey) in enumerate(vals[:2]):
                col_ranks[mkey] = i + 1
            ranks[col][rt] = col_ranks
    return ranks


MAIN_HEADER = r"""\begin{table*}[t]
\centering
\setlength{\tabcolsep}{4pt}
\resizebox{\linewidth}{!}{%
\begin{tabular}{@{}l|l|cccc|ccc|ccc|c@{}}
\toprule
\multirow{3}{*}{\textbf{Model}} & \multirow{3}{*}{\textbf{Type}} &
\multicolumn{4}{c|}{\textbf{Work}} &
\multicolumn{3}{c|}{\textbf{Learning}} &
\multicolumn{3}{c|}{\textbf{Daily}} &
\multirow{3}{*}{\textbf{Overall}} \\
\cmidrule(lr){3-6} \cmidrule(lr){7-9} \cmidrule(lr){10-12}
& & \footnotesize Business & \footnotesize Product & \footnotesize Tech & \footnotesize Work & \footnotesize STEM & \footnotesize Program. & \footnotesize Human. & \footnotesize Personal & \footnotesize Tools \& & \footnotesize Content & \\
& & \footnotesize Metrics & \footnotesize Strategy & \footnotesize Engineer. & \footnotesize Collab. & \footnotesize Lecture & \footnotesize Tutorial & \footnotesize Business & \footnotesize Life & \footnotesize Workflow & \footnotesize Knowl. & \\
\midrule
"""

MAIN_FOOTER = r"""
\bottomrule
\end{tabular}%
}
\caption{Main results on the \textsc{IntentFlow} Proactive Demand Detection Benchmark.
Each cell reports the turn-level accuracy score (0--100).
\textit{Demand} = accuracy on demand turns;
\textit{No-Dem.} = accuracy on non-demand turns;
\textit{Avg.} = balanced average (1:1 demand/non-demand).
Columns are grouped by domain:
\textbf{Work} (Business Metrics, Product Strategy, Tech Engineer., Work Collab.),
\textbf{Learning} (STEM Lecture, Program.\ Tutorial, Human.\ Business),
\textbf{Daily} (Personal Life, Tools \& Workflow, Content \& Knowl.).
\textbf{Bold} = best per column; \textit{Italic} = second best.}
\label{tab:main_results}
\end{table*}
"""


def gen_main_table(out_dir: Path, gt: dict):
    # Load best-level scores per model
    all_data: dict[str, dict] = {}
    level_chosen: dict[str, str] = {}

    for _, key, *_ in MODELS_ORDER:
        level, scores, _ = find_best_level(key, gt)
        if scores:
            all_data[key] = compute_subcat_scores(scores, gt)
            level_chosen[key] = level

    ranks = compute_ranks(all_data)
    cols = SUBCATS + ["overall"]

    # Build LaTeX
    tex = MAIN_HEADER

    entries = [(d, k) for d, k, *_ in MODELS_ORDER]

    for i, (display, score_key) in enumerate(entries):
        bd = all_data.get(score_key) if score_key else None
        is_last = (i == len(entries) - 1)

        def make_row(rt, label, gray, sk=score_key, bd=bd):
            cells = []
            for col in cols:
                v = None if bd is None else bd.get(col, {}).get(rt)
                s = fmt(v)
                rk = 0 if sk is None else ranks.get(col, {}).get(rt, {}).get(sk, 0)
                s = decorate(s, rk)
                cells.append(r"\grayrow{" + s + "}" if gray else s)
            return "   & ".join(cells)

        tex += f"\n\\multirow{{3}}{{*}}{{{display}}}\n"
        tex += f"& {'Demand.':<10} & {make_row('demand', 'Demand.', False)} \\\\\n"
        tex += f"& {'No-Dem.':<10} & {make_row('no_demand', 'No-Dem.', False)} \\\\\n"
        tex += f"& \\grayrow{{Avg.}} & {make_row('avg', 'Avg.', True)} \\\\\n"
        if not is_last:
            tex += "\\midrule\n"

    tex += MAIN_FOOTER

    out_path = out_dir / "main_table.tex"
    out_path.write_text(tex)
    print(f"Written: {out_path}")

    # Summary to console
    print(f"\n{'Model':<25} {'Level':<14} {'W1':>5} {'W2':>5} {'W3':>5} {'W4':>5} "
          f"{'L1':>5} {'L2':>5} {'L3':>5} {'D1':>5} {'D2':>5} {'D3':>5} {'Ovr':>5}")
    print("-" * 100)
    for display, key, *_ in MODELS_ORDER:
        if key not in all_data:
            print(f"  {display:<23} {'(no data)':<14}")
            continue
        bd = all_data[key]
        lv = level_chosen.get(key, "?")
        vals = [fmt(bd.get(sc, {}).get("avg")) for sc in SUBCATS]
        ov = fmt(bd.get("overall", {}).get("avg"))
        print(f"  {display:<23} {lv:<14} " + "  ".join(f"{v:>5}" for v in vals) + f"  {ov:>5}")


# ══════════════════════════════════════════════════════════════════════════════
#  LATENCY TABLE
# ══════════════════════════════════════════════════════════════════════════════

def gen_latency(out_dir: Path, gt: dict):
    rows = []
    for display, key, price_in, price_out in MODELS_ORDER:
        level, _, results = find_best_level(key, gt)
        if not results:
            rows.append((display, key, None, None, None, None, None, price_in, price_out))
            continue
        total_in  = sum(r.get("input_tokens", 0) for r in results)
        total_out = sum(r.get("output_tokens", 0) for r in results)
        cost = (total_in * price_in + total_out * price_out) / 1_000_000
        lats = [r.get("latency_ms", 0) for r in results if r.get("latency_ms", 0) > 0]
        avg_lat = round(sum(lats) / len(lats)) if lats else 0
        p95_lat = round(sorted(lats)[int(len(lats) * 0.95)]) if lats else 0
        rows.append((display, key, cost, avg_lat, p95_lat, total_in, total_out, price_in, price_out))

    # Rank: min cost, min latency
    def rank_min(idx):
        vals = [(r[idx], r[1]) for r in rows if r[idx] is not None and r[idx] > 0]
        vals.sort()
        rk = {}
        for i, (_, mk) in enumerate(vals[:2]):
            rk[mk] = i + 1
        return rk

    cost_rk    = rank_min(2)
    avg_lat_rk = rank_min(3)
    p95_lat_rk = rank_min(4)

    def dec(val, rk_dict, key, fmtstr):
        s = fmtstr(val)
        r = rk_dict.get(key, 0)
        if r == 1: return bold(s)
        if r == 2: return italic(s)
        return s

    lines = []
    lines.append(r"""\begin{table}[t]
\centering
\setlength{\tabcolsep}{4pt}
\resizebox{\linewidth}{!}{%
\begin{tabular}{@{}l|r|rr|rr|rr@{}}
\toprule
\textbf{Model} & \textbf{Cost(\$)} & \textbf{Avg(ms)} & \textbf{P95(ms)} & \textbf{In-Tok(K)} & \textbf{Out-Tok(K)} & \textbf{\$/M-in} & \textbf{\$/M-out} \\
\midrule""")

    for display, key, cost, avg_lat, p95_lat, total_in, total_out, pi, po in rows:
        if cost is None:
            cells = " & ".join(["--"] * 6)
            lines.append(f"{display} & {cells} & {pi:.3f} & {po:.3f} \\\\")
            continue
        c_cost    = dec(cost, cost_rk, key, lambda v: f"{v:.3f}")
        c_avg_lat = dec(avg_lat, avg_lat_rk, key, lambda v: f"{v:,}")
        c_p95_lat = dec(p95_lat, p95_lat_rk, key, lambda v: f"{v:,}")
        c_in  = f"{total_in/1000:.0f}"
        c_out = f"{total_out/1000:.0f}"
        lines.append(f"{display} & {c_cost} & {c_avg_lat} & {c_p95_lat} & {c_in} & {c_out} & {pi:.3f} & {po:.3f} \\\\")

    lines.append(r"""\bottomrule
\end{tabular}%
}
\caption{Cost and latency comparison across models on the full benchmark.
Cost is computed from per-token pricing.
\textbf{Bold} = best (lowest); \textit{Italic} = second best.}
\label{tab:latency}
\end{table}""")

    tex = "\n".join(lines) + "\n"
    out_path = out_dir / "latency.tex"
    out_path.write_text(tex)
    print(f"Written: {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  MULTI-TURN TABLE
# ══════════════════════════════════════════════════════════════════════════════

def gen_multi_turn(out_dir: Path, gt: dict):
    bucket_labels = [f"{b*BUCKET_SIZE+1}-{(b+1)*BUCKET_SIZE}" for b in range(N_BUCKETS)]

    all_data: dict[str, dict[str, list]] = {}

    for _, key, *_ in MODELS_ORDER:
        _, scores, _ = find_best_level(key, gt)
        if not scores:
            continue

        buckets = defaultdict(lambda: {"d_ok": 0, "d_n": 0, "nd_ok": 0, "nd_n": 0})
        for s in scores:
            tid = s["turn_id"]
            bi = tid // BUCKET_SIZE
            if bi >= N_BUCKETS:
                continue
            has_demand = gt.get((s["session_id"], tid), False)
            if has_demand:
                buckets[bi]["d_n"] += 1
                if s["score"] == 1:
                    buckets[bi]["d_ok"] += 1
            else:
                buckets[bi]["nd_n"] += 1
                if s["score"] == 1:
                    buckets[bi]["nd_ok"] += 1

        demand_vals, no_demand_vals, avg_vals = [], [], []
        for bi in range(N_BUCKETS):
            b = buckets[bi]
            d  = b["d_ok"] / b["d_n"] * 100 if b["d_n"] else None
            nd = b["nd_ok"] / b["nd_n"] * 100 if b["nd_n"] else None
            if d is not None and nd is not None:
                avg = (d + nd) / 2
            elif d is not None:
                avg = d
            elif nd is not None:
                avg = nd
            else:
                avg = None
            demand_vals.append(d)
            no_demand_vals.append(nd)
            avg_vals.append(avg)

        all_data[key] = {"demand": demand_vals, "no_demand": no_demand_vals, "avg": avg_vals}

    # Ranks per column per row_type
    def rank_cols(data, n_cols, row_types):
        ranks = {}
        for ci in range(n_cols):
            ranks[ci] = {}
            for rt in row_types:
                vals = []
                for mkey, rows in data.items():
                    v = rows[rt][ci] if ci < len(rows[rt]) else None
                    if v is not None:
                        vals.append((v, mkey))
                vals.sort(reverse=True)
                col_ranks = {}
                for i, (_, mk) in enumerate(vals[:2]):
                    col_ranks[mk] = i + 1
                ranks[ci][rt] = col_ranks
        return ranks

    ranks = rank_cols(all_data, N_BUCKETS, ["demand", "no_demand", "avg"])

    # Delta per row type: relative change first→last bucket
    # {model_key: {row_type: delta_value}}
    deltas = {}
    for key, rows in all_data.items():
        deltas[key] = {}
        for rt in ["demand", "no_demand", "avg"]:
            first = rows[rt][0]
            last = rows[rt][-1]
            if first is not None and last is not None and first > 0:
                deltas[key][rt] = round((last - first) / first * 100, 1)
            else:
                deltas[key][rt] = None

    # Rank delta per row type: least absolute change = best (most stable)
    delta_ranks = {}
    for rt in ["demand", "no_demand", "avg"]:
        vals = []
        for key in all_data:
            v = deltas[key][rt]
            if v is not None:
                vals.append((abs(v), key))
        vals.sort()
        delta_ranks[rt] = {}
        for i, (_, k) in enumerate(vals[:2]):
            delta_ranks[rt][k] = i + 1

    # Build LaTeX
    col_spec = "c" * N_BUCKETS
    cmidrule_end = 2 + N_BUCKETS

    lines = []
    lines.append(r"""\begin{table*}[t]
\centering
\setlength{\tabcolsep}{3pt}
\resizebox{\linewidth}{!}{%""")
    lines.append(r"\begin{tabular}{@{}l|l|" + col_spec + r"|c@{}}")
    lines.append(r"\toprule")
    lines.append(
        r"\multirow{2}{*}{\textbf{Model}} & \multirow{2}{*}{\textbf{Type}} & "
        r"\multicolumn{" + str(N_BUCKETS) + r"}{c|}{\textbf{Turn Position (bucket)}} & "
        r"\multirow{2}{*}{$\Delta$(\%)} \\"
    )
    lines.append(r"\cmidrule(lr){3-" + str(cmidrule_end) + "}")

    label_cells = " & ".join(f"\\footnotesize {lb}" for lb in bucket_labels)
    lines.append(f"& & {label_cells} & \\\\")
    lines.append(r"\midrule")

    model_keys = [key for _, key, *_ in MODELS_ORDER]
    display_map = {key: display for display, key, *_ in MODELS_ORDER}

    for mi, key in enumerate(model_keys):
        display = display_map[key]
        if key not in all_data:
            empty = " & ".join(["--"] * N_BUCKETS)
            gray_empty = " & ".join([r"\grayrow{--}"] * N_BUCKETS)
            lines.append(f"\\multirow{{3}}{{*}}{{{display}}}")
            lines.append(f"& Demand. & {empty} & -- \\\\")
            lines.append(f"& No-Dem. & {empty} & -- \\\\")
            lines.append(f"& \\grayrow{{Avg.}} & {gray_empty} & \\grayrow{{--}} \\\\")
        else:
            def make_cells(rt, gray=False, k=key):
                cells = []
                for ci in range(N_BUCKETS):
                    v = all_data[k][rt][ci]
                    s = fmt(v)
                    rk = ranks.get(ci, {}).get(rt, {}).get(k, 0)
                    s = decorate(s, rk)
                    if gray:
                        s = r"\grayrow{" + s + "}"
                    cells.append(s)
                return " & ".join(cells)

            def make_delta(rt, gray=False, k=key):
                d = deltas[k][rt]
                s = fmt(d)
                rk = delta_ranks.get(rt, {}).get(k, 0)
                s = decorate(s, rk)
                if gray:
                    s = r"\grayrow{" + s + "}"
                return s

            lines.append(f"\\multirow{{3}}{{*}}{{{display}}}")
            lines.append(f"& Demand. & {make_cells('demand')} & {make_delta('demand')} \\\\")
            lines.append(f"& No-Dem. & {make_cells('no_demand')} & {make_delta('no_demand')} \\\\")
            lines.append(f"& \\grayrow{{Avg.}} & {make_cells('avg', gray=True)} & {make_delta('avg', gray=True)} \\\\")

        if mi < len(model_keys) - 1:
            lines.append(r"\midrule")

    lines.append(r"""
\bottomrule
\end{tabular}%
}
\caption{Per-turn-bucket performance under the multi-turn setting.
Turn positions are grouped into buckets of 4 consecutive turns.
\textit{Demand} = accuracy on demand turns;
\textit{No-Dem.} = accuracy on non-demand turns;
\textit{Avg.} = balanced average.
$\Delta$(\%) = relative change from first to last bucket.
\textbf{Bold} = best per column; \textit{Italic} = second best.}
\label{tab:turn_degradation}
\end{table*}""")

    tex = "\n".join(lines) + "\n"
    out_path = out_dir / "multi_turn.tex"
    out_path.write_text(tex)
    print(f"Written: {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  DEMAND TYPE TABLE (Req vs Ins)
# ══════════════════════════════════════════════════════════════════════════════

def compute_demand_type_scores(scores: list[dict], dt_index: dict) -> dict:
    """Return {subcategory: {req, ins, avg}} + 'overall' key.
    Only considers demand turns, split by Req/Ins category."""
    buckets = defaultdict(lambda: {"req": [0, 0], "ins": [0, 0]})

    for s in scores:
        key = (s["session_id"], s["turn_id"])
        sc = s["subcategory"]
        val = s["score"]
        cat = dt_index.get(key)
        if cat == "Req":
            buckets[sc]["req"][0] += val
            buckets[sc]["req"][1] += 1
        elif cat == "Ins":
            buckets[sc]["ins"][0] += val
            buckets[sc]["ins"][1] += 1

    def pct(correct, total):
        return round(correct / total * 100, 1) if total else None

    def avg2(a, b):
        if a is None and b is None:
            return None
        if a is None:
            return b
        if b is None:
            return a
        return round((a + b) / 2, 1)

    result = {}
    for sc in SUBCATS:
        b = buckets.get(sc, {"req": [0, 0], "ins": [0, 0]})
        r = pct(*b["req"])
        i = pct(*b["ins"])
        result[sc] = {"req": r, "ins": i, "avg": avg2(r, i)}

    def macro(key):
        vals = [result[sc][key] for sc in SUBCATS if result[sc][key] is not None]
        return round(sum(vals) / len(vals), 1) if vals else None

    result["overall"] = {
        "req": macro("req"),
        "ins": macro("ins"),
        "avg": macro("avg"),
    }
    return result


DTYPE_HEADER = r"""\begin{table*}[t]
\centering
\setlength{\tabcolsep}{4pt}
\resizebox{\linewidth}{!}{%
\begin{tabular}{@{}l|l|cccc|ccc|ccc|c@{}}
\toprule
\multirow{3}{*}{\textbf{Model}} & \multirow{3}{*}{\textbf{Demand}} &
\multicolumn{4}{c|}{\textbf{Work}} &
\multicolumn{3}{c|}{\textbf{Learning}} &
\multicolumn{3}{c|}{\textbf{Daily}} &
\multirow{3}{*}{\textbf{Overall}} \\
\cmidrule(lr){3-6} \cmidrule(lr){7-9} \cmidrule(lr){10-12}
& & \footnotesize Business & \footnotesize Product & \footnotesize Tech & \footnotesize Work & \footnotesize STEM & \footnotesize Program. & \footnotesize Human. & \footnotesize Personal & \footnotesize Tools \& & \footnotesize Content & \\
& & \footnotesize Metrics & \footnotesize Strategy & \footnotesize Engineer. & \footnotesize Collab. & \footnotesize Lecture & \footnotesize Tutorial & \footnotesize Business & \footnotesize Life & \footnotesize Workflow & \footnotesize Knowl. & \\
\midrule
"""

DTYPE_FOOTER = r"""
\bottomrule
\end{tabular}%
}
\caption{Performance by demand type on the \textsc{IntentFlow} benchmark.
Each cell reports the turn-level accuracy score (0--100) on demand turns only.
\textit{Req.} = Requirement-type demands (decision support, task planning, problem solving, summarization, information lookup);
\textit{Ins.} = Insight-type demands (risk warning, knowledge gap, callback reminder, context synthesis, trend insight, sentiment analysis).
\textbf{Bold} = best per column; \textit{Italic} = second best.}
\label{tab:demand_type}
\end{table*}
"""


def gen_demand_type(out_dir: Path, gt: dict):
    dt_index = build_demand_type_index()

    all_data: dict[str, dict] = {}
    level_chosen: dict[str, str] = {}

    for _, key, *_ in MODELS_ORDER:
        level, scores, _ = find_best_level(key, gt)
        if scores:
            all_data[key] = compute_demand_type_scores(scores, dt_index)
            level_chosen[key] = level

    # Compute ranks (same structure as main table but with req/ins/avg)
    cols = SUBCATS + ["overall"]
    row_types = ["req", "ins", "avg"]
    ranks = {}
    for col in cols:
        ranks[col] = {}
        for rt in row_types:
            vals = []
            for mkey, bd in all_data.items():
                v = bd.get(col, {}).get(rt)
                if v is not None:
                    vals.append((v, mkey))
            vals.sort(reverse=True)
            col_ranks = {}
            for i, (_, mkey) in enumerate(vals[:2]):
                col_ranks[mkey] = i + 1
            ranks[col][rt] = col_ranks

    # Build LaTeX
    tex = DTYPE_HEADER

    entries = [(d, k) for d, k, *_ in MODELS_ORDER]

    for i, (display, score_key) in enumerate(entries):
        bd = all_data.get(score_key) if score_key else None
        is_last = (i == len(entries) - 1)

        def make_row(rt, label, gray, sk=score_key, bd=bd):
            cells = []
            for col in cols:
                v = None if bd is None else bd.get(col, {}).get(rt)
                s = fmt(v)
                rk = 0 if sk is None else ranks.get(col, {}).get(rt, {}).get(sk, 0)
                s = decorate(s, rk)
                cells.append(r"\grayrow{" + s + "}" if gray else s)
            return "   & ".join(cells)

        tex += f"\n\\multirow{{3}}{{*}}{{{display}}}\n"
        tex += f"& {'Req.':<11} & {make_row('req', 'Req.', False)} \\\\\n"
        tex += f"& {'Ins.':<11} & {make_row('ins', 'Ins.', False)} \\\\\n"
        tex += f"& \\grayrow{{Avg.}} & {make_row('avg', 'Avg.', True)} \\\\\n"
        if not is_last:
            tex += "\\midrule\n"

    tex += DTYPE_FOOTER

    out_path = out_dir / "demand_type.tex"
    out_path.write_text(tex)
    print(f"Written: {out_path}")

    # Summary
    print(f"\n{'Model':<25} {'Level':<14} {'Req':>5} {'Ins':>5} {'Avg':>5}")
    print("-" * 55)
    for display, key, *_ in MODELS_ORDER:
        if key not in all_data:
            print(f"  {display:<23} {'(no data)':<14}")
            continue
        bd = all_data[key]
        lv = level_chosen.get(key, "?")
        ov = bd.get("overall", {})
        print(f"  {display:<23} {lv:<14} {fmt(ov.get('req')):>5} {fmt(ov.get('ins')):>5} {fmt(ov.get('avg')):>5}")


# ══════════════════════════════════════════════════════════════════════════════
#  DEMAND-LATENCY TABLE (latency by turn position, split by demand/non-demand)
# ══════════════════════════════════════════════════════════════════════════════

LAT_BUCKET_SIZE = 12
LAT_N_BUCKETS = 5   # 5 × 12 = 60 turns
LAT_BUCKET_LABELS = [f"{b*LAT_BUCKET_SIZE+1}--{(b+1)*LAT_BUCKET_SIZE}" for b in range(LAT_N_BUCKETS)]


def gen_demand_latency(out_dir: Path, gt: dict):
    # {model_key: {row_type: [avg_latency_per_bucket]}}
    all_data: dict[str, dict[str, list]] = {}

    for _, key, *_ in MODELS_ORDER:
        level, _, results = find_best_level(key, gt)
        if not results:
            continue

        buckets = defaultdict(lambda: {"d": [], "nd": []})
        for r in results:
            tid = r["turn_id"]
            lat = r.get("latency_ms", 0)
            if lat <= 0:
                continue
            bi = tid // LAT_BUCKET_SIZE
            if bi >= LAT_N_BUCKETS:
                continue
            has_demand = gt.get((r["session_id"], tid), False)
            if has_demand:
                buckets[bi]["d"].append(lat)
            else:
                buckets[bi]["nd"].append(lat)

        demand_vals, no_demand_vals, avg_vals = [], [], []
        for bi in range(LAT_N_BUCKETS):
            b = buckets[bi]
            d = round(sum(b["d"]) / len(b["d"])) if b["d"] else None
            nd = round(sum(b["nd"]) / len(b["nd"])) if b["nd"] else None
            all_lat = b["d"] + b["nd"]
            avg = round(sum(all_lat) / len(all_lat)) if all_lat else None
            demand_vals.append(d)
            no_demand_vals.append(nd)
            avg_vals.append(avg)

        all_data[key] = {"demand": demand_vals, "no_demand": no_demand_vals, "avg": avg_vals}

    # Ranks: lower latency = better, so rank ascending
    row_types = ["demand", "no_demand", "avg"]
    ranks = {}
    for ci in range(LAT_N_BUCKETS):
        ranks[ci] = {}
        for rt in row_types:
            vals = []
            for mkey, rows in all_data.items():
                v = rows[rt][ci]
                if v is not None:
                    vals.append((v, mkey))
            vals.sort()  # ascending: lower = better
            col_ranks = {}
            for i, (_, mk) in enumerate(vals[:2]):
                col_ranks[mk] = i + 1
            ranks[ci][rt] = col_ranks

    def fmt_ms(v):
        if v is None:
            return "--"
        if v >= 1000:
            return f"{v/1000:.1f}k"
        return str(v)

    # Build LaTeX
    n = LAT_N_BUCKETS
    col_spec = "ccc|" * (n - 1) + "ccc"
    cmidrule_ranges = []
    for i in range(n):
        start = 2 + i * 3
        end = start + 2
        cmidrule_ranges.append(f"\\cmidrule(lr){{{start}-{end}}}")

    t_cols = []
    for i in range(n):
        sep = "|" if i < n - 1 else ""
        t_cols.append(f"\\multicolumn{{3}}{{c{sep}}}{{$T$={LAT_BUCKET_LABELS[i]}}}")

    lines = []
    lines.append(r"""\begin{table*}[t]
\centering
\setlength{\tabcolsep}{4pt}
\resizebox{0.99\linewidth}{!}{%""")
    lines.append(r"\begin{tabular}{@{}l|" + col_spec + r"@{}}")
    lines.append(r"\toprule")
    lines.append(
        r"\multirow{2}{*}{\textbf{Model}} & "
        r"\multicolumn{" + str(n * 3) + r"}{c}{\textbf{Per-Turn Latency (ms)}} \\"
    )
    lines.append(r"\cmidrule(lr){2-" + str(1 + n * 3) + "}")
    lines.append("& " + " & ".join(t_cols) + r" \\")
    lines.append(" ".join(cmidrule_ranges))

    # Sub-column headers
    sub_header = "& " + " & ".join(
        [r"\footnotesize Dem. & \footnotesize N-D. & \footnotesize Avg."] * n
    ) + r" \\"
    lines.append(sub_header)
    lines.append(r"\midrule")

    model_keys = [key for _, key, *_ in MODELS_ORDER]
    display_map = {key: display for display, key, *_ in MODELS_ORDER}

    for mi, key in enumerate(model_keys):
        display = display_map[key]
        if key not in all_data:
            empty = " & ".join(["--"] * (n * 3))
            lines.append(f"{display} & {empty} \\\\")
        else:
            rows = all_data[key]
            cells = []
            for ci in range(LAT_N_BUCKETS):
                for rt in row_types:
                    v = rows[rt][ci]
                    s = fmt_ms(v)
                    rk = ranks.get(ci, {}).get(rt, {}).get(key, 0)
                    s = decorate(s, rk)
                    cells.append(s)
            lines.append(f"{display} & " + " & ".join(cells) + r" \\")

        if mi < len(model_keys) - 1:
            lines.append(r"\midrule")

    lines.append(r"""
\bottomrule
\end{tabular}%
}
\caption{Per-turn inference latency by conversation position.
Turn positions are grouped into buckets of """ + str(LAT_BUCKET_SIZE) + r""" consecutive turns ($T$).
\textit{Dem.} = average latency on demand turns (model generates a response);
\textit{N-D.} = average latency on non-demand turns (model outputs \texttt{[NO\_DEMAND]});
\textit{Avg.} = overall average.
Demand turns consistently incur higher latency due to longer generated outputs.
\textbf{Bold} = fastest; \textit{Italic} = second fastest.}
\label{tab:demand_latency}
\end{table*}""")

    tex = "\n".join(lines) + "\n"
    out_path = out_dir / "demand_latency.tex"
    out_path.write_text(tex)
    print(f"Written: {out_path}")

    # Summary
    print(f"\n{'Model':<25}", end="")
    for lb in LAT_BUCKET_LABELS:
        print(f"  {'T='+lb:>12}", end="")
    print()
    print("-" * (25 + 14 * LAT_N_BUCKETS))
    for _, key, *_ in MODELS_ORDER:
        display = [d for d, k, *_ in MODELS_ORDER if k == key][0]
        if key not in all_data:
            print(f"  {display:<23} (no data)")
            continue
        row = f"  {display:<23}"
        for ci in range(LAT_N_BUCKETS):
            avg = all_data[key]["avg"][ci]
            row += f"  {fmt_ms(avg):>12}"
        print(row)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    gt = build_ground_truth()
    gen_main_table(out_dir, gt)
    gen_latency(out_dir, gt)
    gen_multi_turn(out_dir, gt)
    gen_demand_type(out_dir, gt)
    gen_demand_latency(out_dir, gt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    args = parser.parse_args()
    main(Path(args.out_dir))
