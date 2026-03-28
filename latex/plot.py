"""
Generate publication-quality figures for the IntentFlow benchmark.

Usage:
    python -m latex.plot
    python -m latex.plot --out-dir latex/figure
"""

import json
import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ── Paths ────────────────────────────────────────────────────────────────────

PROJ_ROOT  = Path(__file__).resolve().parent.parent
ANN_FILE   = PROJ_ROOT / "bench" / "data" / "LatentNeeds-Bench.jsonl"
SCORES_DIR = PROJ_ROOT / "eval" / "scores"
FIG_DIR    = PROJ_ROOT / "latex" / "figure"

BUCKET_SIZE = 4
N_BUCKETS   = 15  # 15 × 4 = 60 turns

# Legend order: protagonist first, strong baselines, then the rest
MODELS = [
    ("IntentFlow",            "intentflow"),
    ("GPT-5-Mini",            "gpt-5-mini"),
    ("Gemini-3-Flash",        "gemini-3-flash"),
    ("GPT-5-Nano",            "gpt-5-nano"),
    ("GPT-oss-120b",          "gpt-oss-120b"),
    ("Claude-Haiku-4.5",      "claude-haiku-4.5"),
    ("DeepSeek-V3.2",         "deepseek-v3.2"),
    ("Qwen3.5-Flash",         "qwen3.5-flash"),
    ("Qwen3-30B-A3B",        "qwen3-30b-a3b"),
    ("Gemini-2.5-Flash-Lite", "gemini-2.5-flash-lite"),
]

# ── Visual hierarchy ─────────────────────────────────────────────────────────

PRIMARY   = {"IntentFlow"}
SECONDARY = {"GPT-5-Mini", "Gemini-3-Flash"}

# Muted academic palette
# PALETTE = {
#     "IntentFlow":            "#111111",
#     "GPT-5-Mini":            "#4C78A8",
#     "Gemini-3-Flash":        "#C44E52",
#     "GPT-5-Nano":            "#7DA6D9",
#     "GPT-oss-120b":          "#59A14F",
#     "Claude-Haiku-4.5":      "#E8A63A",
#     "DeepSeek-V3.2":         "#8E6BBE",
#     "Qwen3.5-Flash":         "#B07AA1",
#     "Gemini-2.5-Flash-Lite": "#9D9D9D",
# }

PALETTE = {
    # "IntentFlow":            "#4B3F72",  # 深靛紫，替代纯黑
    "IntentFlow": "#6B5FB5",
    "GPT-5-Mini":            "#5B8FD6",  # 柔和主蓝
    "Gemini-3-Flash":        "#C76D8A",  # 灰调玫红
    "GPT-5-Nano":            "#8DB6E8",  # 浅蓝
    "GPT-oss-120b":          "#7FB77E",  # 柔和绿
    "Claude-Haiku-4.5":      "#E3A857",  # muted gold
    "DeepSeek-V3.2":         "#8C74C9",  # 紫色
    "Qwen3.5-Flash":         "#C08ACD",  # 淡紫粉
    "Qwen3-30B-A3B":         "#9D174D",
    "Gemini-2.5-Flash-Lite": "#B8B0C9",  # 带紫调的灰
}

LSTYLES = {
    "IntentFlow": "-",
    "GPT-5-Mini": "-",       "Gemini-3-Flash": "-",
    "GPT-5-Nano": "--",      "GPT-oss-120b": "--",
    "Claude-Haiku-4.5": "-.","DeepSeek-V3.2": "-.",
    "Qwen3.5-Flash": "--",  "Qwen3-30B-A3B": "--",
    "Gemini-2.5-Flash-Lite": ":",
}

MARKERS = {
    "IntentFlow": "*",
    "GPT-5-Mini": "o",       "Gemini-3-Flash": "D",
    "GPT-5-Nano": None,      "GPT-oss-120b": None,
    "Claude-Haiku-4.5": None, "DeepSeek-V3.2": None,
    "Qwen3.5-Flash": None,   "Qwen3-30B-A3B": "h",
    "Gemini-2.5-Flash-Lite": None,
}


def apply_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 7.5,
        "axes.titlesize": 8.5,
        "axes.labelsize": 8,
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "grid.linewidth": 0.35,
        "grid.alpha": 0.22,
        "legend.fontsize": 6.5,
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
    })


def line_kw(display):
    """Return plot kwargs with visual hierarchy: primary > secondary > rest."""
    marker = MARKERS.get(display)
    if display in PRIMARY:
        return dict(
            color=PALETTE[display], linestyle="-", linewidth=1.8,
            marker=marker, markersize=3.5, markeredgewidth=0,
            alpha=0.98, zorder=5,
        )
    elif display in SECONDARY:
        return dict(
            color=PALETTE[display], linestyle="-", linewidth=1.3,
            marker=marker, markersize=2.5, markeredgewidth=0,
            alpha=0.90, zorder=4,
        )
    else:
        return dict(
            color=PALETTE[display], linestyle=LSTYLES[display], linewidth=0.9,
            marker=marker, markersize=2.0, markeredgewidth=0,
            alpha=0.68, zorder=2,
        )


# ── Data loading ─────────────────────────────────────────────────────────────

def load_gt():
    gt = {}
    with open(ANN_FILE) as f:
        for line in f:
            if not line.strip():
                continue
            s = json.loads(line)
            for i, t in enumerate(s["turns"]):
                gt[(s["session_id"], i)] = t.get("annotation", {}).get("has_demand", False)
    return gt


def find_best_scores(key, gt):
    best_bal = -1
    best_scores = []
    for sf in sorted(SCORES_DIR.glob(f"{key}_*.jsonl")):
        scores = []
        with open(sf) as f:
            for line in f:
                if line.strip():
                    scores.append(json.loads(line))
        if not scores:
            continue
        d_ok = d_n = nd_ok = nd_n = 0
        for s in scores:
            hd = gt.get((s["session_id"], s["turn_id"]))
            if hd is None:
                hd = s["label"] in ("fn", "tp", "tp_bad")
            if hd:
                d_n += 1; d_ok += s["score"]
            else:
                nd_n += 1; nd_ok += s["score"]
        bal = ((d_ok / d_n if d_n else 0) + (nd_ok / nd_n if nd_n else 0)) / 2 * 100
        if bal > best_bal:
            best_bal = bal
            best_scores = scores
    return best_scores


def build_per_turn(gt):
    per_turn = {}
    for display, key in MODELS:
        scores = find_best_scores(key, gt)
        if not scores:
            continue
        buckets = defaultdict(lambda: {"d_ok": 0, "d_n": 0, "nd_ok": 0, "nd_n": 0})
        for s in scores:
            tid = s["turn_id"]
            if tid >= N_BUCKETS * BUCKET_SIZE:
                continue
            hd = gt.get((s["session_id"], tid), False)
            if hd:
                buckets[tid]["d_n"] += 1
                buckets[tid]["d_ok"] += s["score"]
            else:
                buckets[tid]["nd_n"] += 1
                buckets[tid]["nd_ok"] += s["score"]
        per_turn[display] = buckets
    return per_turn


# ── Compute ──────────────────────────────────────────────────────────────────

def _pct(ok, n):
    return ok / n * 100 if n else None

def _bavg(d, nd):
    if d is not None and nd is not None:
        return (d + nd) / 2
    return d if d is not None else nd


def compute_bucket(buckets):
    vals = []
    for bi in range(N_BUCKETS):
        d_ok = d_n = nd_ok = nd_n = 0
        for off in range(BUCKET_SIZE):
            b = buckets[bi * BUCKET_SIZE + off]
            d_ok += b["d_ok"]; d_n += b["d_n"]
            nd_ok += b["nd_ok"]; nd_n += b["nd_n"]
        vals.append(_bavg(_pct(d_ok, d_n), _pct(nd_ok, nd_n)))
    return vals


def compute_bucket_cumul(buckets):
    vals = []
    cum_d_ok = cum_d_n = cum_nd_ok = cum_nd_n = 0
    for bi in range(N_BUCKETS):
        for off in range(BUCKET_SIZE):
            b = buckets[bi * BUCKET_SIZE + off]
            cum_d_ok += b["d_ok"]; cum_d_n += b["d_n"]
            cum_nd_ok += b["nd_ok"]; cum_nd_n += b["nd_n"]
        vals.append(_bavg(_pct(cum_d_ok, cum_d_n), _pct(cum_nd_ok, cum_nd_n)))
    return vals


# ── Shared helpers ───────────────────────────────────────────────────────────

BUCKET_LABELS = [f"{b*BUCKET_SIZE+1}–{(b+1)*BUCKET_SIZE}" for b in range(N_BUCKETS)]


def _top_legend(fig, ax_for_handles):
    """Place a frameless legend at the top of the figure."""
    handles, labels = ax_for_handles.get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=5,
        fontsize=6.5,
        columnspacing=0.8,
        handlelength=2.0,
        handletextpad=0.4,
        frameon=False,
    )


def _format_ax(ax, title, ylabel=None, ylim=(44, 94)):
    ax.set_title(title, fontsize=8.5, pad=5)
    ax.set_xticks(np.arange(N_BUCKETS)[::2])
    ax.set_xticklabels(
        [BUCKET_LABELS[i] for i in range(0, N_BUCKETS, 2)],
        rotation=40, ha="right", fontsize=6.5,
    )
    ax.set_ylim(*ylim)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(10))
    ax.grid(True, axis="y", linestyle="--")
    ax.set_xlabel("Turn position (4-turn buckets)", fontsize=7.5)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=8)


# ── Plotting ─────────────────────────────────────────────────────────────────

def plot_combined(per_turn, out_dir):
    """Main figure: per-bucket (a) + cumulative (b), side by side."""
    bucket_data = {d: compute_bucket(per_turn[d]) for d in per_turn}
    cumul_data  = {d: compute_bucket_cumul(per_turn[d]) for d in per_turn}

    x = np.arange(N_BUCKETS)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=True)

    for display, _ in MODELS:
        if display not in bucket_data:
            continue
        y1 = np.array([v if v is not None else np.nan for v in bucket_data[display]])
        y2 = np.array([v if v is not None else np.nan for v in cumul_data[display]])
        ax1.plot(x, y1, label=display, **line_kw(display))
        ax2.plot(x, y2, label=display, **line_kw(display))

    _format_ax(ax1, "(a) Per-Bucket", ylabel="Balanced Accuracy (%)")
    _format_ax(ax2, "(b) Cumulative")

    _top_legend(fig, ax1)
    fig.tight_layout()
    fig.subplots_adjust(top=0.78, wspace=0.06)

    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"multi_turn_combined.{ext}")
    plt.close(fig)
    print(f"  multi_turn_combined.pdf/.png")


# ── Main ─────────────────────────────────────────────────────────────────────

def main(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    apply_style()

    gt = load_gt()
    per_turn = build_per_turn(gt)

    print("Generating figures:")
    plot_combined(per_turn, out_dir)
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=str(FIG_DIR))
    args = parser.parse_args()
    main(Path(args.out_dir))
