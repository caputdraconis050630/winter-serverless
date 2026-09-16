"""Render the measured cold/warm latency CDFs side by side."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT.parent / "serverless-fewshot/results/runs/testbed_cdf.json"


def generate():
    data = json.loads(SOURCE.read_text())
    records = []
    styles = [
        ("python-ml", "Python (ML deps)", "#000000", "-"),
        ("node-api-real", "Node.js API", "#666666", (0, (4, 1.6))),
        ("java-svc", "Java service", "#9e9e9e", (0, (1, 1.3))),
    ]
    with plt.rc_context({
        "font.family": "DejaVu Sans", "font.size": 8,
        "axes.labelsize": 8, "axes.titlesize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.linewidth": .6, "axes.spines.top": False,
        "axes.spines.right": False, "legend.frameon": False,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    }):
        fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.1), layout="constrained")
        for ax, kind, scale, title, xlabel in zip(
            axes, ("cold", "warm"), (1, 1000),
            ("Cold invocations", "Warm invocations"),
            ("Cold-start latency (s)", "Warm latency (ms)"),
        ):
            for runtime, label, color, linestyle in styles:
                values = np.sort(np.asarray(data[kind][runtime], dtype=float))
                assert np.isfinite(values).all() and np.all(values >= 0)
                assert len(values) == (20 if kind == "cold" else 50)
                cdf = np.arange(1, len(values) + 1) / len(values)
                ax.step(values * scale, cdf, where="post", color=color,
                        linestyle=linestyle, linewidth=1.2,
                        label=f"{label} (n={len(values)})")
                records.append(dict(condition=kind, runtime=runtime,
                                    latency_seconds=values.tolist(), cdf=cdf.tolist()))
            ax.set(title=title, xlabel=xlabel, ylabel="CDF", ylim=(0, 1.02))
            ax.grid(axis="y", linewidth=.3, alpha=.35)
            ax.legend(loc="upper left", fontsize=6.5, handlelength=1.6)
        target = ROOT / "fig11_testbed_latency_cdf.pdf"
        fig.savefig(target, metadata={"CreationDate": None, "ModDate": None})
        plt.close(fig)
    evidence = dict(
        source=str(SOURCE), source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        output=target.name, output_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
        change="Side-by-side panels; measured samples and empirical CDF construction unchanged",
        curves=records,
    )
    (ROOT / "audit/calibration_figure_sources.json").write_text(
        json.dumps(evidence, indent=2) + "\n")
    return evidence


if __name__ == "__main__":
    generate()
