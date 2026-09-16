"""Lightweight progress view; percentages count requests, including DES seeds."""
import json
from pathlib import Path

root = Path(__file__).resolve().parents[2]
out = root/"results/lstm_comparison_v1"
inventory = json.loads((out/"progress_inventory.json").read_text())
completed = 0
for name, spec in inventory.items():
    directory = out/"cases"/name
    if (directory/"summary.json").exists():
        completed += 1
        continue
    requests = 0
    jobs = 0
    for f, count in enumerate(spec["function_requests"]):
        for s, _ in enumerate(spec["seeds"]):
            if ((directory/"functions"/f"{f:05d}.npz").exists() or
                (directory/"seed_functions"/f"{f:05d}_{s:02d}.npz").exists()):
                requests += count
                jobs += 1
            else:
                record = directory/"partial_states"/f"{f:05d}_{s:02d}.json"
                if record.exists():
                    requests += json.loads(record.read_text())["processed_requests"]
    print(f"{name}: {100*requests/spec['total_requests']:.1f}% requests; {jobs} complete function-seeds")
print(f"DES cases complete: {completed}/{len(inventory)}")
for label, directory in [("Development live",out/"live"),("Primary live",out/"strict_round/live")]:
    state = ("complete" if (directory/"results.json").exists() else
             "preparation / replay" if (directory/"protocol.json").exists() else "waiting")
    print(f"{label}: {state}")
