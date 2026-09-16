"""Repeat the five-arm live comparison with EWMA alpha fixed to 0.3."""
from pathlib import Path
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments/lstm_comparison'))
import numpy as np
import live_strict as strict
from common import digest, freeze, write_json
from cases import rates_ewma, decisions_from_rates, newsvendor_quantile

OUT = ROOT / 'results/fixed_ewma_v1/strict_round'
original_build = strict.experiment.live.build_schedules


def build_schedules(counts, features, pts, trainer, pm):
    seg, schedules, actions = original_build(counts, features, pts, trainer, pm)
    history = np.zeros_like(seg, dtype=np.int64)
    history[:, 1:] = np.cumsum(seg[:, :-1], axis=1, dtype=np.int64)
    mask = (history > 0) & (history < strict.experiment.live.GATE_THRESHOLD)
    replacement = decisions_from_rates(rates_ewma(seg, .3),
                                      newsvendor_quantile(strict.experiment.live.RHO))
    for j in range(2):
        np.testing.assert_array_equal(actions['protowarm'][j][mask], actions['ewma'][j][mask])
    actions['protowarm'] = tuple(np.where(mask, replacement[j], actions['protowarm'][j])
                                for j in range(2))
    actions['ewma'] = replacement
    return seg, schedules, actions


def freeze_protocol(path, contract):
    if path.name == 'protocol.json':
        contract.update(ewma_alpha=.3, ewma_weight='new observation',
                        revision='fixed alpha in independent EWMA and count-gated WINTER',
                        success_definition='HTTP 200--299; all attempts retained',
                        individual_requests='timestamp, latency, status, error, function and trace minute',
                        primary_round='fixed-alpha revision; all five arms replayed concurrently',
                        wrapper_sha256=digest(__file__),
                        experiment_sha256=digest(ROOT / 'experiments/lstm_comparison/live.py'),
                        strict_wrapper_sha256=digest(ROOT / 'experiments/lstm_comparison/live_strict.py'),
                        archived_helper_sha256=digest(ROOT / 'scripts/testbed_cohort.py'))
    freeze(path, contract)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    freeze(OUT / 'plan.json', dict(ewma_alpha=.3, arms=5, minutes=60,
        source='same cohort, requests, source checkpoint, controller and HTTP success rule',
        reason='User-authorized global EWMA coefficient revision',
        wrapper_sha256=digest(__file__)))
    strict.experiment.OUT = OUT
    strict.experiment.live.build_schedules = build_schedules
    strict.experiment.live.invoke = strict.invoke
    strict.experiment.live.run_arm = strict.run_arm
    strict.experiment.freeze = freeze_protocol
    stop = threading.Event()
    started = time.monotonic()
    def heartbeat():
        while not stop.wait(60):
            print('fixed-alpha live active', round(time.monotonic()-started), 'seconds', flush=True)
    threading.Thread(target=heartbeat, daemon=True).start()
    try:
        strict.experiment.main()
    finally:
        stop.set()


if __name__ == '__main__':
    main()
