"""Check that resumed seed ledgers preserve their saved cold/idle prefixes."""
import numpy as np
from campaign import OUT
from common import digest, read_json, write_json
from simulator import _charge


def main():
    record = read_json(OUT / 'worker_reallocation.json')
    rows = []
    for name, expected in record['checkpoint_sha256'].items():
        checkpoint = OUT / 'worker_reallocation/checkpoints' / name
        completed = OUT / 'replay/cases/steady_huawei_h_saturated/seed_functions' / name
        assert digest(checkpoint) == expected
        with np.load(checkpoint) as before, np.load(completed) as after:
            assert str(before['contract_sha256']) == str(after['contract_sha256'])
            offset = int(before['offset'])
            reused = set(after['reused_actions'].tolist())
            active = [i for i in range(len(after['cold'])) if i not in reused]
            assert active and offset > 0
            np.testing.assert_array_equal(before['state0'][active, :offset, 1], after['cold'][active, :offset])
            # Column 2 is deferred allocated residency in a checkpoint, not
            # finalized idle memory. Close its open allocations at the boundary
            # and subtract the initialization/execution phases already charged.
            max_error = 0.
            for i in active:
                ledger = before['state0'][i].copy()
                pos, key, birth = before['state2'][i], before['state4'][i], before['state5'][i]
                for c in np.flatnonzero(pos[0] >= 0):
                    _charge(ledger, birth[c], min(key[0, c], offset * 60.), 2, .25)
                idle = ledger[:offset, 2] - ledger[:offset, 3] - ledger[:offset, 4]
                np.testing.assert_allclose(idle, after['idle'][i, :offset], rtol=3e-9, atol=1e-5)
                max_error = max(max_error, float(np.max(np.abs(idle - after['idle'][i, :offset]))))
        rows.append(dict(seed_file=name, saved_prefix_minutes=offset,
                         newly_simulated_action_sequences=len(active),
                         checkpoint_sha256=expected, result_sha256=digest(completed),
                         finalized_idle_max_absolute_error=max_error))
    result = dict(status='passed', checks='Exact cold prefix and independently finalized idle prefix retained under the unchanged execution contract',
                  resumed_seeds=rows, verifier_sha256=digest(__file__))
    write_json(OUT / 'checkpoint_continuation_verification.json', result)
    print('Verified saved cold and finalized idle prefixes for', len(rows), 'resumed seeds')


if __name__ == '__main__':
    main()
