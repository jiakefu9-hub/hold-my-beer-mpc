#!/usr/bin/env python3
"""Independent learned-forecast MPC entry; default is offline preflight only.

Shared transport/stop/release with the successful baseline. See
docs/g1_field_validation/HARDWARE_MPC_LEARNED.md. --predictor hold_current is
the same yaw-aware controller without learned forecasts, for a fair A/B test.
"""
from g1_walk_mpc import main as shared_main


def main(argv=None):
    return shared_main(argv, learned=True)


if __name__ == '__main__':
    raise SystemExit(main())
