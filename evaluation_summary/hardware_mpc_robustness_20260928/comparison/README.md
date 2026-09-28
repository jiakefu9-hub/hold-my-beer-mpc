# Offline torque robustness comparison

Scenario: `combined`. Input summary/NPZ consistency checks passed.

RMSE pools five joints over active-command 2ms intervals, excluding the initial hold.
Timing is saved controller-step wall time, not end-to-end DDS latency or real-time proof.

## Entire observed data (different durations; not a matched-window ranking)

| Run | Outcome | Observed / requested s | Active RMSE rad/s² | Outer margin ° | Max acceleration rad/s² | Core p99 ms |
|---|---|---:|---:|---:|---:|---:|
| baseline | FAILED PREFIX ONLY | 2.484 / 5 | 1.16427 | -0.00522457 | 9.24422 | 2.32737 |
| guard_only | FAILED PREFIX ONLY | 0.114 / 5 | 1.7015 | 1 | 9.24422 | 2.74421 |
| compensated | COMPLETE | 20 / 20 | 0.92806 | 0.264218 | 9.05582 | 4.3503 |
| delay_only | COMPLETE | 5 / 5 | 0.900525 | 0.001513 | 9.05582 | 4.54967 |
| assumed4 | COMPLETE | 5 / 5 | 0.969124 | 0.250263 | 9.11606 | 6.475 |
| assumed8 | COMPLETE | 5 / 5 | 0.834805 | 0.283728 | 9.17566 | 6.56414 |

## Common prefix of all selected runs

Window: 0 ≤ t < 0.114s.

| Run | Active RMSE rad/s² | Active samples | Outer margin ° | Core p99 ms |
|---|---:|---:|---:|---:|
| baseline | 1.7015 | 54 | 1 | 2.55297 |
| guard_only | 1.7015 | 54 | 1 | 2.74421 |
| compensated | 0.712863 | 54 | 1 | 4.26134 |
| delay_only | 0.712863 | 54 | 1 | 5.23323 |
| assumed4 | 0.78198 | 54 | 1 | 6.46416 |
| assumed8 | 0.715806 | 54 | 1 | 6.46663 |

## Explicit comparison window

Window: 0 ≤ t < 2.4s.

| Run | Active RMSE rad/s² | Active samples | Outer margin ° | Core p99 ms |
|---|---:|---:|---:|---:|
| baseline | 1.12878 | 1197 | 0.0866109 | 2.32902 |
| compensated | 0.842074 | 1197 | 0.304917 | 4.94816 |
| delay_only | 0.842005 | 1197 | 0.00272122 | 4.64025 |
| assumed4 | 0.88978 | 1197 | 0.315381 | 6.40217 |
| assumed8 | 0.802533 | 1197 | 0.373456 | 6.33392 |

Excluded from this window (retained in the full report):

- guard_only: observed only 0.114s; shorter than requested window.

## Failure reasons

- baseline: HardwareMpcError: measured-state MPC rejected: daqp_exitflag_-1
- guard_only: HardwareMpcError: measured-state MPC rejected: daqp_exitflag_-1

Do not interpret a short surviving prefix as completion. The plotted ±5° shoulder bounds are this study’s original model bounds, not commissioned physical limits.

Exact input hashes, counts, full-duration/null fields, and comparison values: `comparison.json`.
