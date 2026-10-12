# 2026-10-10 adaptive MPC simulation study

Simulation only. These tools never launch robot transport or send arm commands.
Final results and exact configurations are documented in
`docs/simulation/experiments/20261010_mpc_comprehensive/REPORT.md`.

- `study.py CASES.json --cpus 0,2,4,7 [--keep]` runs independent rollouts.
  Use the `g1_mpc` Python and build the C++ libraries first with
  `./cpp/build_runtime.sh`. The research batch explicitly labels its runtime
  preflight as non-real-time evidence. It does not disable physical contacts,
  torque constraints or solver acceptance checks.
- `simulation_entry.py`, `preview.py`, `strategy.py` preserve the isolated
  research ablations for reproducibility. They do not affect normal `run.sh`.
  The research preview was called once per consecutive controller observation;
  use the maintained predictor implementation for normal simulations.
- `metrics.py` computes matched [0, 8) s left/right metrics from physical states.
  Angular swing uses a normalized mean bottle axis, not scalar tilt variance.
  Linear speed is world speed; command-relative speed error is also recorded.
- `analyze.py` writes the complete ledger including failures, with an explicit
  screening rule and a ranking aid (not an assertion of a global optimum).
- `serial_validation.py CASES.json` runs three repeated formal `run.sh` trials
  on CPU 7 after all parallel batches have finished. Scheduler/governor and
  complete-interval timing must be reported, not only QP solve time.
- `contact_audit.py CASE_NAME ...` reconstructs contacts and physical arm joint
  limits at every 2 ms sample in the full scoring window.

The fixed evidence root is `evaluation/mpc_comprehensive_20261010/`. Existing
result names are never overwritten by runners. To rerun a batch, copy its case
list and give cases fresh names. Exact base YAML, case overrides, independent
initial-state manifests and result ledgers are retained in the report directory.
New poor rollouts retain compact states, metrics, diagnostics, configuration and
logs; redundant large arrays/plots are pruned only after metric extraction.
Older experimental data and hardware evidence are outside the cleanup scope.

Final candidate YAML files are simulation configurations. The real robot uses a
different H0 learned predictor and a coupled five-joint moving-base arm model;
these files are not compatible hardware launch configurations.
