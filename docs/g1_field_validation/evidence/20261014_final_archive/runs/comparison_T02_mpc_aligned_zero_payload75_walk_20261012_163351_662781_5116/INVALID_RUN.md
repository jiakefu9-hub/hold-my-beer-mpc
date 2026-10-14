# Invalid field run

- Run: `comparison_T02_mpc_aligned_zero_payload75_walk_20261012_163351_662781_5116`
- Status: invalid; exclude the entire run from controller evaluation and all aggregate comparisons.
- Reason: the operator reported that the robot was tripped/interfered with by a cable during the run.
- Data handling: retain the raw capture for traceability; do not use its endpoint or timing metrics to assess `--align-accel-forecast`.
- Recorded: 2026-10-12 (Europe/Berlin).
