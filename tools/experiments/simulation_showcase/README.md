# Selected simulation demonstrations

Three simulation-only cases and portable recorded replays. No robot output.
Optional `D01_R50`-`D03_R50` replays preserve the left-shoulder-roll stiffness
sensitivity experiment without replacing the three primary cases.

See [selection, measured results, configuration differences and commands](../../../docs/simulation/experiments/20261103_showcase/README.md).

```bash
/home/fjk/miniforge3/envs/g1_mpc/bin/python tools/experiments/simulation_showcase/replay.py D03
```

`replay.py` only loads recorded states and updates rendering. `run_study.py`
reuses the existing simulation comparison/ablation entry. `package.py` verifies
the matched MPC configuration pair and exports compact, hashed replay files.
