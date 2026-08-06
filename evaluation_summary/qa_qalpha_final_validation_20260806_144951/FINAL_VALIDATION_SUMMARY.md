# Final MPC QA/Qalpha validation

- Validation ID: `qa_qalpha_final_validation_20260806_144951`
- Planned runs: 30
- Completed runs: 30
- Repeats per candidate: 5
- Candidate order was cyclically rotated between repeats.

## Selection policy

Hard gates:
- QP success in every repeat >= 99.0%
- Tilt RMS in every repeat <= 0.030 rad
- DDQ saturation in every repeat <= 10.0%
- Joint safety-box violations = 0
- Host timing overruns are reported separately and are not used to rank controller weights.

Task eligibility:
- Linear-acceleration improvement over baseline >= 5.0%
- Angular-acceleration improvement over baseline >= 5.0%
- Mean tilt RMS <= 0.020 rad

Task-priority score weights:
- tilt RMS: 45%
- linear acceleration RMS: 25%
- angular acceleration RMS: 20%
- DDQ saturation: 10%

## Candidate means

| candidate | QA | Qalpha | n | acc RMS | alpha RMS | tilt RMS rad | DDQ sat | QP | acc imp. | alpha imp. | eligible | score |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|---:|
| baseline | 0 | 0 | 5 | 2.9285 ± 0.0000 | 6.5195 ± 0.0000 | 0.01511 ± 0.00000 | 1.21% | 99.62% | 0.00% | 0.00% | no | NA |
| posture_floor | 0.0003 | 0 | 5 | 2.9189 ± 0.0000 | 6.4823 ± 0.0000 | 0.01468 ± 0.00000 | 1.21% | 100.00% | 0.33% | 0.57% | no | NA |
| final_candidate | 0.01 | 0.0005 | 5 | 2.6014 ± 0.0000 | 5.9426 ± 0.0000 | 0.01634 ± 0.00000 | 2.13% | 100.00% | 11.17% | 8.85% | yes | 0.5328 |
| angular_priority | 0.01 | 0.0015 | 5 | 2.6281 ± 0.0000 | 5.6170 ± 0.0000 | 0.01788 ± 0.00000 | 4.96% | 99.91% | 10.26% | 13.84% | yes | 0.6917 |
| balanced_auto | 0.015 | 0.0015 | 5 | 2.5647 ± 0.0000 | 5.6895 ± 0.0000 | 0.01896 ± 0.00000 | 5.42% | 100.00% | 12.42% | 12.73% | yes | 0.7483 |
| linear_priority | 0.02 | 0.0005 | 5 | 2.4738 ± 0.0000 | 6.1952 ± 0.0000 | 0.01899 ± 0.00000 | 3.35% | 100.00% | 15.53% | 4.97% | no | NA |

## Selected parameters

- Candidate: **final_candidate**
- `mpc_q_ee_acc: 0.01`
- `mpc_q_ee_alpha: 0.0005`
- Task score: 0.53278
- Mean tilt RMS: 0.016338 rad
- Mean linear acceleration RMS: 2.601397
- Mean angular acceleration RMS: 5.942628
- Mean DDQ saturation: 2.135%

The score is a task-oriented ranking within this candidate set, not a proof of a global optimum.
