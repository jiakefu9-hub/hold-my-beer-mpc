# Frozen H0 disturbance model

`bank.npz` is a portable export of the already-fitted 2026-09-25 innovation
lookup. `manifest.json` records source hashes, development/comparison runs,
preprocessing, units and target semantics. No raw recordings are included.

The runtime checks the bank SHA256 before building its KD-tree. It never trains
online. Regeneration requires the local research artifacts and
`tools/g1_commissioning/export_hardware_mpc_predictor.py`; the exporter refuses
to overwrite an existing bank. Do not regenerate merely to change a file date.

This predicts 15 Hz causal filtered signals, not instantaneous raw impacts.
See [hardware MPC guide](../../docs/g1_field_validation/HARDWARE_MPC.md) for
fixed H0, interval/node semantics, filter lag and field-validation limitations.
