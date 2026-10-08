#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"

./tools/g1_commissioning/connect_g1_network.sh

echo "Authorize host performance setup in this terminal. The robot program itself will not run as root."
sudo -v

output_dir="evaluation/hardware_shadow/commissioning/arm_identification_$(date +%Y%m%d_%H%M%S)"
echo "Identification output: ${output_dir}"

exec taskset -c 0-17 /home/fjk/miniforge3/envs/g1_mpc/bin/python \
  tools/g1_commissioning/g1_arm_system_identification.py enx6c1ff701509c \
  --execute --cpu 7 --rt-priority 20 \
  --profile evaluation/hardware_shadow/commissioning/g1_pid_6ms_20261005_132052/arm_profile.conf \
  --output-dir "${output_dir}" --pid-6ms-validated \
  --permit-real-output ARM_ID_STATIONARY
