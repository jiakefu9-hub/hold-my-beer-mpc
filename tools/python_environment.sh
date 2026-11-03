#!/usr/bin/env bash
# Source from simulation launch/build scripts. Resolve one environment for both.
if [[ -z "${G1_MPC_PYTHON:-}" ]]; then
    if [[ "${CONDA_DEFAULT_ENV:-}" == "${CONDA_ENV_NAME:-g1_mpc}" && -n "${CONDA_PREFIX:-}" ]]; then
        G1_MPC_PYTHON="$CONDA_PREFIX/bin/python"
    else
        g1_conda="${G1_MPC_CONDA:-${CONDA_BIN:-${CONDA_EXE:-}}}"
        if [[ -z "$g1_conda" ]]; then
            g1_conda="$(command -v conda || true)"
        fi
        if [[ -z "$g1_conda" && -x "$HOME/miniforge3/bin/conda" ]]; then
            g1_conda="$HOME/miniforge3/bin/conda"
        fi
        if [[ -n "$g1_conda" ]]; then
            G1_MPC_PYTHON="$("$g1_conda" run -n "${CONDA_ENV_NAME:-g1_mpc}" python -c 'import sys; print(sys.executable)')"
        else
            G1_MPC_PYTHON="$(command -v python3 || true)"
        fi
    fi
fi
if [[ ! -x "$G1_MPC_PYTHON" ]]; then
    echo 'Set G1_MPC_PYTHON to the Python executable in the project environment.' >&2
    exit 2
fi
export G1_MPC_PYTHON
