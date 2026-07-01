# ============================================================================
# Shared configuration for the SHARP non-network pipeline jobs on LEAP2.
# Every job script sources this file. EDIT THE VALUES IN THIS SECTION ONCE.
# ============================================================================

# --- Cluster / SLURM ---------------------------------------------------------
# Run `sinfo` on LEAP2 to list partitions, then put the CPU partition here.
export LEAP2_PARTITION="shared"           # CPU partition on LEAP2 (90-day limit, has idle nodes)
export LEAP2_ACCOUNT="urq23@leap2.txstate.edu"                   # optional: set if LEAP2 requires -A <account>

# --- Python environment (conda) ----------------------------------------------
# conda.sh from your anaconda install, and the name of the env that has
# tensorflow, numpy, scipy, scikit-learn, osqp installed.
export CONDA_SH="/mmfs1/home/urq23/anaconda3/etc/profile.d/conda.sh"   # <-- path to conda.sh
export CONDA_ENV="p4"                                                  # <-- conda env name

# --- Project layout ----------------------------------------------------------
# Directory that contains the SHARP Python scripts.
export PYTHON_CODE_DIR="/mmfs1/home/urq23/txstpr4"   # <-- CHANGE if different on LEAP2

# Directory holding the raw input .mat capture folders (S1a, S1b, ...).
export INPUT_DIR="/mmfs1/home/urq23/txstpr4/input_files"   # <-- CHANGE ME to your data path

# Intermediate / output directories. Stages 01/02 hard-code output to
# ./phase_processing (relative to PYTHON_CODE_DIR), so these MUST live under
# PYTHON_CODE_DIR. Absolute paths avoid any CWD confusion.
export PHASE_PROCESSING_DIR="$PYTHON_CODE_DIR/phase_processing/"
export PROCESSED_PHASE_DIR="$PYTHON_CODE_DIR/processed_phase/"
export DOPPLER_DIR="$PYTHON_CODE_DIR/doppler_traces/"

# Where the plot scripts save images. The SHARP plot scripts hard-code "./plots/"
# (relative to PYTHON_CODE_DIR); the plot jobs copy the results into IMAGE_DIR
# below so everything ends up in one collected folder.
export IMAGE_DIR="$PYTHON_CODE_DIR/plots"

# --- Dataset parameters (from the SHARP README examples) ---------------------
# Capture sub-directories to process.
export SUBDIRS_ALL="S1a,S1b,S1c,S2a,S2b,S3a,S4a,S4b,S5a,S6a,S6b,S7a"
export SUBDIRS_TRAIN="S1a,S1b,S1c"
export SUBDIRS_TEST="S2a,S2b,S3a,S4a,S4b,S5a,S6a,S6b,S7a"

export NSS=1                 # number of spatial streams
export N_RX=4                # number of receive antennas (the scripts' "ncore" arg --
                             # a DATA dimension, NOT a CPU count; leave at 4 for SHARP)
export N_TOT=4               # number of streams * number of antennas
export ACTIVITIES="E,L,W,R,J"

# Doppler computation params
export DOPPLER_START=800
export DOPPLER_END=800
export SAMPLE_LENGTH=31
export SLIDING=1
export NOISE_LEVEL=-1.2

# Dataset windowing params
export WINDOW_LENGTH=340
export STRIDE=30

# ============================================================================
# Helper: activate environment. Sourced by every job script — do not edit
# below unless your cluster needs a different activation method.
# ============================================================================
activate_env() {
    # If LEAP2 requires a module to expose python first, uncomment & adjust:
    # module load python

    # Headless plotting: force matplotlib's non-interactive backend so scripts
    # that import matplotlib (incl. plots_utility) work on compute nodes with
    # no display. Without this, importing pyplot can fail on a batch node.
    export MPLBACKEND=Agg

    # Keep numpy/scipy/BLAS thread pools in line with the requested cores.
    export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
    export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
    export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

    # Activate the conda environment. (conda envs have no bin/activate, so we
    # source conda.sh and use `conda activate` rather than a venv activate script.)
    if [ -f "$CONDA_SH" ]; then
        # shellcheck disable=SC1090
        source "$CONDA_SH"
        conda activate "$CONDA_ENV"
    else
        echo "ERROR: conda.sh not found at $CONDA_SH" >&2
        exit 1
    fi
    cd "$PYTHON_CODE_DIR" || { echo "ERROR: cannot cd to $PYTHON_CODE_DIR" >&2; exit 1; }
}
