# Shared task -> paths mapping for the Widar3.0 pipeline on LEAP2.
# Sourced (not run) by wr_submit.sh and wr_csi.sh:  source wr_env.sh har|gait
# har  : gesture recognition  (raw Widar3.0-HAR,  label = gesture)
# gait : gait identification  (raw Widar3.0-Gait, label = user)
WR_BASE="/mmfs1/home/urq23/txstpr4"

case "${1:?usage: source wr_env.sh har|gait}" in
  har)
    export WIDAR_TASK="har"
    export WIDAR_CSI="$WR_BASE/Widar3.0-HAR"
    export WIDAR_WORK="$WR_BASE/widar_csi_work_har"
    export WIDAR_OUT="$WR_BASE/widar_doppler_data_HAR"
    ;;
  gait)
    export WIDAR_TASK="gait"
    export WIDAR_CSI="$WR_BASE/Widar3.0-Gait"
    export WIDAR_WORK="$WR_BASE/widar_csi_work_gait"
    export WIDAR_OUT="$WR_BASE/widar_doppler_data_Gait"
    ;;
  *)
    echo "wr_env.sh: unknown task '$1' (want har|gait)" >&2
    return 1 2>/dev/null || exit 1
    ;;
esac
# 'raw' = Doppler from the coherent raw CSI (the SHARP sanitization cancels the
# common-mode motion Doppler on Intel 5300 -> flat spectrograms). Set
# WIDAR_REPR=sanitized to fall back to the old path.
export WIDAR_REPR="${WIDAR_REPR:-raw}"
