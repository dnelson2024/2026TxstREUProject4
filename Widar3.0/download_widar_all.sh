#!/bin/bash
# Download + unzip the FULL Widar3.0 raw CSI release (~86 GB zipped, 21 zips,
# 15 collection dates) from the Tsinghua Seafile share. Run ON LEAP2 (login
# node is fine -- it's just wget) from anywhere:
#   bash Widar3.0/download_widar_all.sh
# Idempotent: wget -c resumes partial downloads; each zip gets a .done marker
# after a clean unzip and is then DELETED to give the ~86 GB back. Re-running
# skips finished zips, so a dropped connection just needs a re-run.
# Zip list enumerated from the share API on 2026-07-08 (share 2760bb9557ca4d09a74d).
set -uo pipefail

BASE_URL='https://cloud.tsinghua.edu.cn/d/2760bb9557ca4d09a74d/files/?p='
DEST="${WIDAR_CSI:-/mmfs1/home/urq23/txstpr4/Widar_CSI}"
mkdir -p "$DEST"
cd "$DEST"

failed=0
fetch() {  # fetch <remote path under /CSI, URL-encoded> <local zip name> <unzip dir>
    local rp="$1" zip="$2" dir="$3"
    if [ -e ".${zip}.done" ]; then echo "SKIP $zip (already done)"; return 0; fi
    echo "=== $zip -> $dir/"
    wget -c -q --show-progress -O "$zip" "${BASE_URL}%2FCSI%2F${rp}&dl=1" \
        || { echo "DOWNLOAD FAILED: $zip"; failed=1; return 1; }
    mkdir -p "$dir"
    unzip -q -n -d "$dir" "$zip" \
        || { echo "UNZIP FAILED: $zip (zip kept for inspection)"; failed=1; return 1; }
    touch ".${zip}.done"
    rm -f "$zip"
}

# date folders with per-user zips                                   size
fetch '20181109%2Fuser1.zip'      '20181109_user1.zip'  '20181109'  # 6.0G
fetch '20181109%2Fuser2.zip'      '20181109_user2.zip'  '20181109'  # 5.3G
fetch '20181109%2Fuser3.zip'      '20181109_user3.zip'  '20181109'  # 3.3G
fetch '20181121%2Fuser1.zip'      '20181121_user1.zip'  '20181121'  # 2.7G
fetch '20181121%2Fuser2.zip'      '20181121_user2.zip'  '20181121'  # 5.8G
fetch '20181121%2Fuser3.zip'      '20181121_user3.zip'  '20181121'  # 5.8G
# single-zip dates
fetch '20181112.zip'              '20181112.zip'        '20181112'  # 9.2G
fetch '20181115.zip'              '20181115.zip'        '20181115'  # 3.1G
fetch '20181116.zip'              '20181116.zip'        '20181116'  # 2.5G
fetch '20181117.zip'              '20181117.zip'        '20181117'  # 1.4G
fetch '20181118.zip'              '20181118.zip'        '20181118'  # 2.9G
fetch '20181127.zip'              '20181127.zip'        '20181127'  # 2.9G
fetch '20181128.zip'              '20181128.zip'        '20181128'  # 1.0G
fetch '20181204.zip'              '20181204.zip'        '20181204'  # 2.1G
fetch '20181205.zip'              '20181205.zip'        '20181205'  # 2.9G
fetch '20181208.zip'              '20181208.zip'        '20181208'  # 1.6G
fetch '20181209.zip'              '20181209.zip'        '20181209'  # 1.6G
fetch '20181211.zip'              '20181211.zip'        '20181211'  # 5.5G
# 20181130: three user groups, all into the same date dir
fetch '20181130_user5_10_11.zip'  '20181130_user5_10_11.zip'  '20181130'  # 6.5G
fetch '20181130_user12_13_14.zip' '20181130_user12_13_14.zip' '20181130'  # 6.6G
fetch '20181130_user15_16_17.zip' '20181130_user15_16_17.zip' '20181130'  # 7.0G

echo ""
if [ "$failed" -ne 0 ]; then
    echo "SOME DOWNLOADS FAILED -- re-run this script to resume/retry."
    exit 1
fi
echo "ALL 21 ZIPS DONE -> $DEST"
echo "dat file count:"; find "$DEST" -name '*.dat' | wc -l
echo "Next: rm -rf widar_csi_work widar_doppler_data  (stale pilot cache),"
echo "link the tree where the pipeline looks:  ln -s $DEST \$HOME/txstpr4/Widar3.0-HAR"
echo "then: bash Widar3.0/slurm/wr_submit.sh har"
