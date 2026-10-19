#!/bin/bash
# Invoked by probe_h100_storage.sub. Can a job on this machine reach the network shares
# (/radraid, /radraid2) and the machine's own 84 TB home volume, write to them, and fit the dataset in its own workspace? And how
# fast is a copy from /radraid2 into that workspace?
set -u

W="${_CONDOR_SCRATCH_DIR:-$PWD}"
echo "host: $(hostname)   date: $(date -Is)   user: $(id)"
echo "workspace: $W"
echo "==================================================================="
echo "--- mounts that look like data storage"
grep -E ' /(raid|radraid|data|scratch|mnt|home)' /proc/mounts || echo "(none matched)"

echo "--- directories: exists? owner? free space?"
for d in /radraid /radraid/colivares /radraid2 /radraid2/colivares /raid /home /home/colivares "$W"; do
    if [ -d "$d" ]; then
        echo "$(ls -ld "$d")"
        df -h "$d" | tail -1
    else
        echo "absent: $d"
    fi
done

echo "--- can a job write results back?"
for d in /radraid2/colivares /radraid/colivares /home/colivares; do
    [ -d "$d" ] || continue
    t="$d/.condor_write_test_$$"
    if mkdir "$t" 2>/dev/null; then
        echo "writable: $d"
        rmdir "$t"
    else
        echo "not writable: $d"
    fi
done

echo "--- copy speed, /radraid2 -> workspace (the probe bundle's 32 slices, about 200 MB)"
src=/radraid2/colivares/mrdpm_ctp_only/probe_bundle/batch.pt
if [ -f "$src" ]; then
    t0=$(date +%s%N)
    cp "$src" "$W/copy_test.pt"
    t1=$(date +%s%N)
    mb=$(du -m "$W/copy_test.pt" | cut -f1)
    ms=$(( (t1 - t0) / 1000000 ))
    echo "$mb MB in $ms ms: $(( mb * 1000 / (ms > 0 ? ms : 1) )) MB/s"
    rm -f "$W/copy_test.pt"
else
    echo "absent: $src"
fi

echo "--- tools in the image"
for c in rsync cp tar; do
    printf '%-6s %s\n' "$c" "$(command -v "$c" || echo '<missing>')"
done
echo "==================================================================="
