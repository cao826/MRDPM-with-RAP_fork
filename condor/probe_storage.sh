#!/bin/bash
# Invoked by probe_storage.sub (MODE docker: inside our image, what our jobs see) and by
# probe_storage_host.sub (MODE vanilla: on the box itself, no container) as:
#   probe_storage.sh MODE HOST
# What storage can a job on HOST reach, which of it can it write, and how much room is there?
# It creates and removes one empty test directory in each place it tries. Under a minute.

MODE="${1:-docker}"
HOST="${2:-?}"
W="${_CONDOR_SCRATCH_DIR:-$PWD}"
echo "probe: $MODE on $HOST   hostname: $(hostname)   date: $(date -Is)"
echo "user: $(id)"
echo "workspace: $W"
echo "==================================================================="
echo "--- mounts (pseudo filesystems left out)"
grep -vE '^(proc|sysfs|cgroup2?|devpts|mqueue|tmpfs|shm|devtmpfs|securityfs|debugfs|tracefs|pstore|bpf|configfs|fusectl|hugetlbfs|autofs|binfmt_misc|nsfs|overlay|rpc_pipefs|systemd-1) ' /proc/mounts \
    | awk '{printf "%-45s %-10s %s\n", $2, $3, $1}' | sort

echo "--- places a job could keep data: exists? owner? free space?"
for d in /home /home/colivares /raid /raid/colivares /radraid /radraid/colivares /radraid2 /radraid2/colivares \
         /data /data/colivares /scratch /scratch/colivares /local /local/colivares /ssd /nvme /mnt /tmp "$W"; do
    if [ -d "$d" ]; then
        printf '%-28s %s\n' "$d" "$(ls -ld "$d" | awk '{print $1, $3, $4}')   $(df -h "$d" | tail -1 | awk '{print "size", $2, "free", $4, "on", $6}')"
    else
        printf '%-28s absent\n' "$d"
    fi
done
[ -d /mnt ] && echo "/mnt holds: $(ls /mnt 2>/dev/null | tr '\n' ' ')"

echo "--- can a job write there? (outside the workspace, so it would outlive the job)"
writable=()
for d in /home/colivares /raid/colivares /radraid2/colivares /radraid/colivares /data/colivares /scratch/colivares \
         /local/colivares /data /scratch /local /ssd /nvme; do
    [ -d "$d" ] || continue
    t="$d/.condor_write_test_$$"
    if mkdir "$t" 2>/dev/null; then
        echo "writable:     $d"
        rmdir "$t"
        writable+=("$d")
    else
        echo "not writable: $d"
    fi
done

echo "--- this job's slot (Condor's machine ad): disk is in KB"
if [ -n "${_CONDOR_MACHINE_AD:-}" ] && [ -f "$_CONDOR_MACHINE_AD" ]; then
    grep -E '^(Disk|TotalDisk|TotalSlotDisk|Cpus|Memory|HasDocker|DockerVersion) ' "$_CONDOR_MACHINE_AD"
else
    echo "(no machine ad)"
fi

echo "--- copy speed, /radraid2 -> workspace"
src=/radraid2/colivares/mrdpm_ctp_only/probe_bundle/batch.pt
if [ -f "$src" ]; then
    t0=$(date +%s%N); cp "$src" "$W/copy_test.pt"; t1=$(date +%s%N)
    mb=$(du -m "$W/copy_test.pt" | cut -f1); ms=$(( (t1 - t0) / 1000000 ))
    echo "$mb MB in $ms ms: $(( mb * 1000 / (ms > 0 ? ms : 1) )) MB/s"
    rm -f "$W/copy_test.pt"
else
    echo "absent or unreadable: $src"
fi

echo "==================================================================="
echo "VERDICT $MODE $HOST: /home/colivares $([ -d /home/colivares ] && echo seen || echo NOT-SEEN), /radraid2 $([ -d /radraid2/colivares ] && echo seen || echo NOT-SEEN), workspace free $(df -h "$W" | tail -1 | awk '{print $4}'), writable outside the workspace: ${writable[*]:-none}"
