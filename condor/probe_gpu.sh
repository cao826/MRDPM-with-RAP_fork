#!/bin/bash
# Invoked by probe_gpu.sub as: probe_gpu.sh HOST. Does a docker job that asked for a GPU on HOST
# see one? What Condor assigned, what the container was given, and what torch finds. Seconds.
HOST="${1:-?}"
echo "probe: GPU in docker on $HOST   hostname: $(hostname)   date: $(date -Is)"
echo "--- what Condor assigned (the slot's machine ad)"
grep -E '^(AssignedGPUs|GPUs|TotalGPUs|GPUs_DeviceName|DockerVersion|HasDocker) ' "${_CONDOR_MACHINE_AD:-/dev/null}" 2>/dev/null || echo "(no machine ad)"
echo "--- what the container got"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}   NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-<unset>}"
echo "/dev/nvidia*: $(ls /dev/nvidia* 2>/dev/null | tr '\n' ' ')"
nvidia-smi -L 2>&1 | head -5
echo "--- what torch finds"
python -c "import torch; n = torch.cuda.device_count(); print('torch', torch.__version__, 'CUDA', torch.version.cuda, '| devices', n, '|', [torch.cuda.get_device_name(i) for i in range(n)])" 2>&1 | tail -2
n=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 0)
echo "VERDICT gpu $HOST: torch sees ${n:-0} GPU(s)"
