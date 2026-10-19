# registry.cvib.ucla.edu/colivares:mrdpm-env-v1
#
# gan_bpm's job image plus the six packages MRDPM imports that it lacks (condor/probe_box.sh,
# ctp_only_plan.md §13 step 0). Built on the box, not the Mac (the base is linux/amd64 only):
#
#   docker build -t registry.cvib.ucla.edu/colivares:mrdpm-env-v1 - < Dockerfile
#
# No COPY, so no build context is needed. Never rebuild under the same tag; bump to -v2.
FROM registry.cvib.ucla.edu/colivares:pytorch-env-v2

# Only dropblock is actually called (BRAN's UNetUp dropout). einops, torchviz, pydot_ng and cv2
# are imported at module top level but unused. tensorboardX is what core/logger.py tries first.
# The constraints file makes pip fail rather than quietly move torch/numpy off the base's versions.
RUN python -c "import torch, torchvision, numpy; \
[print(f'{m.__name__}=={m.__version__}') for m in (torch, torchvision, numpy)]" > /tmp/base_pins.txt \
 && cat /tmp/base_pins.txt \
 && pip install --no-cache-dir -c /tmp/base_pins.txt \
        einops \
        dropblock \
        torchviz \
        pydot_ng \
        opencv-python-headless \
        tensorboardX

# Fail the build, not a job: the base's torch/CUDA are unchanged and every new import works.
RUN python -c "\
import torch, numpy; \
assert torch.__version__.startswith('2.5.1'), 'torch moved to ' + torch.__version__; \
assert torch.version.cuda == '12.4', 'CUDA moved to ' + str(torch.version.cuda); \
import einops, dropblock, torchviz, pydot_ng, cv2, tensorboardX; \
from dropblock import DropBlock2D; DropBlock2D(0.5, 3)(torch.rand(1, 4, 8, 8)); \
print('torch', torch.__version__, '| cuda', torch.version.cuda, '| numpy', numpy.__version__, \
      '| cv2', cv2.__version__, '| einops', einops.__version__, '| tensorboardX', tensorboardX.__version__)"
