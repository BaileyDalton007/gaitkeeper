# Simple script to make sure PyTorch is playing nice with the GPU.

import torch
print(torch.__version__)
print(torch.version.cuda)       # CUDA version PyTorch is built with
print(torch.backends.cudnn.version())  # cuDNN version
print(torch.cuda.is_available()) # should be True