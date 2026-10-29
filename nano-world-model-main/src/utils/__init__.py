# This project package is imported as ``utils``. DreamSim's DINO backbone also
# imports ``trunc_normal_`` from a top-level ``utils`` module, so re-export the
# PyTorch implementation to keep that hub-loaded module compatible.
from torch.nn.init import trunc_normal_

__all__ = ["trunc_normal_"]
