# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

from .multidist_meta_arch import MultiDistillationMetaArch
from .cam_meta_arch import CAMMetaArch
from .train import get_args_parser, main
