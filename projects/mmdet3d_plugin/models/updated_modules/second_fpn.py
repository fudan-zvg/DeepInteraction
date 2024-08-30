# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
from mmcv.cnn import build_conv_layer, build_norm_layer, build_upsample_layer
from mmcv.runner import BaseModule, auto_fp16
from torch import nn as nn
from mmdet3d.models.necks import SECONDFPN as BaseSECONDFPN
from mmdet3d.models.builder import NECKS
NECKS._module_dict.pop('SECONDFPN')


@NECKS.register_module()
class SECONDFPN(BaseSECONDFPN):

    @auto_fp16()
    def forward(self, x):
        assert len(x) == len(self.in_channels)
        ups = [deblock(x[i]) for i, deblock in enumerate(self.deblocks)]
        if len(ups) > 1:
            out = [torch.cat(ups, dim=1)]
            out.extend(ups)
        else:
            out = ups
        return out