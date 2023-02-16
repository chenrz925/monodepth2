# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision.models as models
from torchvision.models.convnext import CNBlockConfig, ConvNeXt
import torch.utils.model_zoo as model_zoo


class ConvNextMultiImageInput(ConvNeXt):
    """Constructs a convnext model with varying number of input images.
    Adapted from https://github.com/pytorch/vision/blob/master/torchvision/models/convnext.py
    """
    def __init__(self, block_setting, num_input_images=1, **kwargs):
        super(ConvNextMultiImageInput, self).__init__(block_setting, **kwargs)

        self.features[0][0] = nn.Conv2d(num_input_images * 3, 96, kernel_size=4, stride=4, padding=0)
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


def convnext_multiimage_input(num_layers, pretrained=False, num_input_images=1):
    """Constructs a ResNet model.
    Args:
        num_layers (int): Number of resnet layers. Must be 18 or 50
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        num_input_images (int): Number of frames stacked as input
    """
    assert num_layers in [18, 34, 50, 101], "Can only run with 18 or 50 layer resnet"
    block_setting = {
        18: [
            CNBlockConfig(96, 192, 3),
            CNBlockConfig(192, 384, 3),
            CNBlockConfig(384, 768, 9),
            CNBlockConfig(768, None, 3),
        ],
        34: [
            CNBlockConfig(96, 192, 3),
            CNBlockConfig(192, 384, 3),
            CNBlockConfig(384, 768, 27),
            CNBlockConfig(768, None, 3),
        ],
        50: [
            CNBlockConfig(128, 256, 3),
            CNBlockConfig(256, 512, 3),
            CNBlockConfig(512, 1024, 27),
            CNBlockConfig(1024, None, 3),
        ],
        101: [
            CNBlockConfig(192, 384, 3),
            CNBlockConfig(384, 768, 3),
            CNBlockConfig(768, 1536, 27),
            CNBlockConfig(1536, None, 3),
        ]
    }
    stochastic_depth_prob = {18: 0.1, 34: 0.4, 50: 0.5, 101: 0.5}
    model_urls = {
        18: models.ConvNeXt_Tiny_Weights.DEFAULT.url,
        34: models.ConvNeXt_Small_Weights.DEFAULT.url,
        50: models.ConvNeXt_Base_Weights.DEFAULT.url,
        101: models.ConvNeXt_Large_Weights.DEFAULT.url
    }

    model = ConvNextMultiImageInput(
        block_setting[num_layers], stochastic_depth_prob=stochastic_depth_prob[num_layers], num_input_images=num_input_images
    )

    if pretrained:
        loaded = model_zoo.load_url(model_urls[num_layers])
        loaded['features.0.0.weight'] = torch.cat(
            [loaded['features.0.0.weight']] * num_input_images, 1) / num_input_images
        model.load_state_dict(loaded)
    return model


class ConvNeXtEncoder(nn.Module):
    """Pytorch module for a resnet encoder
    """
    def __init__(self, num_layers, pretrained, num_input_images=1):
        super(ConvNeXtEncoder, self).__init__()
        self.num_ch_enc = np.array({
            18: [96, 192, 384, 768],
            34: [96, 192, 384, 768],
            50: [128, 256, 512, 1024],
            101: [192, 384, 768, 1536]
        }[num_layers])

        networks = {
            18: models.convnext_tiny,
            34: models.convnext_small,
            50: models.convnext_base,
            101: models.convnext_large,
        }

        if num_layers not in networks:
            raise ValueError("{} is not a valid number of resnet layers".format(num_layers))

        if num_input_images > 1:
            self.encoder = convnext_multiimage_input(num_layers, pretrained, num_input_images)
        else:
            self.encoder = networks[num_layers](pretrained)

        if num_layers > 34:
            self.num_ch_enc[1:] *= 4

    def forward(self, input_image):
        self.features = []
        x = (input_image - 0.45) / 0.225
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        self.features.append(self.encoder.features[0](x))
        self.features.append(self.encoder.features[1:3](self.features[-1]))
        self.features.append(self.encoder.features[3:5](self.features[-1]))
        self.features.append(self.encoder.features[5:7](self.features[-1]))

        return self.features
