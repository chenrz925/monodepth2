from argparse import ArgumentParser
from typing import Any, Text
from copy import deepcopy

from imagecorruptions import corrupt
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.distributions import Normal
from torchvision.transforms import ToPILImage
from matplotlib import pyplot as plt

from datasets import KITTIRAWDataset
from utils import readlines
import networks
from layers import disp_to_depth


class CorruptionsAnalysis(object):
    def _parse_arguments(self) -> Any:
        parser = ArgumentParser()
        
        parser.add_argument('--encoder_path', '-ep', type=str, required=True, help='Path to encoder model')
        parser.add_argument('--decoder_path', '-dp', type=str, required=True, help='Path to decoder model')
        parser.add_argument('--data_path', '-Dp', type=str, required=True, help='Path to data')
        parser.add_argument('--device', '-d', type=str, default='cuda:0',help='Device to use')
        parser.add_argument('--encoder_layer', '-el', type=int, default=18, help='Encoder layer')

        ns = parser.parse_args()
        for k, v in vars(ns).items():
            setattr(self, f'_{k}', v)

        return self
    
    def _inject_dataloader(self) -> Any:
        dataset = KITTIRAWDataset(self._data_path, readlines('splits/eigen_zhou/train_files.txt'), 192, 640, [0, -1, 1], 4, is_train=True, img_ext='.png')
        self._dataloader = DataLoader(dataset, 1, shuffle=True, num_workers=1, pin_memory=True, drop_last=False)

        return self

    def _inject_encoder(self) -> Any:
        self._encoder = networks.ResnetEncoder(self._encoder_layer, False)
        self._encoder.eval()
        encoder_available_keys = self._encoder.state_dict().keys()
        encoder_state_dict = torch.load(self._encoder_path, map_location=self._device)
        self._encoder.load_state_dict({k: v for k, v in encoder_state_dict.items() if k in encoder_available_keys})
        self._encoder = self._encoder.to(self._device)
        self._decoder = networks.DepthDecoder(self._encoder.num_ch_enc)
        self._decoder.load_state_dict(torch.load(self._decoder_path, map_location=self._device))
        self._decoder = self._decoder.to(self._device)

        return self

    def __init__(self) -> None:
        self._parse_arguments()._inject_dataloader()._inject_encoder()

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        plt.ion()
        fig, ax = plt.subplots(5, 2)
        generator = Normal(0., 1.)
        for i, data in enumerate(self._dataloader):
            for c in range(19):
                raw_img = data[('color', 0, 0)][0]
                cor_img = corrupt((raw_img * 255.).to(torch.uint8).permute(1, 2, 0).numpy(), 5, corruption_number=c)
                raw_img = raw_img.to(torch.float32).to(self._device)
                cor_img = torch.from_numpy(cor_img / 255.).permute(2, 0, 1).to(torch.float32).to(self._device)
                with torch.no_grad():
                    raw_feat = self._encoder(raw_img.unsqueeze(0))
                    cor_feat = self._encoder(cor_img.unsqueeze(0))
                    raw_depth = self._decoder(raw_feat)[('disp', 0)]
                    raw_depth, _ = disp_to_depth(raw_depth, 0.1, 100.0)
                    raw_depth = raw_depth[0, 0].cpu().numpy()
                    cor_depth = self._decoder(cor_feat)[('disp', 0)][0, 0]
                    cor_depth, _ = disp_to_depth(cor_depth, 0.1, 100.0)
                    cor_depth = cor_depth.cpu().numpy()
                    raw_heatmap = raw_feat[-1][0].mean(0).cpu().numpy()
                    cor_heatmap = cor_feat[-1][0].mean(0).cpu().numpy()
                    noise = generator.sample(raw_feat[-1].shape)
                    noise = 11 * F.avg_pool2d(noise, 5, padding=2, stride=1) - 10 * noise 
                    noise = noise[0]
                    noi_feat = deepcopy(raw_feat)
                    noi_feat[-1][0] = noi_feat[-1][0] + noise.to(self._device)
                    noi_depth = self._decoder(noi_feat)[('disp', 0)][0, 0]
                    noi_depth, _ = disp_to_depth(noi_depth, 0.1, 100.0)
                    noi_depth = noi_depth.cpu().numpy()
                    ax[0, 0].imshow((raw_img * 255.).to(torch.uint8).permute(1, 2, 0).cpu().numpy())
                    ax[0, 1].imshow((cor_img * 255.).to(torch.uint8).permute(1, 2, 0).cpu().numpy())
                    ax[1, 0].matshow(raw_heatmap)
                    ax[1, 1].matshow(cor_heatmap)
                    ax[2, 0].matshow(raw_depth)
                    ax[2, 1].matshow(cor_depth)
                    ax[3, 0].matshow(raw_depth - cor_depth)
                    ax[3, 1].matshow(raw_heatmap - cor_heatmap)
                    ax[4, 0].matshow(noi_feat[-1][0].mean(0).cpu().numpy())
                    ax[4, 1].matshow(noi_depth)

                    plt.show()
                    plt.pause(0.1)
                


if __name__ == '__main__':
    CorruptionsAnalysis()()