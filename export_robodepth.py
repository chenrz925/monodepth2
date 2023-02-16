import argparse
from typing import Any, Text
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import networks
from layers import disp_to_depth

from datasets.robodepth_eval_dataset import EvalDataset

class Predictor(object):
    def _inject_arguments(self) -> Any:
        self._parser = argparse.ArgumentParser()

        self._parser.add_argument(
            '--encoder_path', '-ep', type=str, required=True,
            help='Path to encoder model'
        )
        self._parser.add_argument(
            '--encoder_layer', '-el', type=int, default=18,
            help='Encoder layer'
        )
        self._parser.add_argument(
            '--decoder_path', '-dp', type=str, required=True,
            help='Path to decoder model'
        )
        self._parser.add_argument(
            '--data_path', '-Dp', type=str, required=True,
            help='Path to data'
        )
        self._parser.add_argument(
            '--output_path', '-Op', type=str, default='.',
            help='Path to output'
        )
        self._parser.add_argument(
            '--device', '-d', type=str, default='cuda:0',
            help='Device to use'
        )

        ns = self._parser.parse_args()
        for k, v in vars(ns).items():
            setattr(self, f'_{k}', v)
        
        return self
    
    def _load_models(self) -> Any:
        self._encoder = networks.ResnetEncoder(self._encoder_layer, False)
        encoder_available_keys = self._encoder.state_dict().keys()
        encoder_state_dict = torch.load(self._encoder_path, map_location=self._device)
        self._encoder.load_state_dict({k: v for k, v in encoder_state_dict.items() if k in encoder_available_keys})
        self._encoder = self._encoder.to(self._device)
        self._decoder = networks.DepthDecoder(self._encoder.num_ch_enc)
        self._decoder.load_state_dict(torch.load(self._decoder_path, map_location=self._device))
        self._decoder = self._decoder.to(self._device)

        return self
    
    def _build_dataloader(self) -> DataLoader:
        self._dataset = EvalDataset(self._data_path)
        self._dataloader = DataLoader(self._dataset, 1, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)

        return self
    
    def __init__(self) -> None:
        self._inject_arguments()._load_models()._build_dataloader()
    
    def __call__(self) -> Any:
        pred_disps = []

        with torch.no_grad():
            for data in self._dataloader:
                input_color = data.to(self._device)
                output = self._decoder(self._encoder(input_color))
                pred_disp, _ = disp_to_depth(output[("disp", 0)], 0.1, 100.0)
                pred_disp = pred_disp.cpu()[:, 0].numpy()
                pred_disps.append(pred_disp[0])

        output_path = Path(self._output_path) / 'disp.npz'
        np.savez_compressed(output_path, disp=pred_disps)
        output_path.rename(output_path.with_suffix('.zip'))

        return self

if __name__ == '__main__':
    predictor = Predictor()
    predictor()