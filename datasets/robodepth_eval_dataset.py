from pathlib import Path
from typing import Any, Text

from torchvision.io import read_image, ImageReadMode

class EvalDataset(object):
    def __init__(self, root: Text) -> None:
        self._root = Path(root)
    
    def __getitem__(self, index: int) -> Any:
        image_path = self._root / f'sample_{index:0>3}.png'
        raw_image = read_image(str(image_path), ImageReadMode.RGB) / 255.
        return raw_image

    def __len__(self) -> int:
        return len(list(self._root.glob('*.png')))