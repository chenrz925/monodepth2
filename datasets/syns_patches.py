from pathlib import Path
from typing import Any, Text
from re import match

from torchvision.io import read_image, ImageReadMode

class EvalDataset(object):
    def __init__(self, root: Text, split: Text) -> None:
        self._root = Path(root)

        assert split in ['val', 'test']
        self._split = []
        for line in Path(f'{self._root}/splits/{split}_files.txt').read_text().splitlines():
            match_result = match(r'(\d+) (\d+.png)', line)
            scene_id, image_name = match_result.groups()
            self._split.append((scene_id, image_name))

    
    def __getitem__(self, index: int) -> Any:
        scene_id, image_name = self._split[index]
        image_path = self._root / scene_id / 'images' / image_name
        raw_image = read_image(str(image_path), ImageReadMode.RGB) / 255.
        return raw_image

    def __len__(self) -> int:
        return len(self._split)