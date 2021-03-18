import os
import warnings
from pathlib import Path
from typing import List, Dict, Tuple, Union, Optional
import csv
from torch import Tensor
from torch.utils.data import Dataset

import torchaudio


def load_audio(line: List[str],
                          header: List[str],
                          path: str) -> Tuple[Tensor, int, Dict[str, str]]:
    # Each line as the following data:
    # client_id, path, sentence, up_votes, down_votes, age, gender, accent
    assert header[1] == "path"
    filename = os.path.join(path, line[1])
    waveform, sample_rate = torchaudio.load(filename)
    dic = dict(zip(header, line))
    return waveform, sample_rate, dic

class SimClrTestDataset(Dataset):
    def __init__(self,
                 root: Union[str, Path],
                 tsv: str = "test.tsv") -> None:
        self._path = os.fspath(root)
        self._tsv = os.path.join(self._path, tsv)

        with open(self._tsv, "r") as tsv_:
            walker = csv.reader(tsv_, delimiter="\t")
            self._header = next(walker)
            self._walker = list(walker)

    def __getitem__(self, n: int) -> Tuple[Tensor, int, Dict[str, str]]:
        line = self._walker[n]
        return load_audio(line, self._header, self._path)

    def __len__(self) -> int:
        return len(self._walker)

if __name__ == "__main__":
    from utils.config import config
    loader = SimClrTestDataset(root=config.dataset.test_root, tsv=config.dataset.test)
    for i in range(len(loader)):
        example = loader[i]
        print(example[0].shape, example[1], example[2])