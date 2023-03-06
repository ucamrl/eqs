import torch
import torch_geometric as pyg
from torch_geometric.data import Dataset
import os.path as osp
from .lib import Language
import os


class PretrainingDataset(Dataset):
    def __init__(self, lang: Language, root: str, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.root_dir = root
        self.feature_names = lang.feature_names

    @property
    def raw_file_names(self):
        files = os.listdir(self.root)
        files.sort()
        return files

    def len(self):
        return len(self.raw_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.root_dir, self.raw_file_names[idx]))
        return data

