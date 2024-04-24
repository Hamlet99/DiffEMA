import itertools
import math
import os
import pickle
import random
from argparse import Namespace
from functools import partial
import copy

import numpy as np
import pandas as pd
import torch
import yaml
from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from source.datasets.general_dataset import GeneralDataset
from source.utils.diffusion_utils import get_t_schedule
#from source.utils.sampling import randomize_position, sampling
from source.utils.general_utils import get_model
from source.utils.diffusion_utils import t_to_sigma as t_to_sigma_compl


class ListDataset(Dataset):
    def __init__(self, list):
        super().__init__()
        self.data_list = list

    def len(self) -> int:
        return len(self.data_list)

    def get(self, idx: int) -> Data:
        return self.data_list[idx]
