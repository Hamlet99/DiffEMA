from collections.abc import Mapping, Sequence
from typing import List, Optional, Union

import torch.utils.data
from torch.utils.data.dataloader import default_collate

from torch_geometric.data import Batch, Dataset
from torch_geometric.data.data import BaseData


class Collater:
    """
    A collater class which merges data objects.
    """
    def __init__(self, follow_batch, exclude_keys):
        """
        Initializes a new Collater object.
        :param follow_batch: List of keys to recursively follow batch assignment.
        :type follow_batch: List[str]
        :param exclude_keys: List of keys to exclude from batch assignment.
        :type exclude_keys: List[str]
        """
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

    def __call__(self, batch):
        """
        Method to merge a list of data objects to a mini-batch.
        :param batch: List of data objects.
        :type batch: List[BaseData]
        :return: Merged data objects.
        :rtype: Batch
        """
        batch = [x for x in batch if x is not None]
        elem = batch[0]
        if isinstance(elem, BaseData):
            return Batch.from_data_list(batch, self.follow_batch,
                                        self.exclude_keys)
        elif isinstance(elem, torch.Tensor):
            return default_collate(batch)
        elif isinstance(elem, float):
            return torch.tensor(batch, dtype=torch.float)
        elif isinstance(elem, int):
            return torch.tensor(batch)
        elif isinstance(elem, str):
            return batch
        elif isinstance(elem, Mapping):
            return {key: self([data[key] for data in batch]) for key in elem}
        elif isinstance(elem, tuple) and hasattr(elem, '_fields'):
            return type(elem)(*(self(s) for s in zip(*batch)))
        elif isinstance(elem, Sequence) and not isinstance(elem, str):
            return [self(s) for s in zip(*batch)]

        raise TypeError(f'DataLoader found invalid type: {type(elem)}')

    def collate(self, batch):  # Deprecated...
        return self(batch)


class DataLoader(torch.utils.data.DataLoader):
    """
    A data loader which merges data objects from a`torch_geometric.data.Dataset` class to a mini-batch.
    Data objects can be either of type `~torch_geometric.data.Data` or `~torch_geometric.data.HeteroData` class.
    """

    def __init__(
        self,
        dataset: Union[Dataset, List[BaseData]],
        batch_size: int = 1,
        shuffle: bool = False,
        follow_batch: Optional[List[str]] = None,
        exclude_keys: Optional[List[str]] = None,
        **kwargs,
    ):
        """
        Initializes a new DataLoader object.
        :param dataset: The dataset from which to load the data.
        :type dataset: Union[Dataset, List[BaseData]]
        :param batch_size: How many samples per batch to load. (default: 1)
        :type batch_size: int, optional
        :param shuffle: If set to True, the data will be reshuffled at every epoch. (default: False)
        :type shuffle: bool, optional
        :param follow_batch: Creates assignment batch vectors for each key in the list. (default: None)
        :type follow_batch: List[str], optional
        :param exclude_keys: Will exclude each key in the list. (default: None)
        :type exclude_keys: List[str], optional
        :param **kwargs: Additional arguments of torch.utils.data.DataLoader.
        """

        if 'collate_fn' in kwargs:
            del kwargs['collate_fn']

        # Save for PyTorch Lightning:
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

        super().__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=Collater(follow_batch, exclude_keys),
            **kwargs,
        )


def collate_fn(data_list):

    data_list = [x for x in data_list if x is not None]
    return data_list


class DataListLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset: Union[Dataset, List[BaseData]],
                 batch_size: int = 1, shuffle: bool = False, **kwargs):
        if 'collate_fn' in kwargs:
            del kwargs['collate_fn']

        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle,
                         collate_fn=collate_fn, **kwargs)

