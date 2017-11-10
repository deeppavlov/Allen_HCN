from typing import List
import json

from overrides import overrides

from allennlp.common import Params
from allennlp.data.iterators.data_iterator import DataIterator
from allennlp.data.dataset import Dataset
from allennlp.data.instance import Instance


@DataIterator.register("dialog")
class DialogIterator(DataIterator):

    def __init__(self, dialog_indices_path):
        with open(dialog_indices_path) as f:
            self.dialog_indices = json.load(f)

    @overrides
    def get_num_batches(self, dataset: Dataset) -> int:
        return len(self.dialog_indices)

    @overrides
    def _create_batches(self, dataset: Dataset, shuffle: False) -> List[List[Instance]]:
        # form the dialogs
        dialogs = []
        for i, dialog_idx in enumerate(self.dialog_indices):
            # get start and end index of the instances
            start, end = dialog_idx['start'], dialog_idx['end']
            dialogs.append(dataset.instances[start:end])
        return dialogs

    @classmethod
    def from_params(cls, params: Params) -> 'DialogIterator':
        dialog_indices_path = params.pop('dialog_indices_path')
        params.assert_empty(cls.__name__)
        return cls(dialog_indices_path=dialog_indices_path)
