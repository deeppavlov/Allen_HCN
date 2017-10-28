import logging
from overrides import overrides
from typing import Dict

from allennlp.data.dataset import Dataset
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.instance import Instance
from allennlp.data.fields import TextField, IndexField, SequenceField
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.common.checks import ConfigurationError
from allennlp.common import Params

import allen_hcn.util as util

logger = logging.getLogger(__name__)


@DatasetReader.register("babi")
class BabiDatasetReader(DatasetReader):
    """
    Read a tsv file containing paired sequences, and create a dataset suitable for a
    ``HybridCodeNetwork`` model.

    Expected format for each input line is <ID user_utterance [tab] bot_utterance>

     The output of ``read`` is a list of ``Instance``s with the fields:
        source_tokens: ``TextField`` and
        target_tokens: ``TextField``
    """

    def __init__(self, entity_tracker, action_tracker,
                 token_indexers: Dict[str, TokenIndexer] = None):
        self._action_templates = SequenceField(action_tracker.get_action_templates())
        self._et = entity_tracker
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self.dialog_indices = None

    @overrides
    def read(self, file_path) -> Dataset:
        """
        Read data from the `file_path` and return a :class:`Dataset`.
        """
        # get dialogs from file
        logger.info("Reading instances from lines in file at: {}".format(file_path))
        dialogs, dialog_indices = util.read_dialogs(file_path, with_indices=True)
        self.dialog_indices = dialog_indices

        # get utterances
        utterances = util.get_utterances(dialogs)
        # get responses
        responses = util.get_responses(dialogs)
        responses = [self.get_template_id(response) for response in responses]

        instances = []
        for u, r in zip(utterances, responses):
            instances.append(self.text_to_instance(u, r))

        if not instances:
            raise ConfigurationError("No instances read!")

        return Dataset(instances)

    def text_to_instance(self, user_utterance: str, response_template: int = -1) -> Instance:
        tokenized_source = user_utterance.split(' ')
        source_field = TextField(tokenized_source, self._token_indexers)
        if response_template != -1:
            target_field = IndexField(response_template, self._action_templates)
            return Instance({"source_tokens": source_field, "target_template_number": target_field})
        else:
            return Instance({'source_tokens': source_field})

    @classmethod
    def from_params(cls, params: Params) -> 'BabiDatasetReader':
        """
        Constructs the dataset reader described by ``params``.
        """
        entity_tracker = params.pop('entity_tracker', None)
        action_tracker = params.pop('action_tracker', None)

        token_indexers_type = params.pop('source_token_indexers', None)
        if token_indexers_type is None:
            token_indexers = None
        else:
            token_indexers = TokenIndexer.dict_from_params(token_indexers_type)
        params.assert_empty(cls.__name__)
        return BabiDatasetReader(entity_tracker, action_tracker, token_indexers)

    def get_template_id(self, response):

        def extract_(response):
            template = []
            for word in response.split(' '):
                if 'resto_' in word:
                    if 'phone' in word:
                        template.append('<info_phone>')
                    elif 'address' in word:
                        template.append('<info_address>')
                    else:
                        template.append('<restaurant>')
                else:
                    template.append(word)
            return ' '.join(template)

        return self.action_templates.index(
            extract_(self.et.extract_entities(response, update=False))
        )
