from overrides import overrides
from typing import Dict, Callable
import numpy as np
import torch
from torch.autograd import Variable

from allennlp.models import Model
from allennlp.data.instance import Instance
from allennlp.data import Vocabulary
from allennlp.common.params import Params
from allennlp.nn import InitializerApplicator

from allen_hcn.actions import HCNActionTracker
from allen_hcn.entities import HCNEntityTracker
from allen_hcn.bow import BoW_encoder
from allen_hcn.embed import UtteranceEmbed


@Model.register("hcn")
class HybridCodeLSTM(Model):
    """
       This ``Model`` implements the Hybrid Code Network model described in:
       <https://github.com/voicy-ai/DialogStateTracking/tree/master/src/hcn>
    """

    def __init__(self, vocab, nb_hidden, at_path='data/dialog-babi-task5-full-dialogs-trn.txt',
                 initializer: InitializerApplicator = InitializerApplicator()):
        super(HybridCodeLSTM, self).__init__(vocab)
        self.et = HCNEntityTracker()
        self.at = HCNActionTracker(self.et, at_path)
        self.bow_enc = BoW_encoder(at_path)
        self.emb = UtteranceEmbed()
        self.action_size = self.at.action_size
        self.obs_size = self.emb.dim + self.bow_enc.vocab_size + self.et.num_features
        self.nb_hidden = nb_hidden
        self.init_state_c = Variable(
            torch.zeros(1, 1, self.nb_hidden))  # TODO why I have to add a dim??
        self.init_state_h = Variable(torch.zeros(1, 1, self.nb_hidden))
        obs_size = self.emb.dim + self.bow_enc.vocab_size + self.et.num_features
        self.net = torch.nn.LSTM(input_size=obs_size, hidden_size=self.nb_hidden)
        self._loss = torch.nn.CrossEntropyLoss()
        initializer(self)

    @overrides
    def forward(self,
                source_tokens: Dict[str, Callable[[Variable], torch.LongTensor]],
                target_template_number=None) -> Dict[str, torch.FloatTensor]:
        token_indices = source_tokens['tokens']
        for indices in token_indices:
            utt = ' '.join([self.vocab._index_to_token['tokens'][t_idx] for t_idx in indices.data if
                            t_idx != 0])
            # TODO why u_ent unused in the original repo?
            u_ent = self.et.extract_entities(utt)
            u_ent_features = self.et.context_features()
            u_emb = self.emb.encode(utt)
            u_bow = self.bow_enc.encode(utt)
            # concat features
            features = np.array([np.concatenate((u_ent_features, u_emb, u_bow), axis=0)])
            inputs = Variable(torch.from_numpy(features).float())
            output, hn = self.net(inputs, (self.init_state_h, self.init_state_c))

        return {'loss': output}

    @overrides
    def forward_on_instance(self, instance: Instance):
        features = None
        action = None
        action_mask = None
        model_input = {
            self.features_: features.reshape([1, self.obs_size]),
            self.action_: [action],
            self.init_state_c_: self.init_state_c,
            self.init_state_h_: self.init_state_h,
            self.action_mask_: action_mask
        }

        output, _ = self.nn(model_input, (self.init_state_c, self.init_state_h))
        return output

    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params) -> 'HybridCodeLSTM':
        nb_hidden = params.pop("nb_hidden")
        return cls(vocab=vocab,
                   nb_hidden=nb_hidden)

        # def reset_parameters(self):
        #     self.init_state_c = zeros(1, self.hidden_dim)
        #     self.init_state_h = zeros(1, self.hidden_dim)
