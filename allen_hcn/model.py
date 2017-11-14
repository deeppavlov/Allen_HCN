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
        self.init_state_c = Variable(torch.zeros(1, self.nb_hidden))
        self.init_state_h = Variable(torch.zeros(1, self.nb_hidden))
        self.net = torch.nn.LSTMCell(input_size=self.nb_hidden, hidden_size=self.nb_hidden)
        self._loss = torch.nn.CrossEntropyLoss()
        initializer(self)

    @overrides
    def forward(self,
                source_tokens: Dict[str, Callable[[Variable], torch.LongTensor]],
                target_template_number=None) -> Dict[str, torch.FloatTensor]:
        token_indices = source_tokens['tokens']
        output_dict = {'loss': []}
        count_instances = 0

        # Count loss on a single Instance (utterance-response pair)
        for indices, target_number in zip(token_indices, target_template_number):
            utt = ' '.join([self.vocab._index_to_token['tokens'][t_idx] for t_idx in indices.data if
                            t_idx != 0])
            u_ent = self.et.extract_entities(utt)
            u_ent_features = self.et.context_features()
            u_emb = self.emb.encode(utt)
            u_bow = self.bow_enc.encode(utt)

            # concat features
            features = np.array([np.concatenate((u_ent_features, u_emb, u_bow), axis=0)])
            inputs = torch.from_numpy(features).float()

            # input projection
            w_init = torch.nn.init.xavier_uniform(torch.zeros(self.obs_size, self.nb_hidden))
            b_init = torch.nn.init.constant(torch.zeros(1, self.nb_hidden), val=0.)
            projected_inputs = torch.matmul(inputs, w_init) + b_init

            # get hn and state
            hn, cn = self.net(Variable(projected_inputs), (self.init_state_h,
                                                           self.init_state_c))

            # reshape LSTM's state tuple (2,128) -> (1,256)
            state_reshaped = torch.cat((hn, cn), 1).data

            # output projection
            w_out = torch.nn.init.xavier_uniform(torch.zeros(2 * self.nb_hidden, self.action_size))
            b_out = torch.nn.init.constant(torch.zeros(1, self.action_size), val=0.)

            # get logits
            logits = torch.matmul(state_reshaped, w_out) + b_out

            # probabilities
            #  normalization : elemwise multiply with action mask
            smax = torch.nn.Softmax()
            tensor_action_mask = torch.from_numpy(self.at.action_mask()).float()
            squeezed = torch.squeeze(smax(logits))
            probs = torch.mul(squeezed.data, tensor_action_mask)

            # loss
            loss = self._loss(logits, target_number)

        return sum(output_dict['loss'])/len(output_dict)

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
