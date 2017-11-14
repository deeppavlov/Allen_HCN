from allennlp.commands.train import train_model_from_file

# registering the classes, don't delete this!:
from allen_hcn.babi_reader import BabiDatasetReader
from allen_hcn.actions import HCNActionTracker
from allen_hcn.entities import HCNEntityTracker
from allen_hcn.model import HybridCodeLSTM
from allen_hcn.iterator import DialogIterator

PARAMS_PATH = 'hcn.json'
SER_DIR = 'ser'

# LSTM example
# import torch
# from torch import nn
# from torch.autograd import Variable
# rnn = nn.LSTMCell(128, 128)
# input = Variable(torch.randn(1, 128))
# h0 = Variable(torch.randn(1, 128))
# c0 = Variable(torch.randn(1, 128))
# output, hn = rnn(input, (h0, c0))

# TODO fix the bug with always the same loss. Must be mess with Tensors instead of Variables.

model = train_model_from_file(PARAMS_PATH, SER_DIR)
