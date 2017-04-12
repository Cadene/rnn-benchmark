import time
end = time.time()

import torch
import torch.nn as nn
from torch.autograd import Variable

class GRU(nn.Module):

    def __init__(self, input_size, output_size):
        super(GRU, self).__init__()
        self.gru = nn.GRU(input_size=input_size,
                          hidden_size=output_size,
                          batch_first=True)
 
    def _select_last(self, input, lengths):
        x = Variable(input.data.new().resize_((input.size(0), input.size(2))))
        for i in range(input.size(0)):
            x[i] = input[i,lengths[i]-1]
        return x
 
    def forward(self, input, lengths=[26]*100):
        x, hn = self.gru(input)
        if lengths:
            x = self._select_last(x, lengths)
        return x

class Uniskip(nn.Module):

    def __init__(self, input_size, output_size, vocab_size):
        super(Uniskip, self).__init__()
        self.emb = nn.Embedding(num_embeddings=vocab_size+1,
                                embedding_dim=input_size,
                                padding_idx=0)
        self.gru = GRU(input_size, output_size)

    def _process_lengths(self, input):
        max_length = input.size(1)
        lengths = list(max_length - input.data.eq(0).sum(1).squeeze())
        return lengths

    def forward(self, input):
        lengths = self._process_lengths(input)
        x = self.emb(input)
        x = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True)
        x = self.gru(x, lengths=None)
        x, lengths = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        x = self.gru._select_last(x, lengths)
        return x

def create_gru(input_size, output_size):
    input = Variable(torch.rand(batch_size, seq_len, input_size)).cuda()
    model = GRU(input_size, output_size).cuda()
    return input, model

def create_uniskip(input_size, output_size, vocab_size):
    input = Variable((torch.rand(batch_size, seq_len)*vocab_size).long()).cuda()
    model = Uniskip(input_size, output_size, vocab_size).cuda()
    return input, model

nbatch = 100
seq_len = 26
input_size = 620
output_size = 2400
batch_size = 100
vocab_size = 80000

#input, model = create_gru(input_size, output_size)
input, model = create_uniskip(input_size, output_size, vocab_size)
target = Variable(torch.randn(batch_size, output_size)).cuda()

criterion = nn.MSELoss().cuda()

loss = criterion.forward(model.forward(input), target)
loss.backward()
torch.cuda.synchronize()
print("Setup : compile + forward/backward x 1")
print("--- {} seconds ---".format(time.time() - end))


end = time.time()
for i in range(nbatch):
   model.forward(input)
torch.cuda.synchronize()
print("Forward:")
nSamples = nbatch * batch_size
speed = nSamples / (time.time() - end)
print("--- {} samples in {} seconds ({} samples/s, {} microsec/samples) ---".format(
        nSamples, (time.time() - end), speed, 1000000/speed))


end = time.time()
for i in range(nbatch):
    loss = criterion.forward(model.forward(input), target)
    loss.backward()
torch.cuda.synchronize()
print("Forward + Backward:")
speed = nSamples / (time.time() - end)
print("--- {} samples in {} seconds ({} samples/s, {} microsec/samples) ---".format(
        nSamples, (time.time() - end), speed, 1000000/speed))

