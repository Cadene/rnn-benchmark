import time
end = time.time()

import torch
import torch.nn as nn
from torch.autograd import Variable

class Uniskip(nn.Module):

    def __init__(self, input_size, hidden_size):
        super(Uniskip, self).__init__()
        self.gru = nn.GRU(input_size=input_size,
                          hidden_size=hidden_size,
                          batch_first=True)
 
    def _select_last(self, input, lengths):
        x = Variable(input.data.new().resize_((input.size(0), input.size(2))))
        for i in range(input.size(0)):
            x[i] = input[i,lengths[i]-1]
        return x
 
    def forward(self, input, lengths=[26]*100):
        x, hn = self.gru(input)
        x = self._select_last(x, lengths)
        return x

nbatch = 100
seq_len = 26
input_size = 620
hidden_size = 2400
batch_size = 100

model = Uniskip(input_size, hidden_size)
model = model.cuda()
criterion = nn.MSELoss().cuda()

input = Variable(torch.randn(batch_size, seq_len, input_size)).cuda()
target = Variable(torch.randn(batch_size, hidden_size)).cuda()

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

