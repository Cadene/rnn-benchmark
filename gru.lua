local timer = torch.Timer()

require 'nn'
require 'rnn'

require 'cutorch'
require 'cunn'

local nbatch = 100
local seqlen = 26
local batchsize = 100 --256
local inputsize = 620
local hiddensize = 2400

function GRU(inputsize, hiddensize, trimzero)
   local gru = nn.GRU(inputsize, hiddensize, false, 0, true)
   if trimzero then
      gru:trimZero(1)
   end
   local model = nn.Sequential()
      :add(nn.SplitTable(2))
      :add(nn.Sequencer(gru))
      :add(nn.SelectTable(-1))
   return model
end

local model = GRU(inputsize, hiddensize, true):cuda()

local input = torch.rand(batchsize, seqlen, inputsize):cuda()
local target = torch.rand(batchsize, hiddensize):cuda()
input[{{},{1,20},{}}]:fill(0)

local criterion = nn.MSECriterion():cuda()

criterion:forward(model:forward(input), target)
model:backward(input, criterion:backward(model.output, target))
cutorch.synchronize()
print("Setup : compile + forward/backward x 1")
print("--- " .. timer:time().real .. " seconds ---")

timer:reset()
for i=1, nbatch do
   model:forward(input)
end
cutorch.synchronize()
print("Forward:")
local nSamples = nbatch * batchsize
local speed = nSamples / timer:time().real
print("--- " .. nSamples .. " samples in " .. timer:time().real.. " seconds (" .. speed .. " samples/s, " .. 1000000/speed .. " microsec/samples) ---")

timer:reset()
for i = 1, nbatch do
   criterion:forward(model:forward(input), target)
   model:zeroGradParameters()
   model:backward(input, criterion:backward(model.output, target))
   model:updateParameters(0.01)
end
cutorch.synchronize()
print("Forward + Backward:")
local speed = nSamples / timer:time().real
print("--- " .. nSamples .. " samples in " .. timer:time().real .. " seconds (" .. speed .. " samples/s, " .. 1000000/speed .. " microsec/samples) ---")

