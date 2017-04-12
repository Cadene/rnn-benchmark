local timer = torch.Timer()

require 'nn'
require 'rnn'

require 'cutorch'
require 'cunn'

function GRU(input_size, output_size, trimzero)
   local gru = nn.GRU(input_size, output_size, false, 0, true)
   if trimzero then
      --gru:trimZero(1)
   end
   local model = nn.Sequential()
      :add(nn.SplitTable(2))
      :add(nn.Sequencer(gru))
      :add(nn.SelectTable(-1))
   return model
end

function Uniskip(input_size, output_size, trimzero, vocab_size)
   local model = nn.Sequential()
      :add(nn.LookupTableMaskZero(vocab_size, input_size))
      :add(GRU(input_size, output_size, trimzero))
   return model
end

function MLB(input_size, output_size, trimzero, vocab_size)
   local emb_q = Uniskip(input_size, output_size, trimzero, vocab_size)
   local fusion = nn.Sequential()
      :add(
         nn.ParallelTable()
            :add(
               nn.Sequential()
                  :add(nn.Linear(output_size, 1200))
                  :add(nn.Tanh())
            )
            :add(
               nn.Sequential()
                  :add(nn.Linear(2048, 1200))
                  :add(nn.Tanh())
            )
      )
      :add(nn.CMulTable())
      :add(nn.Tanh())
      :add(nn.Linear(1200, 2000))
   local model = nn.Sequential()
      :add(
         nn.ParallelTable()
            :add(nn.Identity())
            :add(emb_q)
      )
      :add(fusion)
   return model
end

function create_gru(batch_size, seq_len, input_size, output_size, trimzero)
   local input = torch.rand(batch_size, seq_len, input_size):cuda()
   input[{{},{1,20},{}}]:fill(0)
   local target = torch.rand(batch_size, output_size):cuda()
   local model = GRU(input_size, output_size, trimzero):cuda()
   return input, target, model
end

function create_uniskip(batch_size, seq_len, input_size, output_size, trimzero, vocab_size)
   local input = (torch.rand(batch_size, seq_len)*vocab_size):long():cuda()
   input[{{},{1,20}}]:fill(0)
   local target = torch.rand(batch_size, output_size):cuda()
   local model = Uniskip(input_size, output_size, trimzero, vocab_size):cuda()
   return input, target, model
end

function create_mlb(input_size, output_size, trimzero, vocab_size)
   local input = (torch.rand(batch_size, seq_len)*vocab_size):long():cuda()
   input[{{},{1,20}}]:fill(0)
   local target = torch.rand(batch_size, output_size):cuda()
   local model = MLB(input_size, output_size, trimzero, vocab_size):cuda()
   return input, target, model
end

local nbatch = 100
local seq_len = 26
local batch_size = 100 --256
local input_size = 620
local output_size = 2400
local vocab_size = 80000
local trimzero = true

-- local input, target, model = create_gru(batch_size, seq_len, input_size, output_size, trimzero)
local input, target, model = create_uniskip(batch_size, seq_len, input_size, output_size, trimzero, vocab_size)

print(model)

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
local nSamples = nbatch * batch_size
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

