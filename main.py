import torch
import torch.nn as nn

with open('data/input.txt', 'r') as fp:
    text = fp.read()

characters = sorted(list(set(text)))
vocab_size = len(characters)

# token (characters)
char_to_int = { chr:i for i, chr in enumerate(characters)}
int_to_char = { i:chr for i, chr in enumerate(characters)}

# encoding and decoding (strings):
encode = lambda string: [char_to_int[char] for char in string]
decode = lambda ints: [int_to_char[i] for i in ints]

data = torch.tensor(encode(text), dtype=torch.long)

# spliting into train and validate
spliter = int(0.9*len(data))
train = data[:spliter]
validate = data[spliter:]

# how many each block?
block_size = 8 
# NOTE: when 8 individuals, there are only 7 combinations (since a token can't be by itself)
# NOTE: when setting a block_size of n, that means we need to truncate after n, since the machine never looked at anything more than n
batch_size = 4 # how many blocks we will deal with at the same time?

def get_batch(type):
    data = train if type is "train" else validate
    indexes = torch.randint(len(data-block_size), (batch_size,)) # note the difference of usage of 'block_size' and 'batch_size' here. this return a batch_size sized tensor with value from 0 to data-block_size
    xs = torch.stack([data[i:i+block_size] for i in indexes]) # we use stack, since the generator generates multiple tensor: eventually 4 tensor of  8 values
    ys = torch.stack([data[i+1:i+block_size+1] for i in indexes]) # (1,) = 2, (1,2,) = 3... ys are always the x+1.
    return xs, ys

class BigramLM(nn.Module):
    
    def __init__(self, vocab_size) -> None:
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
        # here we're storing a table that's vocab*vocab I suppose? 
        # in this case the vocab size is all character (the len of the set of characters)
        # e,g., token_id 24 will take the slice index 24th

        # e.g.,: 
        #  token_id|dim[0]|dim[1]|dim[2]|...dim[vocab_size]
        # 0x01('a')|  2.3|   3.3|   4.2|....

    def forward(self, idx, targets=None):
        # index: for given position
        # targets: the next token
        #   logits: what the model gives you for next token

        logits = self.token_embedding_table(idx) # (batch * time * channel) <- (4 * 8 * vocab_size)
        # the time here is literally time, like 1: I, 2: am, 3: Ruikai, 3 goes after 2, goes after 1
        
        # below wouldn't work since cross_entropy (loss function) expects B * C * T instead of B * T * C
        # loss = nn.functional.cross_entropy(logits, targets) # logics (prediction) : targets ()

        if targets is None:
            return logits, None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C) # cross_entropy wants channel as second dimension...
            targets = targets.view(B*T)
            loss = nn.functional.cross_entropy(logits, targets)
            return logits, loss

    def generate(self, idx, max_length):
        # idx = index of current (B * T)
        # max_length = how long do you want it to go?
        for _ in range(max_length):
            logits, _ = self(idx) # _ is the loss 
            # self calls forward (nn.Module.__call__(self, *args, **kwargs) -> self.forward(*args, **kwargs))
            logits = logits[:,-1,:] # (B, T(the last), C)
            probabalities = nn.functional.softmax(logits, dim=-1)
            next = torch.multinomial(probabalities, num_samples=1) # sample one
            idx = torch.cat((idx, next), dim=1) # (B, T)
        return idx

xb, yb = get_batch('train')
model = BigramLM(vocab_size)
logits, loss = model.forward(xb, yb)

# print(loss) <- the loss here is predictable, 
# statistically speaking we it should be around -ln(1/45) (vocab_size, possible next token)

# let's now generate!
idx = torch.zeros((1,1), dtype=torch.long)
# batch=1, time=1, 1x1 tensor holding zeros
# zero is the token_id for a newline '\n' chatacter

# print(''.join(decode(model.generate(idx, max_length=100)[0].tolist())))
# ^^^ this out puts gibberish, since the model is just randomly "predicts" (sampling) whats after the chatacter, so we need:
# 1. predict the next word not randomly
# 1, make "previous" context related to the nextword
