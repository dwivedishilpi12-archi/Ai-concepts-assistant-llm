import torch
from model import TinyLLM
from tokenizer import Tokenizer

with open("data.txt") as f:
    text = f.read()

tokenizer = Tokenizer(text)
data = torch.tensor(tokenizer.encode(text))

vocab_size = len(tokenizer.stoi)
seq_len = 32

def get_batch():
    ix = torch.randint(0, len(data)-seq_len, (1,))
    x = data[ix:ix+seq_len]
    y = data[ix+1:ix+seq_len+1]
    return x.unsqueeze(0), y.unsqueeze(0)

device = "cuda" if torch.cuda.is_available() else "cpu"

model = TinyLLM(vocab_size).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = torch.nn.CrossEntropyLoss()

for step in range(300):
    x, y = get_batch()
    x, y = x.to(device), y.to(device)

    logits = model(x)
    loss = loss_fn(logits.view(-1, vocab_size), y.view(-1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

torch.save(model.state_dict(), "model.pth")
print("Training Done")
