import re
import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn

with open("D:/Text Data/Machine_Translation/deu.txt", encoding="utf8") as f:
    sentences = f.readlines()

len(sentences)

NUM_INSTANCES = 100000
eng_sentences, deu_sentences = [], []
eng_words, deu_words = set(), set()
for i in tqdm(range(NUM_INSTANCES)):
    rand_idx = np.random.randint(len(sentences))
    # find only letters in sentences
    eng_sent, deu_sent = ["<sos>"], ["<sos>"]
    eng_sent += re.findall(r"\w+", sentences[rand_idx].split("\t")[0])
    deu_sent += re.findall(r"\w+", sentences[rand_idx].split("\t")[1])
    # change to lowercase
    eng_sent = [x.lower() for x in eng_sent]
    deu_sent = [x.lower() for x in deu_sent]
    eng_sent.append("<eos>")
    deu_sent.append("<eos>")
    # add parsed sentences
    eng_sentences.append(eng_sent)
    deu_sentences.append(deu_sent)
    # update unique words
    eng_words.update(eng_sent)
    deu_words.update(deu_sent)

eng_words, deu_words = list(eng_words), list(deu_words)

# print the size of the vocabulary
print(len(eng_words), len(deu_words))

# encode each token into index
for i in tqdm(range(len(eng_sentences))):
  eng_sentences[i] = [eng_words.index(x) for x in eng_sentences[i]]
  deu_sentences[i] = [deu_words.index(x) for x in deu_sentences[i]]

print(eng_sentences[0])
print([eng_words[x] for x in eng_sentences[0]])
print(deu_sentences[0])
print([deu_words[x] for x in deu_sentences[0]])


MAX_SENT_LEN = len(max(eng_sentences, key = len))
ENG_VOCAB_SIZE = len(eng_words)
DEU_VOCAB_SIZE = len(deu_words)
NUM_EPOCHS = 1
HIDDEN_SIZE = 128
EMBEDDING_DIM = 50
num_layers = 1

#creates embedding for the input word and feed into the LSTM network
class Encoder(nn.Module):
    def __init__(self, vocab_size, hidden_size, embedding_dim):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size)
    def forward(self, x, h0, c0):
        x = self.embedding(x).view(1, 1, -1)
        out, (h0, c0) = self.lstm(x, (h0, c0))
        return out, (h0, c0)

#create embedding of the input word and feed it into LSTM, Dense and Softmax layer to predict the word
class Decoder(nn.Module):
    def __init__(self, vocab_size, hidden_size, embedding_dim):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size)
        self.dense = nn.Linear(hidden_size, vocab_size)
        self.softmax = nn.LogSoftmax(dim=1)
    def forward(self, x, h0, c0):
        x = self.embedding(x).view(1, 1, -1)
        x, (h0, c0) = self.lstm(x, (h0, c0))
        x = self.softmax(self.dense(x.squeeze(0)))
        return x, (h0, c0)

encoder = Encoder(ENG_VOCAB_SIZE, HIDDEN_SIZE, EMBEDDING_DIM)
decoder = Decoder(DEU_VOCAB_SIZE, HIDDEN_SIZE, EMBEDDING_DIM)

encoder_opt = torch.optim.Adam(encoder.parameters(), lr = 0.01)
decoder_opt = torch.optim.Adam(decoder.parameters(), lr = 0.01)
criterion = nn.NLLLoss()
current_loss = []

for i in tqdm(range(NUM_EPOCHS)):
    for j in tqdm(range(len(eng_sentences))):
        source, target = eng_sentences[j], deu_sentences[j]
        source = torch.tensor(source, dtype=torch.long).view(-1, 1)
        target = torch.tensor(target, dtype=torch.long).view(-1, 1)
        loss = 0
        h0 = torch.zeros(1, 1, encoder.hidden_size)
        c0 = torch.zeros(1, 1, encoder.hidden_size)
        encoder_opt.zero_grad()
        decoder_opt.zero_grad()
        enc_output = torch.zeros(MAX_SENT_LEN, encoder.hidden_size)
        for k in range(source.size(0)):
            out, (h0, c0) = encoder(source[k].unsqueeze(0), h0, c0)
            enc_output[k] = out.squeeze()
        dec_input = torch.tensor([[deu_words.index("<sos>")]])
        for k in range(target.size(0)):
            out, (h0, c0) = decoder(dec_input, h0, c0)
            _, max_idx = out.topk(1)
            dec_input = max_idx.squeeze().detach()
            loss += criterion(out, target[k])
            if dec_input.item() == deu_words.index("<eos>"):
                break
        loss.backward()
        encoder_opt.step()
        decoder_opt.step()
        current_loss.append(loss.item() / (j + 1))