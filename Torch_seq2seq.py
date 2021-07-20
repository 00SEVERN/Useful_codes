# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 10:25:36 2021

@author: csevern
"""
import torch

from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import random
import pickle
import clean_text as ct
CUDA_LAUNCH_BLOCKING=1
file = open("L:\\Knowledge Management\\_KM (Secure)\\Inspec\\Inspec2\\Inspec 2 Development\\CCPipeline\\Pytorch\\Smaller\\B.pickle",'rb')
filelines = pickle.load(file)
#%% create vocab

tokenizer = get_tokenizer('basic_english')
train_iter, test_iter = filelines[:int(0.8*len(filelines))], filelines[int(0.8*len(filelines)):]

def yield_tokens(data_iter):
    for _, text in data_iter:
        yield text

labels = list(set([label for label, text in train_iter]))
l2 = {label:i for i, label in enumerate(labels)}    


        
v2= build_vocab_from_iterator(yield_tokens(train_iter),min_freq=5, specials=["<unk>"])
v2.set_default_index(-1)

text_pipeline = lambda x: v2(ct.cleantext(x, "alpha_num","lemma").words)
label_pipeline = lambda x: l2[x]

#%% generate data batch

from torch.utils.data import DataLoader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
def collate_batch(batch):
    label_list, text_list, offsets = [], [], [0]
    for (_label, _text) in batch:
         label_list.append(label_pipeline(_label))
         processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
         text_list.append(processed_text)
         offsets.append(processed_text.size(0))
    label_list = torch.tensor(label_list, dtype=torch.int64)
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text_list = torch.cat(text_list)
    return label_list.to(device), text_list.to(device), offsets.to(device)


dataloader = DataLoader(train_iter, batch_size=8, shuffle=False, collate_fn=collate_batch)

#%% defining model
from torch import nn

class TextClassificationModel(nn.Module):

    def __init__(self, vocab_size, embed_dim, num_class):
        super(TextClassificationModel, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        self.fc = nn.Linear(embed_dim, num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        return self.fc(embedded)
    
#%% initiate instance

num_class = len(labels)
vocab_size = len(v2)
emsize = 64
model = TextClassificationModel(vocab_size, emsize, num_class).to(device)

#%% train stuff
import time

def train(dataloader):
    print("opening data loader")
    model.train()
    total_acc, total_count = 0, 0
    log_interval = 500
    start_time = time.time()
    print("starting dataloader")
    for idx, (label, text, offsets) in enumerate(dataloader):
        print(label, text, offsets)
        print(idx, "I get here")
        optimizer.zero_grad()
        predited_label = model(text, offsets)
        print("Predicted Label", predited_label)
        loss = criterion(predited_label, label)
        print(loss)
        loss.backward()
        print("Label", label)
        print("loss done")
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        print("stepping optimizer")
        optimizer.step()
        print("adding accuracy")
        print("Label", label)
        print(predited_label.argmax(1))
        print(predited_label.argmax(1), label)
        total_acc += (predited_label.argmax(1) == label).sum().item()
        
        print("total accuracy", total_acc)
        total_count += label.size(0)
        print("Total count", total_count)
        if idx % log_interval == 0 and idx > 0:
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches '
                  '| accuracy {:8.3f}'.format(epoch, idx, len(dataloader),
                                              total_acc/total_count))
            total_acc, total_count = 0, 0
            start_time = time.time()

def evaluate(dataloader):
    model.eval()
    total_acc, total_count = 0, 0

    with torch.no_grad():
        for idx, (label, text, offsets) in enumerate(dataloader):
            predited_label = model(text, offsets)
            loss = criterion(predited_label, label)
            total_acc += (predited_label.argmax(1) == label).sum().item()
            total_count += label.size(0)
    return total_acc/total_count

#%% split data

from torch.utils.data.dataset import random_split
from torchtext.data.functional import to_map_style_dataset
# Hyperparameters
EPOCHS = 10 # epoch
LR = 5  # learning rate
BATCH_SIZE = 1 # batch size for training

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.1)
total_accu = None
train_dataset = to_map_style_dataset(train_iter)
test_dataset = to_map_style_dataset(test_iter)
num_train = int(len(train_dataset) * 0.95)
split_train_, split_valid_ = \
    random_split(train_dataset, [num_train, len(train_dataset) - num_train])

train_dataloader = DataLoader(split_train_, batch_size=BATCH_SIZE,
                              shuffle=True, collate_fn=collate_batch)
valid_dataloader = DataLoader(split_valid_, batch_size=BATCH_SIZE,
                              shuffle=True, collate_fn=collate_batch)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                             shuffle=True, collate_fn=collate_batch)

for epoch in range(1, EPOCHS + 1):
    print(epoch, "I get to this EPOCH")
    epoch_start_time = time.time()
    train(train_dataloader)
    accu_val = evaluate(valid_dataloader)
    if total_accu is not None and total_accu > accu_val:
      scheduler.step()
    else:
       total_accu = accu_val
    print('-' * 59)
    print('| end of epoch {:3d} | time: {:5.2f}s | '
          'valid accuracy {:8.3f} '.format(epoch,
                                           time.time() - epoch_start_time,
                                           accu_val))
    print('-' * 59)