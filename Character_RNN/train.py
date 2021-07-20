import torch
from data import *
from model import *
import random
import time
import math
import re
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker



n_hidden = 256
n_epochs = 1000000
print_every = 5000
plot_every = 1000
learning_rate = 0.005 # If you set this too high, it might explode. If too low, it might not learn

def categoryFromOutput(output):
    top_n, top_i = output.data.topk(1) # Tensor out of Variable with .data
    
    category_i = top_i[0][0]
    if category_i > 1:
        category_i = 1
    return all_categories[category_i], category_i

def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]

def randomTrainingPair():
    category = randomChoice(all_categories)
    line = randomChoice(category_lines[category])
    category_tensor = Variable(torch.LongTensor([all_categories.index(category)]))
    line_tensor = Variable(lineToTensor(line))
    return category, line, category_tensor, line_tensor

rnn = RNN(n_letters, n_hidden, n_categories)


optimizer = torch.optim.SGD(rnn.parameters(), lr=learning_rate)
criterion = nn.NLLLoss()

def train(category_tensor, line_tensor):
    hidden = rnn.initHidden()
    optimizer.zero_grad()

    for i in range(line_tensor.size()[0]):
        
        
        output, hidden = rnn(line_tensor[i], hidden)
        
    
    loss = criterion(output, category_tensor)
    loss.backward()

    optimizer.step()

    
    return output, loss.data

# Keep track of losses for plotting
current_loss = 0
all_losses = []
batch_loss = 0
def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

start = time.time()

for epoch in range(1, n_epochs + 1):
    category, line, category_tensor, line_tensor = randomTrainingPair()
    line = re.sub('[\W_]+', '', line)
    if line != "":
        
        output, loss = train(category_tensor, line_tensor)
        current_loss += loss
        batch_loss += loss
    # Print epoch number, loss, name and guess
    if epoch % print_every == 0:
        guess, guess_i = categoryFromOutput(output)
        correct = '✓' if guess == category else '✗ (%s)' % category
        print('%d %d%% (%s) %.7f %s / %s %s' % (epoch, epoch / n_epochs * 100, timeSince(start), batch_loss/print_every, line, guess, correct))
        batch_loss = 0

    # Add current loss avg to list of losses
    if epoch % plot_every == 0:
        all_losses.append(current_loss / plot_every)
        current_loss = 0
        
plt.figure()
plt.plot(all_losses)
fig = plt.figure()
ax = fig.add_subplot(111)
confusion = torch.zeros(n_categories, n_categories)
n_confusion = 10000

cax = ax.matshow(confusion.numpy())
fig.colorbar(cax)

# Set up axes
ax.set_xticklabels([''] + all_categories, rotation=90)
ax.set_yticklabels([''] + all_categories)

# Force label at every tick
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

# sphinx_gallery_thumbnail_number = 2
plt.show()
torch.save(rnn, 'NameRec.pt')

