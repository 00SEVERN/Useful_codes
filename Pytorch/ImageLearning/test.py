# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 15:26:11 2019

@author: csevern
"""
import matplotlib.pyplot as plt

import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import time
start_time = time.time()
data_dir = 'L:/Knowledge Management/_KM (Secure)/Inspec/Inspec2/Inspec 2 Development/PDF_Images/Resized/'

def load_split_train_test(datadir, valid_size = .2):
    train_transforms = transforms.Compose([#transforms.RandomRotation(30),  # data augmentations are great
                                       #transforms.Crop(512),  # but not in this case of map tiles                                       
                                       #transforms.RandomHorizontalFlip()
                                       #transforms.Resize(512,672),
                                       #transforms.CenterCrop([512,672]),
                                       transforms.Grayscale(3),
                                       transforms.ToTensor(),
                                       #transforms.Normalize([0.485, 0.456, 0.406], # PyTorch recommends these but in this
                                       #                     [0.229, 0.224, 0.225]) # case I didn't get good results
                                       ])

    test_transforms = transforms.Compose([
                                      #transforms.Resize(512,672),
                                      #transforms.CenterCrop([512,672]),
                                      transforms.Grayscale(3),
                                      transforms.ToTensor(),
                                      #transforms.Normalize([0.485, 0.456, 0.406],
                                      #                     [0.229, 0.224, 0.225])
                                      ])

    train_data = datasets.ImageFolder(datadir, transform=train_transforms)
    test_data = datasets.ImageFolder(datadir, transform=test_transforms)

    #for data, x in train_data:
        #print(data.size())

        

    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))
    np.random.shuffle(indices)
    from torch.utils.data.sampler import SubsetRandomSampler
    train_idx, test_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(test_idx)
    trainloader = torch.utils.data.DataLoader(train_data, sampler=train_sampler, batch_size=100)
    testloader = torch.utils.data.DataLoader(test_data, sampler=test_sampler, batch_size=100)
    return trainloader, testloader


trainloader, testloader = load_split_train_test(data_dir, .2)
print(trainloader.dataset.classes)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet50(pretrained=True)
for param in model.parameters():
    param.requires_grad = False

model.fc = nn.Sequential(nn.Linear(2048, 1024),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(1024, 512),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(512, 2),
                                 nn.LogSoftmax(dim = 1))
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.0005)
model.to(device)
epochs = 100
steps = 0
running_loss = 0
print_every = 35
trainlow = 0.5
testacc = 0.5
train_losses, test_losses = [], []
for epoch in range(epochs):
    start_time2 = time.time()
    for inputs, labels in trainloader:
        #print(inputs.size())
        #print(labels.size())

        steps += 1

        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        logps = model.forward(inputs)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        if steps % print_every == 0:

            test_loss = 0
            accuracy = 0
            model.eval()
            
            with torch.no_grad():

                for inputs, labels in testloader:

                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model.forward(inputs)
                    batch_loss = abs(criterion(logps, labels))
                    test_loss += batch_loss.item()
                    
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += (torch.mean(equals.type(torch.FloatTensor)).item())
            
            train_losses.append(running_loss/len(trainloader))
            test_losses.append(test_loss/len(testloader))
            time_taken = time.time()-start_time2
            predicted_time = time_taken*(epochs-epoch)                 
            print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {running_loss/print_every:.3f}  "
                  f"Test loss: {test_loss/len(testloader):.3f}  "
                  f"Test accuracy: {1-(accuracy/len(testloader)):.3f}  "
                  f"Time taken: {time_taken:.3f}  "
                  f"Time predicted: {predicted_time:.3f}  ")
        
            
            if (test_loss/len(testloader)) < trainlow and (accuracy/len(testloader)) > testacc:
                testacc = (accuracy/len(testloader))
                trainlow = (test_loss/len(testloader))
                torch.save(model, 'L:/Knowledge Management/_KM (Secure)/Inspec/Inspec2/Inspec 2 Development/PDF_Images/Resized/AbRec2.pth')
                print("="*25, "Saved Model", "="*25)
                print("Training bar:", trainlow, "Accuracy Bar:", testacc) 
            running_loss = 0
                           
            model.train()
#torch.save(model, 'L:/Knowledge Management/_KM (Secure)/Inspec/Inspec2/Inspec 2 Development/PDF_Images/Resized/AbRec.pth')
plt.plot(train_losses, label='Training loss')
plt.plot(test_losses, label='Validation loss')
plt.legend(frameon=False)
plt.show()
print("--- %s seconds ---" % (time.time() - start_time))