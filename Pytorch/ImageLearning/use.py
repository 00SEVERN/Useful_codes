# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 10:53:06 2019

@author: csevern
"""

import matplotlib.pyplot as plt
import numpy
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import Image
import os
import random
from shutil import copyfile
from torch.autograd import Variable

Abstract = 'L:/Knowledge Management/_KM (Secure)/Inspec/Inspec2/Inspec 2 Development/PDF_Images/Abstract/'
NotAbstract = 'L:/Knowledge Management/_KM (Secure)/Inspec/Inspec2/Inspec 2 Development/PDF_Images/NotAbstract/'
Abend = 'L:/Knowledge Management/_KM (Secure)/Inspec/Inspec2/Inspec 2 Development/PDF_Images/Test/Abstract/'
NAbend = 'L:/Knowledge Management/_KM (Secure)/Inspec/Inspec2/Inspec 2 Development/PDF_Images/Test/Not Abstract/'
Abend1 = 'L:/Knowledge Management/_KM (Secure)/Inspec/Inspec2/Inspec 2 Development/PDF_Images/Test1/Abstract/'
NAbend1 = 'L:/Knowledge Management/_KM (Secure)/Inspec/Inspec2/Inspec 2 Development/PDF_Images/Test1/Not Abstract/'
'''

Ablist = os.listdir(Abstract)
NAblist = os.listdir(NotAbstract)

random.shuffle(Ablist)
random.shuffle(NAblist)
print("Shuffled files")
for x in range(0, 100):
    imageref = NAblist[x]
    path2 = NotAbstract + imageref
    img = Image.open(path2)
    #print(float(img.size[1]), float(img.size[2]))
    height = img.size[0]
    width = img.size[1]
    area = (100, 100, height-100, width-100)
    new_img = img.crop(area)
    basewidth = 256
    hsize = int(basewidth*1.4)
 
    endpath2 = NAbend + imageref
    img = new_img.resize((basewidth,hsize), Image.ANTIALIAS)
    img.save(endpath2)
    endpath = NAbend1 + imageref
    copyfile(path2, endpath)     

for x in range(0, 100):
    imageref = Ablist[x]
    path2 = Abstract + imageref
    img = Image.open(path2)
    #print(float(img.size[1]), float(img.size[2]))
    height = img.size[0]
    width = img.size[1]
    area = (100, 100, height-100, width-100)
    new_img = img.crop(area)
    basewidth = 256
    hsize = int(basewidth*1.4)
 
    endpath2 = Abend + imageref
    img = new_img.resize((basewidth,hsize), Image.ANTIALIAS)
    img.save(endpath2)
    endpath = Abend1 + imageref
    copyfile(path2, endpath)        

print("Saved Files")

'''
data_dir = 'L:/Knowledge Management/_KM (Secure)/Inspec/Inspec2/Inspec 2 Development/PDF_Images/Test/'


test_transforms = transforms.Compose([transforms.Grayscale(3),
                                      transforms.ToTensor(),
                                      #transforms.Normalize([0.485, 0.456, 0.406],
                                      #                     [0.229, 0.224, 0.225])
                                     ])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model=torch.load('L:/Knowledge Management/_KM (Secure)/Inspec/Inspec2/Inspec 2 Development/PDF_Images/Resized/AbRec2.pth')
model.eval()

def predict_image(image):
    image_tensor = test_transforms(image).float()
    image_tensor = image_tensor.unsqueeze_(0)
    input = Variable(image_tensor)
    input = input.to(device)
    output = model(input)
    maxnum = output.data.cpu().numpy().max()
    minnum = output.data.cpu().numpy().min()
    sumnum = output.data.cpu().numpy().sum()
    sumlog = 1- numpy.exp(sumnum)
    #print(abs(numpy.exp(maxnum)/sumlog)**10, sumlog)
    index = output.data.cpu().numpy().argmax()

    return index  

def get_random_images(num):
    data = datasets.ImageFolder(data_dir, transform=test_transforms)

    classes = data.classes
    print(classes)
    indices = list(range(len(data)))

    np.random.shuffle(indices)
    idx = indices[:num]

    from torch.utils.data.sampler import SubsetRandomSampler
    sampler = SubsetRandomSampler(idx)

    loader = torch.utils.data.DataLoader(data, sampler=sampler, batch_size=num)
    dataiter = iter(loader)
    images, labels = dataiter.next()

    return images, labels, classes;



to_pil = transforms.ToPILImage()
images, labels, classes = get_random_images(460)
fig=plt.figure(figsize=(50,50))
tru=0
fal = 0
incorrect= []
label = []
for ii in range(len(images)):

    image = to_pil(images[ii])
    index = predict_image(image)
    
    res = int(labels[ii]) == index

    if str(res) == "True":
        tru +=1
    else:
        fal +=1
        incorrect.append(image)        
        label.append(str(classes[index]))
        
for i in range(len(incorrect)):
    image = incorrect[i]   
    lab = label[i]

    sub = fig.add_subplot(1, len(incorrect), i+1)
    sub.set_title(str(lab) + ":" + "False")
    plt.axis('off')
    plt.imshow(image)
plt.show()
#print(tru)
#print(tru/(tru+fal))

