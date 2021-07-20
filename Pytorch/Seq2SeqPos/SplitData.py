# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 11:59:19 2020

@author: csevern
"""
import pandas as pd

file = "L:\Knowledge Management\_KM (Secure)\Inspec\Inspec2\Inspec 2 Development\CCtoCC\CCtoCCtest.csv"
file = file.replace("\\", "/")
print("opened")
fd = pd.read_csv(file, encoding='utf-8')
CES = fd['CES'].tolist()
Hum = fd['HUMAN'].tolist()

trainsize = int(0.6*len(CES))
validsize = int(0.2*len(CES))
testsize = int(0.2*len(CES))

filetrainces = "L:\Knowledge Management\_KM (Secure)\Inspec\Inspec2\Inspec 2 Development\CCtoCC\Split/train.ces"
filetrainces = filetrainces.replace("\\", "/")
print(filetrainces)
filetrainhum = "L:\Knowledge Management\_KM (Secure)\Inspec\Inspec2\Inspec 2 Development\CCtoCC\Split/train.hum"
filetrainhum = filetrainhum.replace("\\", "/")
filevalidces = "L:\Knowledge Management\_KM (Secure)\Inspec\Inspec2\Inspec 2 Development\CCtoCC\Split/val.ces"
filevalidces = filevalidces.replace("\\", "/")
filevalidhum = "L:\Knowledge Management\_KM (Secure)\Inspec\Inspec2\Inspec 2 Development\CCtoCC\Split/val.hum"
filevalidhum = filevalidhum.replace("\\", "/")
filetestces = "L:\Knowledge Management\_KM (Secure)\Inspec\Inspec2\Inspec 2 Development\CCtoCC\Split/test.ces"
filetestces = filetestces.replace("\\", "/")
filetesthum = "L:\Knowledge Management\_KM (Secure)\Inspec\Inspec2\Inspec 2 Development\CCtoCC\Split/test.hum"
filetesthum = filetesthum.replace("\\", "/")

Ctrain = CES[:trainsize]
Cvalid = CES[trainsize:trainsize+validsize]
Ctest = CES[-testsize:]
Htrain = Hum[:trainsize]
Hvalid = Hum[trainsize:trainsize+validsize]
Htest = Hum[-testsize:]
print("saving")
with open(filetrainces, 'a+', encoding='utf-8') as f:
    for item in Ctrain:
        f.write('%s\n' %item)

with open(filevalidces, 'a+', encoding='utf-8') as f:
    for item in Cvalid:
        f.write('%s\n' %item)

with open(filetestces, 'a+', encoding='utf-8') as f:
    for item in Ctest:
        f.write('%s\n' %item)

with open(filetrainhum, 'a+', encoding='utf-8') as f:
    for item in Htrain:
        f.write('%s\n' %item)

with open(filevalidhum, 'a+', encoding='utf-8') as f:
    for item in Hvalid:
        f.write('%s\n' %item)

with open(filetesthum, 'a+', encoding='utf-8') as f:
    for item in Htest:
        f.write('%s\n' %item)



