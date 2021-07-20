import torch
import glob
import unicodedata
import string
import re
all_letters = string.ascii_letters + " .,;'-" + string.digits


n_letters = len(all_letters)

def findFiles(path): return glob.glob(path)
filelocation = 'L://Knowledge Management/_KM (Secure)/Inspec/Inspec2/Inspec 2 Development/Abstract Quality/Names/*.txt'
# Turn a Unicode string to plain ASCII, thanks to http://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

# Read a file and split into lines
def readLines(filename):
    lines = open(filename,encoding = "utf-8").read().strip().split('\n')
    
    return [unicodeToAscii(line) for line in lines]

# Build the category_lines dictionary, a list of lines per category
category_lines = {}
all_categories = []
for filename in findFiles(filelocation):
    filename = str(filename).replace('\\','/')
    category = filename.split('/')[-1].split('.')[0]
    
    all_categories.append(category)
    lines = readLines(filename)
    category_lines[category] = lines

n_categories = len(all_categories)

# Find letter index from all_letters, e.g. "a" = 0
def letterToIndex(letter):
    return all_letters.find(letter)

# Turn a line into a <line_length x 1 x n_letters>,
# or an array of one-hot letter vectors
def lineToTensor(line):
    line = re.sub('[\W_]+', '', line)
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor
