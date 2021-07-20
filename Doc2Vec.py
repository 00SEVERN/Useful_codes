# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 15:42:31 2021

@author: csevern
"""


from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
import pandas as pd
import pickle
text = open("",encoding='utf-8').read().splitlines()

    
#DataDF = pd.DataFrame(list(zip(Text, CCs)), columns=["Text", "CCs"])

data = text[:1000000]

tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(data)]

max_epochs = 300
vec_size = 1000
alpha = 0.025

model = Doc2Vec(vector_size=vec_size,
                alpha=alpha, 
                min_alpha=0.00001,
                min_count=2,
                dm =0)
  
model.build_vocab(tagged_data)
import time
for epoch in range(max_epochs):
    time_start = time.time()
    
    model.train(tagged_data,
                total_examples=model.corpus_count,
                epochs=model.iter)
    # decrease the learning rate
    model.alpha -= 0.0002
    # fix the learning rate, no decay
    model.min_alpha = model.alpha
    print('iteration % 2d, time taken %5.2f' %(epoch, time.time()-time_start))


model.save("d2v_Full_3.model")
print("Model Saved")
filename1 = ""
pickle.dump(model, open(filename1, 'wb'))
"""
test_data = word_tokenize("I love chatbots".lower())
v1 = model.infer_vector(test_data)
print("V1_infer", v1)

# to find most similar doc using tags
similar_doc = model.docvecs.most_similar([v1])
print(similar_doc)
"""

