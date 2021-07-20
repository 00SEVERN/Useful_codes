# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 09:54:10 2019

@author: csevern
"""


import pandas as pd
import nltk
import random
from nltk import DecisionTreeClassifier
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB,BernoulliNB, GaussianNB, ComplementNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.gaussian_process.kernels import RBF
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.linear_model import LogisticRegressionCV,SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
import pickle
import os
import time
import gc
import FeatureFunction as FF


vectorizer = CountVectorizer()
print("Started")

#import grams from predetermined dictionary.
Excelfile = open("", encoding = 'utf-8')
wordfeatures = Excelfile.readlines()
word_features = []
for word in wordfeatures:
    word = word.replace("\n", "")
    word_features.append(word)


filesdone = os.listdir("")
filesdone = [x.strip(".pickle") for x in filesdone]
#file directories to iterate over
filedir = ""
n  = 1
#create file for machine learning training outputs (how well the machine learning model did for that file)
updatedetails = "CC" + "," + "Training Sample" +  "," +"Linear SVC" + "," + "Complement NB" + "," + "Time Taken (s)"
#where do you want to save the training outputs, helps you keep track of the size of training documents and how well each model did
savedata = ""
with open(savedata, 'a+', encoding = 'utf-8') as f:    
    f.write("%s\n" % updatedetails)
start_time2 = time.time()
for f in os.listdir(filedir):
    #collect and reallocate ram 
    gc.collect()
    start_time = time.time()
    name = f[:-4]
    print(name)
    #if this file is already done, skip it
    #if name in filesdone:
        #continue
    location = filedir + f
    file = open(location, encoding = 'utf-8')
    lines = file.readlines()
    #decide how many lines in the file you are going to use, 10000 is a good number IF YOU HAVE A LOT OF RAM
    lenpapers = 10000
    if lenpapers > len(lines):
        lenpapers = len(lines)
    lines = lines[:lenpapers]
    documents = []
    featuresets = []
    
    correctdata = "no"
    correctcount = 0
    #Each line is split into "abstract \t yes/no"
    #Split the lines and check that there are "yes" lines in the data
    for w in range(0,lenpapers):
        line = lines[w]
        splitline = line.split('\t')
        wordline = splitline[0].lower()
        words = nltk.word_tokenize(wordline)

        if len(splitline) >1 :
            category = splitline[1].replace("\n", "")
            tup = (words, category)
            if "yes" in tup:
                correctdata = "yes"
                correctcount += 1
            documents.append(tup)
    #print(documents)
    print("finished documents")
    #if "yes" documents make up less than 1/3 of the training set, skip because it is not worth it
    if correctcount < 0.3*(len(documents)):
        continue
    documents = [x for x in documents if x != None]
    #shuffle the documents each time, so that you get different results
    random.shuffle(documents)
    #print(documents)
    #this tags the "abstracts" with the relevant words from the dictionary
    #in the format contains{word} = True/False
    def document_features(document):
        document_words = set(document)
        features = {}
        for word in word_features:
            features['contains({})'.format(word)] = (word in document_words)
    
        return features;
    
        
    print("starting featuresets")
    #Creating the tagged docuemtn feature sets
    featuresets = [(document_features(d), c) for (d,c) in documents] 
    print(featuresets[0])
    break
    print("created featuresets :", len(featuresets))
    #split the data into training data, and data to test the models on 
    trainnum = int(0.7*len(featuresets))
    testnum = int(0.28*len(featuresets))
    train_set, test_set = featuresets[:trainnum], featuresets[-testnum:]
    print("Train set:", len(train_set), "Test set:", len(test_set) )
    print("Training the set")
    #just figure stuff out man, its just me changing numbers to see if things work. Read the documentation
    LinearSVC_classifier = SklearnClassifier(SGDClassifier(alpha=0.00005, average=False, class_weight=None,
       early_stopping=False, epsilon=0.1, eta0=0.0, fit_intercept=True,
       l1_ratio=0.15, learning_rate='optimal', loss='log', max_iter=1000,
       n_iter_no_change=3, n_jobs=2, penalty=None, power_t=0.5,
       random_state=None, shuffle=True, tol=0.0001, validation_fraction=0.1,
       verbose=0, warm_start=False))
    #Train the model
    LinearSVC_classifier.train(train_set)
    #Test how good that model is
    LinearSVC_accuracy = nltk.classify.accuracy(LinearSVC_classifier, test_set)    
    print("Linear SVC classifier:", LinearSVC_accuracy)


    CNB_classifier = SklearnClassifier(ComplementNB(alpha=1.0, class_prior=None, fit_prior=True, norm=False))
    CNB_classifier.train(train_set)
    
    CNB_accuracy = nltk.classify.accuracy(CNB_classifier, test_set)
    print("CNB Classifier classifier:",CNB_accuracy)
    
    savename = "/SVC/" + name + ".pickle"
    save_class = open(savename, "wb")
    print("saving file")
    pickle.dump(LinearSVC_classifier, save_class)
    save_class.close()
    #Helpful, Nothelpful = FF.MostInformative(test_set, CNB_classifier, 5)

    #file2 = "L://Knowledge Management/_KM (Secure)/Inspec/Inspec2/Inspec 2 Development/GoldenCorpus/Subject/NotHelpful/" + name + ".csv"
    #Nothelpful.to_csv(file2, encoding='utf-8')        
    
    savename2 = "ComplementNB/" + name + ".pickle"
    save_class = open(savename2, "wb")
    print("saving file")
    pickle.dump(CNB_classifier, save_class)
    save_class.close()
    timetaken = (time.time() - start_time)
    print("--- %s seconds ---" % (timetaken))
    updatedetails = str(name) + "," + str(trainnum) + "," + str(LinearSVC_accuracy) + "," + str(CNB_accuracy) + "," + str(timetaken)
    with open(savedata, 'a+', encoding = 'utf-8') as f:    
        f.write("%s\n" % updatedetails)
    documents = []
    lines = []
    featuresets = []
    gc.collect()
    
print("--- %s seconds ---" % (time.time() - start_time2))