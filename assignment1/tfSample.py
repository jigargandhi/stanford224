import tensorflow as tf

import pandas as pd

import numpy as np
def softmax(x):
    

def loadGloveModel(gloveFile):
    print ("Loading Glove Model")
    f = open(gloveFile,'r',encoding='utf8')
    model = {}
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = [float(val) for val in splitLine[1:]]
        model[word] = embedding
    print ("Done.",len(model)," words loaded!")
    return model

model = loadGloveModel("utils/datasets/glove.6B.50d.txt")
frog = np.array(model["frog"])
cat = np.aray(model["cat"])
lion = np.array(model["lion"])

frog_lion = frog.T.dot(lion)
cat_lion = cat.T.dot(lion)


print("Frog and lion are closer", frog_lion)
print("Cat and lion are closer", cat_lion)


words = "I am talking about a pet."
average = np.zeros_like(np.array(model["dog"]))
for w in words.split():
    if w.lower() in model.keys():
        average +=np.array(model[w.lower()])

average = average/len(w)

predCat = average.T.dot(cat)
predCar = average.T.dot(np.array(model["car"]))

#probabilities
catProb = np.exp(predCat)/(np.exp(predCat)+np.exp(predCar))
carProb = np.exp(predCar)/(np.exp(predCat)+np.exp(predCar))

print("Cat: ",catProb,", Dog: ",carProb)