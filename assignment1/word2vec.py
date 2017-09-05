import numpy as np
from q1_softmax import softmax 
from q2_sigmoid import sigmoid, sigmoid_grad
from q2_gradcheck import gradcheck_naive
import re

class Word2Vec:
    """Word2Vec models of skipgram and cbow with negative sampling"""
    def __init__(self, corpus, d = 10, window = 3):
        self.path = corpus        
        self.words=[]
        self.freq ={}
        self.tokens=[]
        self.window = window
        self.tokensize = self.preprocess()
        self.V = np.random.rand(self.tokensize, d)
        self.U = np.random.rand(self.tokensize, d) 
        
        
    def preprocess(self):
         
         with open(self.path, 'r',encoding='utf8') as f:
             for line in f:
                 self.words+=[re.sub('[^a-z]','',x.lower()) for x in line.split()]
         #print(len(self.words))
         for wrd in self.words:
             if wrd in self.freq.keys():
                 self.freq[wrd]+=1
             else:
                 self.freq[wrd] = 1
         self.tokens = list(self.freq.keys())
         return len(self.freq.keys())
    
    def getWindowVectors(self, index):
        val = []
        arr= []
        if index < self.window:
            val =np.array((self.window, self.tokensize), dtype=float)
            arr = [i for i in range(index+1, index+self.window+1)]
        else:
            val = np.array((2*self.window, self.tokensize), dtype= float)
            arr = [i for i in range(index-self.window, index + self.window +1) if i != index]
        
        print(arr)
        for idx, val in enumerate(arr):
            # create a np array of row vectors of output
            print(self.getVector(self.words[val]).shape)
        return val
                    
        pass
    
    def getVector(self, word):
        index =self.tokens.index(word)
        x = np.zeros((self.tokensize,1))
        x[index] = 1
        return x.T
    
    def softmaxCostAndGradient(self, predicted, target, outputVectors):
        pass
    
    def sgd():
        pass
    
    


if __name__=="__main__":
    w = Word2Vec(corpus = "utils/corpus/834-0.txt")
    w.getWindowVectors(10)
    
    