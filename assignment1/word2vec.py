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
        self.b1 = np.random.rand(self.tokensize,d)
        self.U = np.random.rand(self.tokensize, d)
        self.b2 = np.random.rand(self.tokensize,d)
        self.rate = 0.005
        
        
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
        
        outputVectors =None
        for idx, val in enumerate(arr):
            if idx ==0:
                outputVectors = self.getVector(self.words[val])
            else:
                #find a better way to do it
                outputVectors = np.concatenate((outputVectors,self.getVector(self.words[val])),axis=0)
        return outputVectors
    
    def getVector(self, word):
        index =self.tokens.index(word)
        x = np.zeros((self.tokensize,1))
        x[index] = 1
        return x.T
    
    def softmaxCostAndGradient(self, idx, outputVectors, negativeLabels=[]):
        cost=None
        gradU = None
        gradV = None
        
        #feedforward
        vc= self.V[idx,:]
        uo= self.U[idx,:]
        #http://kb.timniven.com/?p=181
        sigmoid_uo_vc =sigmoid(uo.T.dot(vc)) 
        del_vc_1 = (1-sigmoid_uo_vc)*uo
        del_vc_2 = np.zeros_like(del_vc_1)
        del_uc_1 = (sigmoid_uo_vc-1)*vc
        del_uc_2 = np.zeros_like(del_vc_1)
        secondCost=0
        for i in negativeLabels:
            uk = self.U[i,:]
            sigm= sigmoid(-1*uk.T.dot(vc))
            secondCost += np.log(sigm)
            del_vc_2 += uk*(sigm+1)
            del_uc_2+= -1*(sigm-1)*vc
        
        del_vc = del_vc_1+ del_vc_2
        del_uc = del_uc_1+ del_uc_2
        cost = 0
        cost += np.log(sigmoid_uo_vc)+ secondCost       
        
        return cost, del_vc, del_uc
        
    
    def sgd(self):
        for idx, val in enumerate(self.words):
            cost, gradU, gradV = self.softmaxCostAndGradient(idx, self.getWindowVectors(idx))
            self.U = self.U + self.rate* gradU
            self.V = self.V +self.rate*gradV
    
    


if __name__=="__main__":
    w = Word2Vec(corpus = "utils/corpus/834-0.txt")
    w.getWindowVectors(10)
    
    