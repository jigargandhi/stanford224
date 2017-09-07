import numpy as np
import math
from q2_sigmoid import sigmoid, sigmoid_grad
from q2_gradcheck import gradcheck_naive
import re
import pickle
from pathlib import Path
class Unigram:
    def __init__(self, freqDist, index):
        self.freqDist= freqDist
        self.table=[]
        self.index=index
        self.fileName = "unigram.pkl"
    
    def fillUnigramTable(self):
        print("Filling Unigram table")
        
        if Path(self.fileName).exists():
            print("picking up from unigram.pkl file")
            with open(self.fileName,'rb') as f:
                self.table = pickle.load(f)
            return
        else:
            print("Calculating Manually")
        power = 0.75
        table_size = int(1e7)
        table = np.zeros(table_size, dtype=np.uint32)
        Z = sum([math.pow(self.freqDist[e], power) for e in self.freqDist])
        p=0
        i=0
        #stupidest way to fill unigram table
        #alternate way project the probability to a number , create an array of index and append the array
        for j, unigram in enumerate(self.freqDist):
            
            p+=float(math.pow(self.freqDist[unigram],power))/Z
            while i<table_size and float(i)/table_size < p:
                table[i] = self.index.index(unigram)
                i+=1
            
            if i%1000 ==0:
                print("Filling progress: ",i)
        self.table= table
        
        with open(self.fileName, 'wb') as f:
            pickle.dump(self.table, f)
    
    def sample(self,count):
        indices = np.random.randint(low=0, high=len(self.table), size=count)
        return [self.table[i]-1 for i in indices]
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
        self.rate = 0.05
        self.unigram = Unigram(self.freq,self.tokens)
        self.unigram.fillUnigramTable()
        self.K = 4
        
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
    
    def getWindowIdx(self, index):
        val = []
        arr= []
        if index < self.window:
            val =np.array((self.window, self.tokensize), dtype=float)
            arr = [i for i in range(index+1, index+self.window+1)]
        else:
            val = np.array((2*self.window, self.tokensize), dtype= float)
            arr = [i for i in range(index-self.window, index + self.window +1) if i != index and i <self.tokensize]
        
        return arr
    
    def getVector(self, word):
        index =self.tokens.index(word)
        x = np.zeros((self.tokensize,1))
        x[index] = 1
        return x.T
    
    def softmaxCostAndGradient(self, idx, targetIdx, negativeSample):
        if negativeSample ==None:
            return
        V = self.V[idx,:] #d*1
        labels = [targetIdx]+negativeSample
        
        directions = [1]+[-1 for i in negativeSample]
        
        cost = 0
        gradCenter = np.zeros_like(V)
        gradOutput = np.zeros_like(V)
        for idx, label in enumerate(labels):
            
            u_o_idx = self.U[label,:]
            
            uovc = u_o_idx.T.dot(V)
            print(V, u_o_idx, uovc)
            sigmoid = self.sigmoid(directions[idx]*uovc)
            print(sigmoid)
            delt_c = (sigmoid+1)*u_o_idx
            delt_o = (1-sigmoid)*V
            cost = cost + np.log(sigmoid)
            if idx==0:
                gradCenter = delt_c
                gradOutput = delt_o
            else:
                gradCenter -= delt_c
                gradOutput += delt_o
                
        return cost,labels, gradCenter, gradOutput
            
        
    
    def train(self):
        print("Training started")
        for i in range(1):
            print("Starting iteration")
            cost = 0
            for idx, val in enumerate(self.words):
                if idx > 15:
                    break
                for context in self.getWindowIdx(idx):
                    dataId = self.tokens.index(val)
                    neg = self.unigram.sample(self.K)
                    newNeg = [self.tokens.index(self.words[k]) for k in neg]
                    cost, labels, gradUs, gradVs = self.softmaxCostAndGradient(dataId, context, newNeg)
                    #print(gradUs.shape)
                    
                    self.U[labels,:] += self.rate*gradUs
                    self.V[labels,:] += self.rate*gradVs
                    
            self.saveWeights()
            print ("Error: ",cost)
            
        print("Training complete")
    
    def saveWeights(self):
        with open("U.pkl","wb") as f:
            pickle.dump(self.U,f)
        
        with open("V.pkl","wb") as f:
            pickle.dump(self.V, f)
    
    def test(self):
        pass
    
    def softmax(self,x):
        """Compute the softmax function for each row of the input x.
    
        It is crucial that this function is optimized for speed because
        it will be used frequently in later code. You might find numpy
        functions np.exp, np.sum, np.reshape, np.max, and numpy
        broadcasting useful for this task.
    
        Numpy broadcasting documentation:
        http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html
    
        You should also make sure that your code works for a single
        N-dimensional vector (treat the vector as a single row) and
        for M x N matrices. This may be useful for testing later. Also,
        make sure that the dimensions of the output match the input.
    
        You must implement the optimization in problem 1(a) of the
        written assignment!
    
        Arguments:
        x -- A N dimensional vector or M x N dimensional numpy matrix.
    
        Return:
        x -- You are allowed to modify x in-place
        """
        orig_shape = x.shape
    
        if len(x.shape) > 1:
            # Matrix
            ### YOUR CODE HERE
            m,n= x.shape
            x = x - np.max(x,axis = 1).reshape((m,1))
            x = np.exp(x)
            sum_x = np.sum(x, axis = 1)
            divisor = np.tile(sum_x, (n,1)).T
            x = x / divisor
            
            ### END YOUR CODE
        else:
            # Vector
            ### YOUR CODE HERE
            maxV = np.max(x)
            x = np.exp(x-maxV)
            x = np.divide(x, np.sum(x))
            ### END YOUR CODE
    
        assert x.shape == orig_shape
        return x

    def sigmoid(self, x):
        """
        Compute the sigmoid function for the input here.
    
        Arguments:
        x -- A scalar or numpy array.
    
        Return:
        s -- sigmoid(x)
        """
        
        s = x*-1
        s = 1/(1+np.exp(s))
        #print(x, s)
        return s


    def sigmoid_grad(self,s):
        """
        Compute the gradient for the sigmoid function here. Note that
        for this implementation, the input s should be the sigmoid
        function value of your original input x.
        Arguments:
            s -- A scalar or numpy array.
            
            Return:
                ds -- Your computed gradient.
                """
                
        ds = s*(1-s)
        return ds


if __name__=="__main__":
    w = Word2Vec(corpus = "utils/corpus/834-0.txt")
    w.train()
    
    v = np.array([ 1.9863067, 1.18012618 ,1.78814159 , 1.64008512 ,1.42083795 ,1.69617157, 1.61545756 ,0.7543406  ,1.29930119 ,1.08029599] )
    u = np.array([ 0.51840154  ,0.91489921,  0.56757563,  0.90155081,  0.83421002,  0.54491402,  0.91915876,  0.82957221,  0.37149862 , 0.143370539.46068469959])