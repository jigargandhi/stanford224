import numpy as np
import math
from q2_sigmoid import sigmoid, sigmoid_grad
from q2_gradcheck import gradcheck_naive
import re
class Unigram:
    def __init__(self, freqDist, index):
        self.freqDist= freqDist
        self.table=[]
        self.index=index
    
    def fillUnigramTable(self):
        length = len(self.freqDist)
        power = 0.75
        table_size = int(1e8)
        table = np.zeros(table_size, dtype=np.uint32)
        Z = sum([math.pow(e, power) for e in self.freqDist])
        p=0
        i=0
        for j, unigram in enumerate(self.freqDist):
            p+=float(math.pow(unigram,power))/Z
            while i<table_size and float(i)/table_size < p:
                table[i] = self.index.index(j)
                i+=1
        self.table= table
    
    def sample(self,count):
        indices = np.random.randint(low=0, high=len(self.table), size=count)
        return [self.table[i] for i in indices]
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
        self.unigram = Unigram(self.freq,self.tokens)
        
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
            arr = [i for i in range(index-self.window, index + self.window +1) if i != index]
        
        return arr
    
    def getVector(self, word):
        index =self.tokens.index(word)
        x = np.zeros((self.tokensize,1))
        x[index] = 1
        return x.T
    
    def softmaxCostAndGradient(self, idx, targetIdx, negativeSample):
        X = self.getVector(self.words[idx])
        V = self.V[idx,:] #d*1
        labels = [target]+negativeSample
        

            
        
    
    def train(self):
        for i in range(5000):
            for idx, val in enumerate(self.words):
                for window in self.getWindowIdx(idx):
                    cost, labels, gradUs, gradVs = self.softmaxCostAndGradient(idx, window, self.unigram.sample())
                self.U = self.U - self.rate* gradU
                self.V = self.V - self.rate*gradV
            
            if i%1000 == True:
                print ("Error: ",cost)
    
    def test(self):
        pass
    
    def softmax(x):
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

    def sigmoid(x):
        """
        Compute the sigmoid function for the input here.
    
        Arguments:
        x -- A scalar or numpy array.
    
        Return:
        s -- sigmoid(x)
        """
        s = x*-1
        s = 1/(1+np.exp(s))
        return s


    def sigmoid_grad(s):
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
    
    
    