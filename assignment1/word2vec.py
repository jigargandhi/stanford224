import numpy as np
from q1_softmax import softmax 
from q2_sigmoid import sigmoid, sigmoid_grad
from q2_gradcheck import gradcheck_naive
class Word2Vec:
    """Word2Vec models of skipgram and cbow with negative sampling"""
    def __init__(self, corpus, d = 10, window = 5, corpussize = 1000):
        self.path = corpus
        self.D = d
        self.V = np.zeros((corpussize, d))
        self.U = np.zeros((corpussize, d)) 
        
    def preprocess(self):
         "Use NLTK to remove punctuation and create a corpus"
         self.words =[]   
    
    def softmaxCostAndGradient(self, predicted, target, outputVectors):
        
        
        
        pass   
    
    


if __name__=="__main__":
    w = Word2Vec(corpus = "/utils/corpus/834-0.txt")
    
    