'''
Created on Feb 6, 2018

@author: David
'''
from nltk import ngrams
import codecs
import math

class LanguageModel():
    '''
    classdocs
    '''

    def __init__(self, uni,bi,tri,q):
        '''
        Constructor
        '''
        self.uni = self.__load(uni)
        self.bi = self.__load(bi)
        self.tri = self.__load(tri)
        self.quad = self.__load(q)
        self.triback = 10e-6
        self.biback = 10e-6
        self.uniback = 10e-4
        self.qback = 10e-6

    def __load(self, ffile):
        mdict = dict()
        with codecs.open(ffile, encoding="utf-8") as f:
            for l in f:
                if not l.strip():
                    continue
                w,r = l.split("\t")
                mdict[w] = float(r)
        return mdict
    
    def get_ngram_prob(self, word, n):
        prob = 1
        ngram_iter = ngrams(word,n)
        m = len(word)-n+1
        for ng in ngram_iter:
            ngs = "".join(ng)
            if n == 1:
                if ngs in self.uni.keys():
                    prob *= self.uni[ngs]
                else:
                    prob *= self.uniback
            if n == 2:
                if ngs in self.bi.keys():
                    prob *= self.bi[ngs]
                else:
                    prob *= self.biback
            if n == 3:
                if ngs in self.tri.keys():
                    prob *= self.tri[ngs]
                else:
                    prob *= self.triback
            if n == 4:
                if ngs in self.quad:
                    prob *= self.quad[ngs]
                else:
                    prob *= self.qback
        #print(word, prob, -math.log(prob))
        return prob/m*10e10 if prob != 1 else 0
    
    def get_unigram_prob(self, word):
        return self.get_ngram_prob(word, 1)
    
    def get_bigram_prob(self, word):
        return self.get_ngram_prob(word, 2)
    
    def get_trigram_prob(self, word):
        return self.get_ngram_prob(word, 3)
    def get_quadgram_prob(self, word):
        return self.get_ngram_prob(word, 4)
