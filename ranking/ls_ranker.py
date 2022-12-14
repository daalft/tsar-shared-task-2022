import pickle
from functools import cmp_to_key
from nltk import ngrams
import random

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
        with open(ffile, "r", encoding="utf-8") as f:
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

        return prob/m*10e10 if prob != 1 else 0
    
    def get_unigram_prob(self, word):
        return self.get_ngram_prob(word, 1)
    
    def get_bigram_prob(self, word):
        return self.get_ngram_prob(word, 2)
    
    def get_trigram_prob(self, word):
        return self.get_ngram_prob(word, 3)
    def get_quadgram_prob(self, word):
        return self.get_ngram_prob(word, 4)

class LanguageModelWrapper:
    def __init__(self, lm):
        self.lm = lm
    
    def get_freqs(self, word):
        pword = "^" + word + "$"
        uniprob = self.lm.get_unigram_prob(pword)
        biprob = self.lm.get_bigram_prob(pword)
        triprob = self.lm.get_trigram_prob(pword)
        quadprob = self.lm.get_quadgram_prob(pword)
        return [uniprob, biprob, triprob, quadprob]
    
    def vectorize(self, word):
        return self.get_freqs(word)

class LSRanker(object):
    def __init__(self, vectorizer=None, ml_algorithm=None, scaler=None):
        self.vectorizer = vectorizer
        self.ml_algorithm = ml_algorithm
        self.scaler = scaler
    
    def __custom_cmp(self, w1, w2):
        v1 = self.vectorizer.vectorize(w1)
        v2 = self.vectorizer.vectorize(w2)
        v3 = [v1 + v2]
        
        v3s = self.scaler.transform(v3)
        prediction = self.ml_algorithm.predict(v3s)
        if prediction == 0:
            return 1
        elif prediction == 1:
            return -1
        else:
            return 0
        
    def __custom_sort(self, list_to_sort):
        from itertools import permutations
        
        perms = permutations(list_to_sort, 2)
        r = {}
        for perm in perms:
            w1, w2 = perm
            v1 = self.vectorizer.vectorize(w1)
            v2 = self.vectorizer.vectorize(w2)
            v3 = [v1 + v2]
            v3s = self.scaler.transform(v3)
            pred = self.ml_algorithm.predict(v3s)
            if w1 not in r:
                r[w1] = 0
            if w2 not in r:
                r[w2] = 0
            if pred == 0:
                r[w1] += 3
                r[w2] -= 1
            else:
                r[w1] -= 1
                r[w2] += 3
        return [y[0] for y in sorted(r.items(), key=lambda x: x[1], reverse=True)]
        
        
    def rank(self, list_to_rank, custom_sort=False):
        if type(list_to_rank) != list:
            if type(list_to_rank) == tuple:
                list_to_rank = list(list_to_rank)
            elif type(list_to_rank) == set:
                list_to_rank = list(list_to_rank)
            else:
                raise Exception("Expected type <class 'list'> to rank. Received type {}!".format(type(list_to_rank)))
        if custom_sort:
            return self.__custom_sort(list_to_rank)
        return sorted(list_to_rank, key=cmp_to_key(self.__custom_cmp), reverse=True)
    
class IdentityScaler:
    def transform(self, x):
        return x


id_scaler = IdentityScaler()
vectorizer = pickle.load(open("/lmw-en.pickle", "rb"))
ml_algo = pickle.load(open("/models/en-vc-full.pickle", "rb"))

lsr = LSRanker(vectorizer, ml_algo, id_scaler)

tsar_gold = "/tsar2022_en_trial_gold.tsv"
out = open("/eval/tsar_eval_ranking_en_vc_us_custom.csv", "w", encoding="utf-8")
with open(tsar_gold, "r", encoding="utf-8") as f:
    for l in f:
        if not l.strip():
            continue
        sentence, cw, *candidates = l.rstrip().split("\t")
        random.shuffle(candidates)
        candidates = set(candidates)
        ranked = lsr.rank(candidates, True)
        out.write("{}\t{}\t{}\n".format(sentence, cw, "\t".join(ranked)))
out.close()